#include <Info.cuh>
#include <AmpGen.cuh>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <set>
#include <string>

// Helper: print spin as proper J (config stores 2J+1)
static inline void printJ(int spin) {
    if (spin % 2 != 0) std::cout << (spin - 1) / 2;
    else std::cout << spin - 1 << "/2";
}

// Helper: 内部 2S+1 → 物理自旋字符串（"0"/"1"/"1/2"/"3/2"...）
static inline std::string physSpinStr(int twoS1) {
    if (twoS1 % 2 != 0) return std::to_string((twoS1 - 1) / 2);
    return std::to_string(twoS1 - 1) + "/2";
}

DecayInfo::DecayInfo(const std::string& config_file)
    : config_parser_(config_file)
{
    if (!config_parser_.isValid()) return;
    initialize(config_file);
    initialized_ = true;
}

DecayInfo::DecayInfo(const ConfigParser& parser)
    : config_parser_()  // dummy, data comes from parser arg
{
    if (!parser.isValid()) return;
    particles_   = parser.getParticles();
    constraints_ = parser.getConstraints();
    buildDecayChains(parser.getDecayChains(),
                     parser.getResonances(),
                     parser.getGlobalMaxL());
    initialized_ = true;
}

void DecayInfo::initialize(const std::string&)
{
    particles_   = config_parser_.getParticles();
    constraints_ = config_parser_.getConstraints();
    buildDecayChains(config_parser_.getDecayChains(),
                     config_parser_.getResonances(),
                     config_parser_.getGlobalMaxL());
}

void DecayInfo::buildDecayChains(
    const std::vector<DecayChainConfig>& chains,
    const std::map<std::string, ResonanceConfig>& config_resonances,
    int global_max_l)
{

    for (const auto& chain : chains) {
        ChainInfo info;
        info.name = chain.name;

        // --- Build intermediate_resonance_map & intermediate_combs ---
        std::map<std::pair<std::string, std::vector<int>>, std::vector<Resonance>>
            intermediate_resonance_map;
        std::vector<std::vector<Particle>> intermediate_particles;

        for (const auto& rc : chain.resonance_chains) {
            std::vector<Particle> particles_jp;
            for (const auto& sc : rc.spin_chains) {
                Particle ip = {rc.intermediate, (int)sc.spin_parity[0], (int)sc.spin_parity[1], -1.0};
                std::vector<Resonance> rlist;
                for (const auto& rname : sc.resonances) {
                    auto it = config_resonances.find(rname);
                    // J 哨兵（-1，Resonances 段未写 J）→ 跳过一致性过滤，
                    // 量子数完全由 intermediates 的 [J, P] 决定
                    if (it != config_resonances.end()
                        && (it->second.J < 0 || it->second.J == sc.spin_parity[0])) {
                        std::vector<std::pair<double,double>> ch;
                        for (const auto& c : it->second.channels)
                            if (c.size() >= 2) ch.emplace_back(c[0], c[1]);
                        // 势垒因子: 把该共振态作为母粒子的衰变步的解析后最终值
                        // （三级作用域已在 Config::resolveStepBF 决议到 step）
                        // 注入 options["d"]/options["has_bf"]，供 addBlock 构建符号微分 aux。
                        std::map<std::string, std::string> ropts = it->second.options;
                        for (const auto& step : chain.decay_steps) {
                            if (step.mother == rc.intermediate) {
                                ropts["d"] = std::to_string(step.bf_d);
                                ropts["has_bf"] = step.has_bf ? "1" : "0";
                                break;
                            }
                        }
                        rlist.emplace_back(rname, rc.intermediate, ip.spin, ip.parity,
                                           it->second.type, it->second.parameters, ch,
                                           ropts);
                        // Free theta params with readable names
                        // Custom 模型: 参数默认全部自由（params 列表即声明自由参数）
                        if (it->second.type == "custom" || it->second.type == "Custom") {
                            const auto& pnames = rlist.back().getOrderedParamNames();
                            for (size_t pi = 0; pi < pnames.size(); ++pi)
                                resonance_param_names_.push_back(rname + "_" + pnames[pi]);
                        } else if (!it->second.free.empty()) {
                            const auto& pnames = rlist.back().getOrderedParamNames();
                            bool all_free = (it->second.free.size() == 1 && it->second.free[0] == -1);
                            for (size_t pi = 0; pi < pnames.size(); ++pi) {
                                if (all_free || std::find(it->second.free.begin(), it->second.free.end(), (int)pi) != it->second.free.end())
                                    resonance_param_names_.push_back(rname + "_" + pnames[pi]);
                            }
                        }
                    }
                }
                intermediate_resonance_map[{rc.intermediate, {(int)sc.spin_parity[0], (int)sc.spin_parity[1]}}] = rlist;
                particles_jp.push_back(ip);
            }
            intermediate_particles.push_back(particles_jp);
        }
        info.intermediate_resonance_map = intermediate_resonance_map;

        // Cartesian product of intermediate J^P choices
        std::vector<std::vector<Particle>> intermediate_combs = {{}};
        for (const auto& plist : intermediate_particles) {
            decltype(intermediate_combs) temp;
            for (const auto& comb : intermediate_combs)
                for (const auto& p : plist) {
                    auto nc = comb; nc.push_back(p); temp.push_back(nc);
                }
            intermediate_combs = std::move(temp);
        }
        info.intermediate_combs = intermediate_combs;

        // --- Process each intermediate J^P combination ---
        std::map<std::string, std::vector<int>> chain_step_map;

        for (const auto& comb : intermediate_combs) {
            auto cas = std::make_shared<AmpCasDecay>(particles_);

            // Save step info locally (without resonance name yet)
            struct StepInfo {
                std::string mother, d1, d2;
                std::array<int,3> spins, parities;
                bool id_d, is_boson, p_break, has_bf;
                double bf_d;
                std::vector<SL> sl_list;
            };
            std::vector<StepInfo> step_infos;

            for (const auto& step : chain.decay_steps) {
                std::array<int, 3> spins = {0};
                std::array<int, 3> parities = {0};
                for (const auto& p : particles_) {
                    if (p.name == step.mother) { spins[0] = p.spin; parities[0] = p.parity; }
                    for (size_t i = 0; i < step.daughters.size(); ++i)
                        if (p.name == step.daughters[i]) { spins[i+1] = p.spin; parities[i+1] = p.parity; }
                }
                for (const auto& res_jp : comb) {
                    if (res_jp.name == step.mother) { spins[0] = res_jp.spin; parities[0] = res_jp.parity; }
                    for (size_t i = 0; i < step.daughters.size(); ++i)
                        if (res_jp.name == step.daughters[i]) { spins[i+1] = res_jp.spin; parities[i+1] = res_jp.parity; }
                }
                // 全同粒子由 Constraints.identical 自动检测（无需手动开关）
                bool identical_d = false; bool is_boson = true;
                const Particle *d1 = nullptr, *d2 = nullptr;
                for (const auto& p : particles_)
                    { if (p.name == step.daughters[0]) d1 = &p; if (p.name == step.daughters[1]) d2 = &p; }
                if (d1 && d2 && !d1->identical_group.empty() && d1->identical_group == d2->identical_group)
                    { identical_d = true; is_boson = !d1->is_fermion(); }
                // 分波白名单: [[S, L], ...] → set<pair<int,int>>
                std::set<std::pair<int, int>> sl_filter;
                for (const auto& sf : step.sl_filter)
                    if (sf.size() >= 2) sl_filter.insert({ sf[0], sf[1] });
                int maxL = global_max_l;
                Amp2BD amp2bd(spins, parities, identical_d, is_boson, maxL, step.p_break, step.has_bf,
                              step.bf_d, std::move(sl_filter));
                cas->addDecay(amp2bd, step.mother, step.daughters[0], step.daughters[1]);

                // Save step info for later registration with resonance name
                std::vector<SL> sl_list;
                for (const auto& sl : amp2bd.getSL()) sl_list.push_back(sl);
                step_infos.push_back({step.mother, step.daughters[0], step.daughters[1],
                    spins, parities, identical_d, is_boson, step.p_break, step.has_bf, step.bf_d, sl_list});
            }

            auto slcombs = cas->getSLCombinations();

            // Print decay chain structure for this J^P combination
            // （默认关闭, 5 体全模型会刷屏; CTPWA_VERBOSE_CHAINS=1 打开调试）
            if (getenv("CTPWA_VERBOSE_CHAINS")) {
                for (size_t si = 0; si < step_infos.size(); ++si) {
                    if (si > 0) std::cout << ", ";
                    const auto& info = step_infos[si];
                    std::cout << info.mother << "("; printJ(info.spins[0]);
                    std::cout << (info.parities[0] == 1 ? "+)" : info.parities[0] == -1 ? "-)" : ")");
                    std::cout << "→" << info.d1 << "("; printJ(info.spins[1]);
                    std::cout << (info.parities[1] == 1 ? "+)" : info.parities[1] == -1 ? "-)" : ")");
                    std::cout << info.d2 << "("; printJ(info.spins[2]);
                    std::cout << (info.parities[2] == 1 ? "+)" : info.parities[2] == -1 ? "-)" : ")");
                    // Print SL list for this step（用户侧: LS 序 + 物理自旋）
                    std::cout << " {";
                    for (size_t sli = 0; sli < info.sl_list.size(); ++sli) {
                        if (sli > 0) std::cout << " ";
                        std::cout << "(" << info.sl_list[sli].L << ",";
                        printJ(info.sl_list[sli].S);
                        std::cout << ")";
                    }
                    std::cout << "}";
                }
                std::cout << std::endl;
            }

            // Build resonance combinations
            std::vector<std::vector<std::pair<std::string,std::string>>> res_combos = {{}};
            for (const auto& p : comb) {
                auto key = std::make_pair(p.name, std::vector<int>{p.spin, p.parity});
                const auto& rlist = intermediate_resonance_map[key];
                decltype(res_combos) temp;
                for (const auto& rc : res_combos)
                    for (const auto& r : rlist) {
                        auto nc = rc; nc.push_back({r.getTag(), r.getName()}); temp.push_back(nc);
                    }
                res_combos = std::move(temp);
            }

            // Print resonances for each combination（CTPWA_VERBOSE_CHAINS=1 时打开）
            if (getenv("CTPWA_VERBOSE_CHAINS") && !res_combos.empty()) {
                std::cout << "Res:";
                for (size_t ki = 0; ki < res_combos.size(); ++ki) {
                    std::cout << " {";
                    for (size_t rj = 0; rj < res_combos[ki].size(); ++rj) {
                        if (rj > 0) std::cout << ", ";
                        std::cout << res_combos[ki][rj].second;
                    }
                    std::cout << "}";
                }
                std::cout << std::endl;
            }

            for (size_t ki = 0; ki < res_combos.size(); ++ki) {
                // Build readable topology name
                auto replace_tag = [&](const std::string& name) -> std::string {
                    for (const auto& rp : res_combos[ki])
                        // if (rp.first == name) return name + "[" + rp.second + "]";
                        if (rp.first == name) return rp.second;
                    return name;
                };
                std::string topo_name;
                for (const auto& step : chain.decay_steps) {
                    if (!topo_name.empty()) topo_name += "_";
                    topo_name += replace_tag(step.mother) + "→"
                              + replace_tag(step.daughters[0]) + "+"
                              + replace_tag(step.daughters[1]);
                }

                // res_name = topology (human-readable amplitude prefix)
                std::string res_name = topo_name;

                // chain_key: chain.name + simplified key for sharing/trans
                // Same intermediate+resonance in different CP channels share chain param
                std::string chain_key = chain.name;
                for (const auto& rp : res_combos[ki])
                    chain_key += "_" + rp.first + "[" + rp.second + "]";

                // chain display name = topology (for param_names_ output)
                chain_display_map_[chain_key] = topo_name;

                // --- Register steps WITH resonance name (key fix) ---
                std::vector<int> step_indices;
                for (const auto& info : step_infos) {
                    // Find resonance for this step's intermediate (mother or daughter)
                    std::string rname_for_step;
                    for (const auto& rp : res_combos[ki]) {
                        if (rp.first == info.mother || rp.first == info.d1 || rp.first == info.d2) {
                            rname_for_step = rp.second; break;
                        }
                    }
                    std::string step_key = chain.name + "___"  // chain identity
                        + info.mother + ">" + info.d1 + "," + info.d2
                        + "_J" + std::to_string(info.spins[0]) + "-" + std::to_string(info.spins[1]) + "-" + std::to_string(info.spins[2])
                        + "_P" + std::to_string(info.parities[0]) + "-" + std::to_string(info.parities[1]) + "-" + std::to_string(info.parities[2])
                        + (info.id_d ? (info.is_boson ? "_idB" : "_idF") : "")
                        + (info.p_break ? "_pb" : "") + (!info.has_bf ? "_nbf" : "")
                        + "_R" + rname_for_step;
                    auto step_label_r = [&](const std::string& nm) -> std::string {
                        for (const auto& rp : res_combos[ki])
                            if (rp.first == nm) return rp.second;
                        return nm;
                    };
                    std::string step_label = step_label_r(info.mother) + "→"
                        + step_label_r(info.d1) + "+" + step_label_r(info.d2);
                        // + "(" + chain.name + ")";
                    std::vector<SLKey> sl_keys;
                    for (const auto& sl : info.sl_list) sl_keys.push_back({sl.S, sl.L});
                    int s_idx = coupling_matrix_builder_.addStep(step_key, step_label, sl_keys);
                    step_indices.push_back(s_idx);
                }

                nsl_vectors_.push_back(static_cast<int>(slcombs.size()));

                for (size_t si = 0; si < slcombs.size(); ++si) {
                    const auto& slcomb = slcombs[si];
                    // Build name with per-step LS: step1_LS(L,S)_step2_LS(L,S)...（S 物理自旋）
                    std::string full_name;
                    for (size_t sni = 0; sni < slcomb.size() && sni < chain.decay_steps.size(); ++sni) {
                        if (!full_name.empty()) full_name += "_";
                        const auto& step = chain.decay_steps[sni];
                        const auto& sl = slcomb[sni];
                        full_name += replace_tag(step.mother) + "→"
                                  + replace_tag(step.daughters[0]) + "+"
                                  + replace_tag(step.daughters[1])
                                  + "_LS(" + std::to_string(sl.L) + ","
                                  + physSpinStr(sl.S) + ")";
                    }
                    amplitude_names_.push_back(full_name);
                    info.amplitude_names.push_back(full_name);

                    // Step→SL mapping for coupling matrix
                    std::vector<std::pair<int,int>> step_sl_pairs;
                    for (size_t ni = 0; ni < slcomb.size() && ni < step_indices.size(); ++ni) {
                        int si_step = step_indices[ni];
                        const auto& sl = slcomb[ni];
                        const auto& sl_list = coupling_matrix_builder_.getSteps()[si_step].sl_list;
                        int sl_idx = -1;
                        for (size_t ssi = 0; ssi < sl_list.size(); ++ssi)
                            if (sl_list[ssi].S == sl.S && sl_list[ssi].L == sl.L)
                                { sl_idx = (int)ssi; break; }
                        step_sl_pairs.push_back({si_step, sl_idx});
                    }
                    coupling_matrix_builder_.addAmplitude(
                        (int)amplitude_names_.size() - 1, chain_key, step_sl_pairs);
                }
                resonance_names_.push_back(res_name);
            }
        }
        chains_info_.push_back(info);
    }

    // --- Build coupling matrix & apply trans constraints ---
    if (!amplitude_names_.empty()) {
        // Build trans data from constraints
        std::vector<std::vector<std::string>> trans_names;
        std::vector<std::complex<double>> trans_vals;
        for (const auto& c : constraints_) {
            if (c.type == "trans") {
                trans_names.push_back(c.names);
                trans_vals.push_back(c.values.empty()
                    ? std::complex<double>(1.0, 0.0) : c.values[0]);
            }
        }
        coupling_matrix_ = coupling_matrix_builder_.buildWithTrans(
            trans_names, trans_vals);
        use_coupling_matrix_ = true;

        // Build param_names_ from result (only active params, readable names)
        for (const auto& cn : coupling_matrix_.chain_names) {
            auto dit = chain_display_map_.find(cn);
            param_names_.push_back(dit != chain_display_map_.end()
                ? dit->second : cn);
        }
        n_chain_free_after_trans_ = static_cast<int>(coupling_matrix_.chain_names.size());

        for (size_t si = 0; si < coupling_matrix_.steps.size(); ++si) {
            const auto& s = coupling_matrix_.steps[si];
            if (s.first_free_idx < 0) continue; // folded or all-fixed
            for (int sli = 1; sli < s.n_sl(); ++sli) {
                param_names_.push_back(
                    s.label + "_LS("
                    + std::to_string(s.sl_list[sli].L) + ","
                    + physSpinStr(s.sl_list[sli].S) + ")");
            }
        }
    }
}

// 内部 {2S+1, L} 白名单 → 用户侧文本 "(L,S物理)"
static std::string slFilterText(const std::vector<std::vector<int>>& f)
{
    if (f.empty()) return "";
    std::string s = " {";
    for (size_t i = 0; i < f.size(); ++i) {
        if (i) s += " ";
        s += "(" + std::to_string(f[i][1]) + "," + physSpinStr(f[i][0]) + ")";
    }
    s += "}";
    return s;
}

void DecayInfo::summary() const
{
    printf("=== DecayInfo ===\n");
    printf("Particles: %zu\n", particles_.size());
    for (const auto& p : particles_)
        printf("  %s J=%s P=%d mass=%.4f\n", p.name.c_str(),
               physSpinStr(p.spin).c_str(), p.parity, p.mass);
    size_t n_ec = 0;
    for (const auto& c : chains()) n_ec += (size_t)c.counts()[2];
    printf("Chains: %zu | Amplitudes: %zu | Exact chains: %zu\n",
           chains_info_.size(), amplitude_names_.size(), n_ec);
    std::set<std::string> unique_theta;
    for (const auto& n : resonance_param_names_) unique_theta.insert(n);
    printf("Constraints: %zu | Param names: %zu | Resonance theta: %zu (unique %zu)\n",
           constraints_.size(), param_names_.size(),
           resonance_param_names_.size(), unique_theta.size());
}

void DecayInfo::print(int level) const
{
    summary();
    if (level <= 0) return;
    auto cvs = chains();
    printf("Chains (%zu):\n", cvs.size());
    for (size_t ci = 0; ci < cvs.size(); ++ci) {
        const auto& c = cvs[ci];
        const auto cnt = c.counts();
        printf("  [%zu] %s: %s  (ints=%d res=%d chains=%d waves=%d)\n",
               ci, c.name.c_str(), c.topology.c_str(),
               cnt[0], cnt[1], cnt[2], cnt[3]);
        if (level >= 2) {
            for (const auto& s : c.steps) printf("      step %s\n", s.c_str());
            for (const auto& s : c.intermediates) printf("      int  %s\n", s.c_str());
            auto ec = c.exactchains();
            printf("      Exact chains (%zu):\n", ec.size());
            for (const auto& s : ec) printf("        %s\n", s.c_str());
        }
    }
    if (level >= 2) {
        printf("Constraints (%zu):\n", constraints_.size());
        for (const auto& c : constraints_) {
            std::string names_str;
            for (size_t ni = 0; ni < c.names.size(); ++ni) {
                if (ni > 0) names_str += ", ";
                names_str += c.names[ni];
            }
            printf("  trans: [%s]", names_str.c_str());
            if (!c.values.empty()) printf(" = (%.1f)", std::real(c.values[0]));
            printf("\n");
        }
    }
    if (level >= 3) {
        printf("Amplitudes (%zu):\n", amplitude_names_.size());
        for (size_t i = 0; i < amplitude_names_.size(); ++i)
            printf("  [%zu] %s\n", i, amplitude_names_[i].c_str());
        printf("Param names (%zu):\n", param_names_.size());
        for (size_t i = 0; i < param_names_.size(); ++i)
            printf("  [%zu] %s\n", i, param_names_[i].c_str());
    }
}

std::vector<ChainView> DecayInfo::chains() const
{
    std::vector<ChainView> out;
    const auto& dcs = config_parser_.getDecayChains();
    for (size_t ci = 0; ci < dcs.size() && ci < chains_info_.size(); ++ci) {
        const auto& dc = dcs[ci];
        const auto& info = chains_info_[ci];
        ChainView cv;
        cv.name = dc.name;
        {
            std::string topo;
            for (const auto& st : dc.decay_steps) {
                if (!topo.empty()) topo += "_";
                topo += st.mother + "→" + st.daughters[0] + "+" + st.daughters[1];
            }
            cv.topology = topo;
        }
        cv.exact_chain_strings = config_parser_.getExactChainStrings(dc);
        cv.amplitude_names = info.amplitude_names;
        for (const auto& st : dc.decay_steps) {
            cv.steps.push_back(st.mother + "->" + st.daughters[0]
                               + "+" + st.daughters[1]
                               + slFilterText(st.sl_filter));
        }
        for (const auto& rc : dc.resonance_chains) {
            std::string s = rc.intermediate + ":";
            for (const auto& sc : rc.spin_chains) {
                s += " [" + physSpinStr(sc.spin_parity[0]) + ","
                   + std::to_string(sc.spin_parity[1]) + "]:";
                for (const auto& r : sc.resonances) {
                    s += " " + r;
                    ++cv.n_resonances;
                }
            }
            cv.intermediates.push_back(s);
        }
        out.push_back(std::move(cv));
    }
    return out;
}

std::vector<std::string> DecayInfo::exactchains(
    int chain, const std::string& containing) const
{
    if (chain < 0) {
        std::vector<std::string> out;
        const auto& dcs = config_parser_.getDecayChains();
        for (size_t i = 0; i < dcs.size(); ++i) {
            auto v = config_parser_.getExactChainStrings(dcs[i], containing);
            out.insert(out.end(), v.begin(), v.end());
        }
        return out;
    }
    return config_parser_.getExactChainStrings(chain, containing);
}

std::vector<std::string> DecayInfo::amplitudes(
    int chain, const std::string& resonance) const
{
    const std::vector<std::string>* names = nullptr;
    if (chain < 0) names = &amplitude_names_;
    else if ((size_t)chain < chains_info_.size()) names = &chains_info_[chain].amplitude_names;
    else return {};
    if (resonance.empty()) return *names;
    std::vector<std::string> out;
    for (const auto& n : *names)
        if (n.find(resonance) != std::string::npos) out.push_back(n);
    return out;
}

void DecayInfo::printExactChains(const std::string& containing) const
{
    auto ec = exactchains(-1, containing);
    printf("# exact chains: %zu\n", ec.size());
    for (const auto& s : ec) printf("%s\n", s.c_str());
}

// ---- ChainView ----
std::vector<std::string> ChainView::exactchains(const std::string& containing) const
{
    if (containing.empty()) return exact_chain_strings;
    std::vector<std::string> out;
    for (const auto& s : exact_chain_strings)
        if (s.find(containing) != std::string::npos) out.push_back(s);
    return out;
}

std::vector<int> ChainView::counts() const
{
    std::vector<int> out(4, 0);
    out[0] = (int)intermediates.size();
    out[1] = n_resonances;
    out[2] = (int)exact_chain_strings.size();
    out[3] = (int)amplitude_names.size();
    return out;
}

void ChainView::print() const
{
    auto cnt = counts();
    printf("Chain: %s  (ints=%d res=%d chains=%d waves=%d)\n",
           name.c_str(), cnt[0], cnt[1], cnt[2], cnt[3]);
    for (const auto& s : steps) printf("  step %s\n", s.c_str());
    for (const auto& s : intermediates) printf("  int  %s\n", s.c_str());
    auto ec = exactchains();
    printf("  Exact chains (%zu):\n", ec.size());
    for (const auto& s : ec) printf("    %s\n", s.c_str());
    printf("  Amplitudes (%zu):\n", amplitude_names.size());
    for (const auto& s : amplitude_names) printf("    %s\n", s.c_str());
}
