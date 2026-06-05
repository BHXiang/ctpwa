#include <Info.cuh>
#include <AmpGen.cuh>
#include <complex>
#include <iostream>
#include <memory>
#include <set>

// Helper: print spin as proper J (config stores 2J+1)
static inline void printJ(int spin) {
    if (spin % 2 != 0) std::cout << (spin - 1) / 2;
    else std::cout << spin - 1 << "/2";
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
                    if (it != config_resonances.end() && it->second.J == sc.spin_parity[0]) {
                        std::vector<std::pair<double,double>> ch;
                        for (const auto& c : it->second.channels)
                            if (c.size() >= 2) ch.emplace_back(c[0], c[1]);
                        rlist.emplace_back(rname, rc.intermediate, ip.spin, ip.parity,
                                           it->second.type, it->second.parameters, ch);
                        // Free theta params with readable names (preserves order)
                        if (!it->second.free.empty()) {
                            const auto& pnames = rlist.back().getOrderedParamNames();
                            for (const auto& pn : pnames)
                                resonance_param_names_.push_back(rname + "_" + pn);
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
                bool id_d, is_boson, p_break, is_bf;
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
                bool identical_d = false; bool is_boson = true;
                if (chain.symmetrize) {
                    const Particle *d1 = nullptr, *d2 = nullptr;
                    for (const auto& p : particles_)
                        { if (p.name == step.daughters[0]) d1 = &p; if (p.name == step.daughters[1]) d2 = &p; }
                    if (d1 && d2 && !d1->identical_group.empty() && d1->identical_group == d2->identical_group)
                        { identical_d = true; is_boson = !d1->is_fermion(); }
                }
                int maxL = global_max_l;
                Amp2BD amp2bd(spins, parities, identical_d, is_boson, maxL, step.p_break, step.is_bf);
                cas->addDecay(amp2bd, step.mother, step.daughters[0], step.daughters[1]);

                // Save step info for later registration with resonance name
                std::vector<SL> sl_list;
                for (const auto& sl : amp2bd.getSL()) sl_list.push_back(sl);
                step_infos.push_back({step.mother, step.daughters[0], step.daughters[1],
                    spins, parities, identical_d, is_boson, step.p_break, step.is_bf, sl_list});
            }

            auto slcombs = cas->getSLCombinations();

            // Print decay chain structure for this J^P combination
            std::cout << "  ";
            for (size_t si = 0; si < step_infos.size(); ++si) {
                if (si > 0) std::cout << ", ";
                const auto& info = step_infos[si];
                std::cout << info.mother << "("; printJ(info.spins[0]);
                std::cout << (info.parities[0] == 1 ? "+)" : info.parities[0] == -1 ? "-)" : ")");
                std::cout << "→" << info.d1 << "("; printJ(info.spins[1]);
                std::cout << (info.parities[1] == 1 ? "+)" : info.parities[1] == -1 ? "-)" : ")");
                std::cout << info.d2 << "("; printJ(info.spins[2]);
                std::cout << (info.parities[2] == 1 ? "+)" : info.parities[2] == -1 ? "-)" : ")");
                // Print SL list for this step
                std::cout << " {";
                for (size_t sli = 0; sli < info.sl_list.size(); ++sli) {
                    if (sli > 0) std::cout << " ";
                    std::cout << "(" << info.sl_list[sli].S << "," << info.sl_list[sli].L << ")";
                }
                std::cout << "}";
            }
            std::cout << std::endl;

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

            // Print resonances for each combination
            if (!res_combos.empty()) {
                std::cout << "  Resonances:";
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
                        + (info.p_break ? "_pb" : "") + (!info.is_bf ? "_nbf" : "")
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
                    // Build name with per-step SL: step1_SL(S,L)_step2_SL(S,L)...
                    std::string full_name;
                    for (size_t sni = 0; sni < slcomb.size() && sni < chain.decay_steps.size(); ++sni) {
                        if (!full_name.empty()) full_name += "_";
                        const auto& step = chain.decay_steps[sni];
                        const auto& sl = slcomb[sni];
                        full_name += replace_tag(step.mother) + "→"
                                  + replace_tag(step.daughters[0]) + "+"
                                  + replace_tag(step.daughters[1])
                                  + "_SL(" + std::to_string(sl.S) + ","
                                  + std::to_string(sl.L) + ")";
                    }
                    amplitude_names_.push_back(full_name);

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
                    s.label + "_SL("
                    + std::to_string(s.sl_list[sli].S) + ","
                    + std::to_string(s.sl_list[sli].L) + ")");
            }
        }
    }
}

void DecayInfo::print() const
{
    printf("=== DecayInfo ===\n");
    printf("Particles: %zu\n", particles_.size());
    for (const auto& p : particles_)
        printf("  %s J=%d P=%d mass=%.4f\n", p.name.c_str(), p.spin, p.parity, p.mass);

    printf("Chains: %zu\n", chains_info_.size());
    auto all_chains = config_parser_.getDecayChains();
    for (const auto& ch : all_chains) {
        std::string topo;
        for (const auto& step : ch.decay_steps) {
            if (!topo.empty()) topo += "_";
            topo += step.mother + "→" + step.daughters[0] + "+" + step.daughters[1];
        }
        printf("  %s: %s\n", ch.name.c_str(), topo.c_str());
    }
    printf("Amplitudes: %zu\n", amplitude_names_.size());
    for (size_t i = 0; i < amplitude_names_.size(); ++i)
        printf("  [%zu] %s\n", i, amplitude_names_[i].c_str());

    printf("Constraints (%zu):\n", constraints_.size());
    for (const auto& c : constraints_) {
        std::string names_str;
        for (size_t ni = 0; ni < c.names.size(); ++ni) {
            if (ni > 0) names_str += ", ";
            names_str += c.names[ni];
        }
        printf("  trans: [%s]", names_str.c_str());
        if (!c.values.empty())
            printf(" = (%.1f)", std::real(c.values[0]));
        printf("\n");
    }

    printf("Param names (%zu):\n", param_names_.size());
    // if (n_chain_free_after_trans_ > 0 && n_chain_free_after_trans_ < (int)param_names_.size())
    //     printf("  (chain params after trans: %d)\n", n_chain_free_after_trans_);
    for (size_t i = 0; i < param_names_.size(); ++i)
        printf("  [%zu] %s\n", i, param_names_[i].c_str());

    printf("Resonance theta params (%zu):\n", resonance_param_names_.size());
    for (size_t i = 0; i < resonance_param_names_.size(); ++i)
        printf("  [%zu] %s\n", i, resonance_param_names_[i].c_str());
}
