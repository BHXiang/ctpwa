#include <Info.cuh>
#include <AmpGen.cuh>
#include <iostream>
#include <memory>

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

void DecayInfo::initialize(const std::string&)
{
    particles_   = config_parser_.getParticles();
    constraints_ = config_parser_.getConstraints();

    auto chains = config_parser_.getDecayChains();
    const auto& config_resonances = config_parser_.getResonances();

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
                        // Free theta params
                        if (!it->second.free.empty()) {
                            int nf = 0;
                            for (int fi : it->second.free)
                                nf += (fi == -1) ? (int)it->second.parameters.size() : 1;
                            for (int pi = 0; pi < (int)it->second.parameters.size(); ++pi)
                                resonance_param_names_.push_back(rname + "_param" + std::to_string(pi));
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
                int maxL = config_parser_.getGlobalMaxL();
                Amp2BD amp2bd(spins, parities, identical_d, is_boson, maxL, step.p_break, step.is_bf);
                cas->addDecay(amp2bd, step.mother, step.daughters[0], step.daughters[1]);

                // Step key with full quantum numbers
                std::string step_key =
                    step.mother + ">" + step.daughters[0] + "," + step.daughters[1]
                    + "_J" + std::to_string(spins[0]) + "-" + std::to_string(spins[1]) + "-" + std::to_string(spins[2])
                    + "_P" + std::to_string(parities[0]) + "-" + std::to_string(parities[1]) + "-" + std::to_string(parities[2])
                    + (identical_d ? (is_boson ? "_idB" : "_idF") : "")
                    + (step.p_break ? "_pb" : "") + (!step.is_bf ? "_nbf" : "");
                std::string step_label = step.mother + "→" + step.daughters[0] + step.daughters[1];
                std::vector<SLKey> sl_keys;
                for (const auto& sl : amp2bd.getSL())
                    sl_keys.push_back({sl.S, sl.L});
                int step_idx = coupling_matrix_builder_.addStep(step_key, step_label, sl_keys);
                chain_step_map[chain.name].push_back(step_idx);
            }

            auto slcombs = cas->getSLCombinations();

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

            for (size_t ki = 0; ki < res_combos.size(); ++ki) {
                // res_name for amplitude identification, chain_key for coupling matrix
                std::string res_name = chain.name;
                std::string chain_key;
                for (const auto& rp : res_combos[ki]) {
                    res_name += "_" + rp.first + "_" + rp.second;
                    if (!chain_key.empty()) chain_key += "_";
                    chain_key += rp.second;  // resonance name only
                }

                for (size_t si = 0; si < slcombs.size(); ++si) {
                    const auto& slcomb = slcombs[si];
                    std::string full_name = res_name + "_SL";
                    for (const auto& sl : slcomb)
                        full_name += "_" + std::to_string(sl.S) + std::to_string(sl.L);
                    amplitude_names_.push_back(full_name);

                    // Step→SL mapping for coupling matrix
                    const auto& step_indices = chain_step_map[chain.name];
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

    // --- Build coupling matrix ---
    if (!amplitude_names_.empty()) {
        coupling_matrix_ = coupling_matrix_builder_.build();
        use_coupling_matrix_ = true;

        for (const auto& cn : coupling_matrix_.chain_names)
            param_names_.push_back("chain_" + cn);
        for (size_t si = 0; si < coupling_matrix_.steps.size(); ++si) {
            const auto& s = coupling_matrix_.steps[si];
            for (int sli = 1; sli < s.n_sl(); ++sli) {
                param_names_.push_back(
                    "step_" + s.label + "_SL("
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
    printf("Amplitudes: %zu\n", amplitude_names_.size());
    for (size_t i = 0; i < amplitude_names_.size(); ++i)
        printf("  [%zu] %s\n", i, amplitude_names_[i].c_str());

    printf("Param names (%zu):\n", param_names_.size());
    for (size_t i = 0; i < param_names_.size(); ++i)
        printf("  [%zu] %s\n", i, param_names_[i].c_str());

    printf("Resonance theta params (%zu):\n", resonance_param_names_.size());
    for (size_t i = 0; i < resonance_param_names_.size(); ++i)
        printf("  [%zu] %s\n", i, resonance_param_names_[i].c_str());
}
