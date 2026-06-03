#include <Info.cuh>
#include <AmpGen.cuh>
#include <iostream>

DecayInfo::DecayInfo(const std::string& config_file)
    : config_parser_(config_file)
{
    if (!config_parser_.isValid()) {
        std::cerr << "DecayInfo: invalid config " << config_file << std::endl;
        return;
    }
    initialize(config_file);
    initialized_ = true;
}

void DecayInfo::initialize(const std::string&)
{
    particles_   = config_parser_.getParticles();
    constraints_ = config_parser_.getConstraints();

    // --- Build decay chain structure ---
    auto chains = config_parser_.getDecayChains();
    const auto& config_resonances = config_parser_.getResonances();

    auto resonance_free = [&](const std::string& rname, std::string& model, int& nparams) {
        auto it = config_resonances.find(rname);
        if (it == config_resonances.end()) return false;
        model = it->second.type;
        nparams = (int)it->second.parameters.size();
        return !it->second.free.empty();
    };

    for (const auto& chain : chains) {
        ChainInfo info;
        info.name = chain.name;

        // Build cas for this chain. Need spins/parities for ALL particles including intermediates.
        std::map<std::string, std::pair<int,int>> spin_parity_map;
        for (const auto& p : particles_) spin_parity_map[p.name] = {p.spin, p.parity};
        // Add intermediate particle spins from resonance chains
        for (const auto& rc : chain.resonance_chains) {
            for (const auto& sc : rc.spin_chains) {
                spin_parity_map[rc.intermediate] = {(int)sc.spin_parity[0], (int)sc.spin_parity[1]};
            }
        }

        auto cas = std::make_shared<AmpCasDecay>(particles_);
        for (const auto& step : chain.decay_steps) {
            std::array<int, 3> spins = {0, 0, 0};
            std::array<int, 3> parities = {0, 0, 0};
            auto it_m = spin_parity_map.find(step.mother);
            if (it_m != spin_parity_map.end()) { spins[0] = it_m->second.first; parities[0] = it_m->second.second; }
            for (size_t di = 0; di < step.daughters.size(); ++di) {
                auto it_d = spin_parity_map.find(step.daughters[di]);
                if (it_d != spin_parity_map.end()) { spins[di+1] = it_d->second.first; parities[di+1] = it_d->second.second; }
            }
            bool identical_d = false; bool is_boson = true;
            if (chain.symmetrize) {
                const Particle *d1=nullptr, *d2=nullptr;
                for (const auto& p : particles_) {
                    if (p.name == step.daughters[0]) d1 = &p;
                    if (p.name == step.daughters[1]) d2 = &p;
                }
                if (d1 && d2 && !d1->identical_group.empty() && d1->identical_group == d2->identical_group) {
                    identical_d = true; is_boson = !d1->is_fermion();
                }
            }
            cas->addDecay(Amp2BD(spins, parities, identical_d, is_boson, config_parser_.getGlobalMaxL(),
                                 step.p_break, step.is_bf),
                          step.mother, step.daughters[0], step.daughters[1]);
        }
        auto slcombs_all = cas->getSLCombinations();

        // Per-chain: map intermediate particles → resonances
        for (const auto& rc : chain.resonance_chains) {
            std::vector<Particle> int_particles;
            for (const auto& sc : rc.spin_chains) {
                Particle ip = {rc.intermediate, (int)sc.spin_parity[0], (int)sc.spin_parity[1], -1.0};
                std::vector<Resonance> res_list;
                for (const auto& rname : sc.resonances) {
                    auto it = config_resonances.find(rname);
                    if (it != config_resonances.end() && it->second.J == sc.spin_parity[0]) {
                        std::vector<std::pair<double,double>> ch;
                        for (const auto& c : it->second.channels)
                            if (c.size() >= 2) ch.emplace_back(c[0], c[1]);
                        res_list.emplace_back(rname, rc.intermediate,
                            (int)sc.spin_parity[0], (int)sc.spin_parity[1],
                            it->second.type, it->second.parameters, ch);
                    }
                }
                info.intermediate_resonance_map[{rc.intermediate, {(int)sc.spin_parity[0], (int)sc.spin_parity[1]}}] = res_list;
                int_particles.push_back(ip);
            }
            // Build intermediate_combs (Cartesian product)
            if (info.intermediate_combs.empty()) info.intermediate_combs.push_back({});
            std::vector<std::vector<Particle>> new_combs;
            for (const auto& comb : info.intermediate_combs)
                for (const auto& p : int_particles) { auto nc = comb; nc.push_back(p); new_combs.push_back(nc); }
            info.intermediate_combs = new_combs;
        }

        // Register decay steps
        std::map<int,int> step_idx_map;
        if (!slcombs_all.empty()) {
            for (size_t ni = 0; ni < chain.decay_steps.size(); ++ni) {
                const auto& st = chain.decay_steps[ni];
                std::string key = st.mother + "->" + st.daughters[0] + "+" + st.daughters[1];
                std::string label = st.mother + "->" + st.daughters[0] + st.daughters[1];
                std::vector<SLKey> sl_list;
                for (size_t si = 0; si < slcombs_all.size(); ++si) {
                    bool found = false;
                    for (size_t ni2 = 0; ni2 < slcombs_all[si].size(); ++ni2) {
                        if (ni2 == ni) { sl_list.push_back({slcombs_all[si][ni2].S, slcombs_all[si][ni2].L}); found = true; break; }
                    }
                    if (found) break;
                }
                if (sl_list.empty()) {
                    for (const auto& sl : slcombs_all[0])
                        sl_list.push_back({sl.S, sl.L});
                }
                step_idx_map[(int)ni] = coupling_matrix_builder_.addStep(key, label, sl_list);
            }
        }

        // Build amplitudes and coupling matrix
        for (const auto& comb : info.intermediate_combs) {
            // Build resonance combinations
            std::vector<std::vector<std::pair<std::string,std::string>>> resonance_combos = {{}};
            for (const auto& p : comb) {
                auto key = std::make_pair(p.name, std::vector<int>{p.spin, p.parity});
                const auto& rlist = info.intermediate_resonance_map[key];
                decltype(resonance_combos) temp;
                for (const auto& rc : resonance_combos)
                    for (const auto& r : rlist) {
                        auto nc = rc; nc.push_back({r.getTag(), r.getName()}); temp.push_back(nc);
                    }
                resonance_combos = std::move(temp);
            }

            std::vector<std::vector<SL>> slcombs; // need to rebuild per-chain SL list
            // Use cas SL combinations
            for (const auto& sl : slcombs_all) slcombs.push_back(sl);

            for (size_t ki = 0; ki < resonance_combos.size(); ++ki) {
                std::string res_name = chain.name;
                std::string chain_key;  // resonance-only key for coupling matrix
                for (const auto& rp : resonance_combos[ki]) {
                    res_name += "_" + rp.first + "_" + rp.second;
                    if (!chain_key.empty()) chain_key += "_";
                    chain_key += rp.second;  // resonance name
                }

                for (const auto& slcomb : slcombs) {
                    std::string full_name = res_name + "_SL";
                    for (const auto& sl : slcomb)
                        full_name += "_" + std::to_string(sl.S) + std::to_string(sl.L);
                    amplitude_names_.push_back(full_name);

                    // Register with coupling matrix
                    std::vector<std::pair<int,int>> step_sl_pairs;
                    for (size_t ni = 0; ni < slcomb.size() && ni < chain.decay_steps.size(); ++ni) {
                        auto sit = step_idx_map.find((int)ni);
                        if (sit != step_idx_map.end()) {
                            int sl_idx = 0;
                            step_sl_pairs.push_back({sit->second, sl_idx});
                        }
                    }
                    coupling_matrix_builder_.addAmplitude(
                        (int)amplitude_names_.size() - 1, chain_key, step_sl_pairs);
                }
                resonance_names_.push_back(res_name);

                // Resonance param names
                for (const auto& rp : resonance_combos[ki]) {
                    std::string model; int np;
                    if (resonance_free(rp.second, model, np)) {
                        for (int pi = 0; pi < np; ++pi)
                            resonance_param_names_.push_back(rp.second + "_param" + std::to_string(pi));
                    }
                }
            }
        }

        chains_info_.push_back(info);
    }

    // Build coupling matrix
    if (!amplitude_names_.empty()) {
        coupling_matrix_ = coupling_matrix_builder_.build();
        use_coupling_matrix_ = true;
        for (int ci = 0; ci < coupling_matrix_.n_chain_free; ++ci)
            param_names_.push_back("chain_" + coupling_matrix_.chain_names[ci]);
        for (int si = 0; si < coupling_matrix_.n_step_free; ++si)
            param_names_.push_back("step_" + std::to_string(si));
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

    if (use_coupling_matrix_) {
        printf("Coupling params: %d (chain=%d step=%d)\n",
               coupling_matrix_.n_free, coupling_matrix_.n_chain_free, coupling_matrix_.n_step_free);
        for (size_t i = 0; i < param_names_.size(); ++i)
            printf("  [%zu] %s\n", i, param_names_[i].c_str());
    }

    printf("Resonance theta params: %zu\n", resonance_param_names_.size());
    for (size_t i = 0; i < resonance_param_names_.size(); ++i)
        printf("  [%zu] %s\n", i, resonance_param_names_[i].c_str());
}
