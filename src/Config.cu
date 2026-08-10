#include "Config.cuh"
#include <complex>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <queue>
#include <set>
#include <sstream>
#include <string>

float transJValue(const std::string &str)
{
    // 先检查是否有斜杠
    if (str.find('/') != std::string::npos) {
        size_t slash_pos = str.find('/');
        float num = std::stof(str.substr(0, slash_pos));
        float den = std::stof(str.substr(slash_pos + 1));
        return (den != 0) ? num / den : 0.0f;
    }

    // 尝试直接转换为浮点数
    std::stringstream ss(str);
    float value;
    if (ss >> value) {
        return value;
    }

    return 0.0f; // 默认值
}

ConfigParser::ConfigParser(const std::string &config_file)
{
    try {
        YAML::Node config = YAML::LoadFile(config_file);

        if (config["Particles"])
            parseParticles(config["Particles"]);

        if (config["Data"])
            parseData(config["Data"]);

        // Parse identical groups from Constraints BEFORE DecayChains
        // so that symmetrize auto-detection can see them
        if (config["Constraints"] && config["Constraints"]["identical"]) {
            int group_idx = 1;
            for (const auto& group : config["Constraints"]["identical"]) {
                auto names = group.as<std::vector<std::string>>();
                std::string group_name = "identical" + std::to_string(group_idx++);
                for (auto& p : particles_) {
                    for (const auto& name : names) {
                        if (p.name == name) p.identical_group = group_name;
                    }
                }
            }
        }

        if (config["Resonances"])
            parseResonances(config["Resonances"]);

        if (config["DecayChains"])
            parseDecayChains(config["DecayChains"]);

        if (config["Constraints"])
            parseConstraints(config["Constraints"]);

        if (config["Plot"])
            parsePlotConfig(config["Plot"]);

        // 用户请求的精度（auto=不检查，显式 float/double 时与 .so 比对）
        if (config["precision"])
            precision_ = config["precision"].as<std::string>();
    } catch (const YAML::Exception &e) {
        std::cerr << "Warning: Failed to parse config file \"" << config_file
                  << "\": " << e.what() << std::endl;
    }
}

std::map<std::string, std::vector<std::string>> ConfigParser::getIdenticalGroups() const
{
    std::map<std::string, std::vector<std::string>> groups;
    for (const auto& p : particles_) {
        if (!p.identical_group.empty()) {
            groups[p.identical_group].push_back(p.name);
        }
    }
    return groups;
}

std::vector<std::string> ConfigParser::getLegends() const
{
    std::vector<std::string> legends;

    for (const auto &chain : decay_chains_) {
        if (!chain.legend_template.empty()) {
            struct PlaceholderInfo {
                std::string full_name;
                std::string base_name;
                int index;
            };

            std::vector<PlaceholderInfo> placeholder_infos;
            std::vector<std::string> template_items;

            // 首先，收集所有唯一的占位符基础名
            std::map<std::string, std::vector<std::string>>
                placeholder_resonances;
            std::map<std::string, PlaceholderInfo>
                placeholder_info_map; // full_name -> PlaceholderInfo

            for (const auto &item : chain.legend_template) {
                template_items.push_back(item);

                // 检查是否是中间态占位符（包含"R_"）
                if (item.find("R_") != std::string::npos) {
                    PlaceholderInfo info;
                    info.full_name = item;

                    // 解析索引语法：R_Kpeta[0]
                    size_t bracket_pos = item.find('[');
                    if (bracket_pos != std::string::npos &&
                        item.back() == ']') {
                        // 带有索引
                        info.base_name = item.substr(0, bracket_pos);
                        std::string index_str = item.substr(
                            bracket_pos + 1, item.length() - bracket_pos - 2);
                        try {
                            info.index = std::stoi(index_str);
                        } catch (...) {
                            info.index = -1; // 索引解析失败
                        }
                    } else {
                        // 无索引
                        info.base_name = item;
                        info.index = -1;
                    }

                    placeholder_infos.push_back(info);
                    placeholder_info_map[info.full_name] = info;

                    // 如果这个基础占位符还没有处理过，收集对应的共振态
                    if (placeholder_resonances.find(info.base_name) ==
                        placeholder_resonances.end()) {
                        std::vector<std::string> resonances;
                        for (const auto &res_chain : chain.resonance_chains) {
                            if (res_chain.intermediate == info.base_name) {
                                for (const auto &spin_chain :
                                     res_chain.spin_chains) {
                                    resonances.insert(
                                        resonances.end(),
                                        spin_chain.resonances.begin(),
                                        spin_chain.resonances.end());
                                }
                                break;
                            }
                        }
                        placeholder_resonances[info.base_name] = resonances;
                    }
                }
            }

            // 如果没有占位符，直接生成legend
            if (placeholder_infos.empty()) {
                std::string legend;
                for (const auto &item : chain.legend_template) {
                    // 检查是否是粒子
                    auto particle_it = std::find_if(
                        particles_.begin(), particles_.end(),
                        [&](const Particle &p) { return p.name == item; });
                    if (particle_it != particles_.end()) {
                        // 粒子，使用其tex（如果是数组，使用所有部分）
                        for (const auto &tex_part : particle_it->tex) {
                            legend += tex_part;
                        }
                    } else {
                        // 检查是否是共振态
                        auto res_it = resonances_.find(item);
                        if (res_it != resonances_.end()) {
                            // 共振态，使用所有tex部分
                            for (const auto &tex_part : res_it->second.tex) {
                                legend += tex_part;
                            }
                        } else {
                            // 字符串字面量，直接添加
                            legend += item;
                        }
                    }
                }
                legends.push_back(legend);
                continue;
            }

            // 按基础名分组：相同基础名的占位符使用同一个共振态
            // 收集唯一的基础名
            std::vector<std::string> unique_base_names;
            for (const auto &info : placeholder_infos) {
                if (std::find(unique_base_names.begin(),
                              unique_base_names.end(),
                              info.base_name) == unique_base_names.end()) {
                    unique_base_names.push_back(info.base_name);
                }
            }

            // 为每个基础名生成共振态选择（所有共振态）
            std::map<std::string, std::vector<std::string>> base_name_choices;
            for (const auto &base_name : unique_base_names) {
                base_name_choices[base_name] =
                    placeholder_resonances[base_name];
            }

            // 生成组合：每个基础名选择一个共振态
            std::vector<std::vector<std::string>> combinations = {{}};
            for (const auto &base_name : unique_base_names) {
                const auto &choices = base_name_choices[base_name];
                std::vector<std::vector<std::string>> temp;
                for (const auto &current_combo : combinations) {
                    for (const auto &choice : choices) {
                        std::vector<std::string> new_combo = current_combo;
                        new_combo.push_back(
                            choice); // 这个choice是该基础名选择的共振态
                        temp.push_back(new_combo);
                    }
                }
                combinations = std::move(temp);
            }

            // 为每个组合生成legend
            for (const auto &combo : combinations) {
                // 创建基础名到共振态的映射
                std::map<std::string, std::string> base_name_to_resonance;
                for (size_t i = 0; i < unique_base_names.size(); ++i) {
                    base_name_to_resonance[unique_base_names[i]] = combo[i];
                }

                // 创建占位符到共振态的映射（通过基础名）
                std::map<std::string, std::string> placeholder_to_resonance;
                for (const auto &info : placeholder_infos) {
                    auto it = base_name_to_resonance.find(info.base_name);
                    if (it != base_name_to_resonance.end()) {
                        placeholder_to_resonance[info.full_name] = it->second;
                    }
                }

                // 直接构建legend字符串
                std::string legend;
                for (const auto &item : template_items) {
                    if (item.find("R_") != std::string::npos) {
                        // 是占位符
                        auto it_placeholder = placeholder_info_map.find(item);
                        auto it_resonance = placeholder_to_resonance.find(item);

                        if (it_placeholder != placeholder_info_map.end() &&
                            it_resonance != placeholder_to_resonance.end()) {
                            const PlaceholderInfo &info =
                                it_placeholder->second;
                            const std::string &resonance_name =
                                it_resonance->second;

                            // 查找共振态的tex数组
                            auto res_it = resonances_.find(resonance_name);
                            if (res_it != resonances_.end()) {
                                const auto &tex_parts = res_it->second.tex;
                                if (info.index >= 0 &&
                                    info.index < (int)tex_parts.size()) {
                                    // 有索引，选择对应的tex部分
                                    legend += tex_parts[info.index];
                                } else if (info.index == -1) {
                                    // 无索引，使用所有tex部分
                                    for (const auto &tex_part : tex_parts) {
                                        legend += tex_part;
                                    }
                                }
                            } else {
                                // 共振态未找到，使用名称
                                legend += resonance_name;
                            }
                        } else {
                            legend += item;
                        }
                    } else {
                        // 普通字符串或粒子名
                        // 检查是否是粒子
                        auto particle_it = std::find_if(
                            particles_.begin(), particles_.end(),
                            [&](const Particle &p) { return p.name == item; });
                        if (particle_it != particles_.end()) {
                            // 粒子，使用其tex（如果是数组，使用所有部分）
                            for (const auto &tex_part : particle_it->tex) {
                                legend += tex_part;
                            }
                        } else {
                            // 字符串字面量，直接添加
                            legend += item;
                        }
                    }
                }

                legends.push_back(legend);
            }
        }
    }

    return legends;
}

void ConfigParser::parseParticles(const YAML::Node &node)
{
    for (const auto &particle_node : node) {
        std::string name = particle_node.first.as<std::string>();
        const auto &props = particle_node.second;

        Particle particle;
        particle.name = name;
        // particle.spin = props["J"].as<int>();
        // particle.spin = 2 * props["J"].as<float>() + 1;
        std::string spin_str = props["J"].as<std::string>();
        particle.spin = static_cast<int>(2 * transJValue(spin_str) + 1);
        particle.parity = props["P"].as<int>();
        particle.mass = props["mass"].as<double>();

        // 处理tex字段，可能是字符串或字符串数组
        if (props["tex"].IsSequence()) {
            // tex是数组
            for (const auto &tex_part : props["tex"]) {
                particle.tex.push_back(tex_part.as<std::string>());
            }
        } else {
            // 单个字符串，放入向量中
            particle.tex.push_back(props["tex"].as<std::string>());
        }

        // 处理极化设置: polarization: [1, -1] 表示仅 m=±1
        if (props["polarization"]) {
            if (props["polarization"].IsSequence()) {
                for (const auto& m_node : props["polarization"]) {
                    std::string m_str = m_node.as<std::string>();
                    int two_m = static_cast<int>(2 * transJValue(m_str));
                    particle.polarization_2m.push_back(two_m);
                }
            }
        }

        particles_.push_back(particle);
    }
}

void ConfigParser::parseData(const YAML::Node &node)
{
    if (node["order"])
        data_order_ = node["order"].as<std::vector<std::string>>();

    if (node["data"])
        data_files_["data"] = node["data"].as<std::vector<std::string>>();

    if (node["phsp"])
        data_files_["phsp"] = node["phsp"].as<std::vector<std::string>>();

    if (node["phsp_truth"])
        data_files_["phsp_truth"] =
            node["phsp_truth"].as<std::vector<std::string>>();

    if (node["bkg"])
        data_files_["bkg"] = node["bkg"].as<std::vector<std::string>>();

    if (node["bkg_weights"])
        data_files_["bkg_weights"] =
            node["bkg_weights"].as<std::vector<std::string>>();

    if (node["data_weights"])
        data_files_["data_weights"] =
            node["data_weights"].as<std::vector<std::string>>();

    if (node["phsp_weights"])
        data_files_["phsp_weights"] =
            node["phsp_weights"].as<std::vector<std::string>>();
}

void ConfigParser::parseDecayChains(const YAML::Node &node)
{
    // Build set of known particle names for compact format detection
    std::set<std::string> particle_names;
    for (const auto& p : particles_) particle_names.insert(p.name);

    for (const auto &chain_node : node) {
        std::string chain_name = chain_node.first.as<std::string>();
        const auto &chain_data = chain_node.second;

        // ---------------------------------------------------------------
        // Detect compact format:
        //   No "decay" key, has a particle-name key whose value is a
        //   sequence of [bachelor, intermediate] pairs.
        // ---------------------------------------------------------------
        std::string mother;
        bool is_compact = false;

        if (!chain_data["decay"]) {
            for (const auto& kv : chain_data) {
                std::string key = kv.first.as<std::string>();
                if (key == "intermediates" || key == "legend" || key == "symmetrize")
                    continue;
                if (particle_names.count(key) && kv.second.IsSequence()) {
                    // Must be a sequence of [bachelor, intermediate] pairs
                    if (kv.second.size() > 0 &&
                        kv.second[0].IsSequence() &&
                        kv.second[0].size() == 2) {
                        mother = key;
                        is_compact = true;
                        break;
                    }
                }
            }
        }

        if (is_compact) {
            // ============================================================
            // Compact format — expand one block into N DecayChainConfigs
            // ============================================================
            const auto& channels = chain_data[mother];

            // --- Parse intermediate decays ---
            // Single mode:  R_KK: [Kp, Km]  or  [Kp, Km, {opts}]
            // Multi mode:   R1: [[R2, R3], [R4, R5, {p_break: true}]]
            struct IntDecay {
                std::string d1, d2;
                bool is_bf = true;
                bool p_break = false;
            };
            std::map<std::string, std::vector<IntDecay>> int_decay_modes;
            for (const auto& kv : chain_data) {
                std::string key = kv.first.as<std::string>();
                if (key == mother || key == "intermediates" ||
                    key == "legend" || key == "symmetrize")
                    continue;
                if (!kv.second.IsSequence() || kv.second.size() == 0) continue;

                auto parseOneMode = [](const YAML::Node& node) -> IntDecay {
                    IntDecay id;
                    id.d1 = node[0].as<std::string>();
                    id.d2 = node[1].as<std::string>();
                    if (node.size() >= 3 && node[2].IsMap()) {
                        const auto& dopts = node[2];
                        if (dopts["is_bf"])   id.is_bf   = dopts["is_bf"].as<bool>();
                        if (dopts["p_break"]) id.p_break = dopts["p_break"].as<bool>();
                    }
                    return id;
                };

                if (kv.second[0].IsSequence()) {
                    // Multi-mode: [[R2,R3], [R4,R5], ...]
                    for (const auto& mode : kv.second)
                        int_decay_modes[key].push_back(parseOneMode(mode));
                } else if (kv.second.size() >= 2 && kv.second[0].IsScalar()) {
                    // Single mode: [Kp, Km]
                    int_decay_modes[key].push_back(parseOneMode(kv.second));
                }
            }

            // Helper: get default mode for an intermediate (for BFS)
            auto getDecayMode = [&](const std::string& name, size_t idx = 0) -> const IntDecay* {
                auto it = int_decay_modes.find(name);
                if (it == int_decay_modes.end() || idx >= it->second.size()) return nullptr;
                return &it->second[idx];
            };

            // --- Parse intermediates resonance sub-block ---
            // Two formats:
            //   Explicit JP:  R_KK: [{J: 1, P: -1}: [phi1680]]
            //   Auto-detect:  R_KK: [phi1680]   or   [phi1680, phi2170]
            std::map<std::string, ResonanceChainConfig> res_chain_map;
            if (chain_data["intermediates"]) {
                for (const auto& res_node : chain_data["intermediates"]) {
                    std::string int_name = res_node.first.as<std::string>();
                    ResonanceChainConfig res_chain;
                    res_chain.intermediate = int_name;

                    for (const auto& spin_node : res_node.second) {
                        if (spin_node.IsMap()) {
                            // Explicit JP: {[J:1,P:-1]: [phi1680]}
                            for (const auto& spin_pair : spin_node) {
                                SpinChainConfig spin_chain;

                                if (spin_pair.first.IsSequence()) {
                                    for (const auto& jp : spin_pair.first) {
                                        if (jp["J"]) {
                                            std::string j_str = jp["J"].as<std::string>();
                                            spin_chain.spin_parity.push_back(
                                                2 * transJValue(j_str) + 1);
                                        }
                                        if (jp["P"])
                                            spin_chain.spin_parity.push_back(
                                                jp["P"].as<int>());
                                    }
                                }

                                spin_chain.resonances =
                                    spin_pair.second.as<std::vector<std::string>>();
                                res_chain.spin_chains.push_back(spin_chain);
                            }
                        } else if (spin_node.IsSequence()) {
                            // Auto-detect: [phi1680] or [phi1680, phi2170]
                            // Group resonances by (J, P) from the Resonances section
                            std::map<std::pair<int,int>, std::vector<std::string>> jp_groups;
                            for (const auto& rname_node : spin_node) {
                                std::string rname = rname_node.as<std::string>();
                                auto rit = resonances_.find(rname);
                                if (rit != resonances_.end()) {
                                    jp_groups[{rit->second.J, rit->second.P}].push_back(rname);
                                } else {
                                    std::cerr << "Warning: resonance '" << rname
                                              << "' not found in Resonances section" << std::endl;
                                }
                            }
                            for (const auto& [jp, names] : jp_groups) {
                                SpinChainConfig spin_chain;
                                spin_chain.spin_parity = {jp.first, jp.second};
                                spin_chain.resonances = names;
                                res_chain.spin_chains.push_back(spin_chain);
                            }
                        }
                    }
                    res_chain_map[int_name] = res_chain;
                }
            }

            // --- symmetrize ---
            bool symmetrize = false;
            if (chain_data["symmetrize"])
                symmetrize = chain_data["symmetrize"].as<bool>();

            // --- legends: one per channel, in order ---
            std::vector<std::vector<std::string>> all_legends;
            if (chain_data["legends"] && chain_data["legends"].IsSequence()) {
                for (const auto& leg : chain_data["legends"])
                    all_legends.push_back(leg.as<std::vector<std::string>>());
            }

            // --- Expand each channel: for multi-mode intermediates, generate one chain per mode ---
            size_t ch_idx = 0;
            for (const auto& ch : channels) {
                std::string bachelor = ch[0].as<std::string>();
                std::string intermediate = ch[1].as<std::string>();

                // Parse per-channel constraints (3rd element) — overrides all defaults
                bool ch_symmetrize = symmetrize;
                bool ch_is_bf1 = true, ch_is_bf2 = true;
                bool ch_p_break1 = false, ch_p_break2 = false;
                bool has_ch_is_bf = false, has_ch_p_break = false;
                std::vector<std::string> ch_legend;
                // Fall back to top-level legends list
                if (ch_idx < all_legends.size())
                    ch_legend = all_legends[ch_idx];

                if (ch.size() >= 3 && ch[2].IsMap()) {
                    const auto& opts = ch[2];
                    if (opts["symmetrize"])
                        ch_symmetrize = opts["symmetrize"].as<bool>();
                    if (opts["legend"])
                        ch_legend = opts["legend"].as<std::vector<std::string>>();
                    if (opts["is_bf"]) {
                        has_ch_is_bf = true;
                        if (opts["is_bf"].IsSequence()) {
                            ch_is_bf1 = opts["is_bf"][0].as<bool>();
                            ch_is_bf2 = opts["is_bf"][1].as<bool>();
                        } else {
                            ch_is_bf1 = ch_is_bf2 = opts["is_bf"].as<bool>();
                        }
                    }
                    if (opts["p_break"]) {
                        has_ch_p_break = true;
                        if (opts["p_break"].IsSequence()) {
                            ch_p_break1 = opts["p_break"][0].as<bool>();
                            ch_p_break2 = opts["p_break"][1].as<bool>();
                        } else {
                            ch_p_break1 = ch_p_break2 = opts["p_break"].as<bool>();
                        }
                    }
                }

                // --- Determine multi-mode iteration driver ---
                // Priority: intermediate modes > bachelor modes
                size_t n_modes = 1;
                bool bachelor_drives_modes = false;
                {
                    auto modes_it = int_decay_modes.find(intermediate);
                    if (modes_it != int_decay_modes.end()) {
                        n_modes = modes_it->second.size();
                    } else {
                        auto bmodes_it = int_decay_modes.find(bachelor);
                        if (bmodes_it != int_decay_modes.end()) {
                            n_modes = bmodes_it->second.size();
                            bachelor_drives_modes = true;
                        }
                    }
                }

                for (size_t mi = 0; mi < n_modes; ++mi) {
                    // Multi-mode filter: skip modes that duplicate the other daughter
                    if (n_modes > 1) {
                        if (bachelor_drives_modes) {
                            const IntDecay* mode = getDecayMode(bachelor, mi);
                            if (mode && (mode->d1 == intermediate || mode->d2 == intermediate))
                                continue;
                        } else {
                            const IntDecay* mode = getDecayMode(intermediate, mi);
                            if (mode && (mode->d1 == bachelor || mode->d2 == bachelor))
                                continue;
                        }
                    }

                    DecayChainConfig chain;
                    chain.name = chain_name + "_" + (bachelor_drives_modes ? bachelor : intermediate);
                    if (n_modes > 1)
                        chain.name += "_" + std::to_string(mi);

                    // --- Step 1: mother → bachelor + intermediate ---
                    DecayStep step1;
                    step1.mother = mother;
                    step1.daughters = {bachelor, intermediate};
                    step1.is_bf = ch_is_bf1;
                    step1.p_break = ch_p_break1;
                    chain.decay_steps.push_back(step1);

                    // --- Determine which daughters are intermediates (not known particles) ---
                    bool intermediate_decays = (int_decay_modes.count(intermediate) > 0);
                    bool bachelor_decays = (int_decay_modes.count(bachelor) > 0);

                    // Push resonance chains for daughters that have JP but no decay mode
                    if (!intermediate_decays && res_chain_map.count(intermediate))
                        chain.resonance_chains.push_back(res_chain_map[intermediate]);
                    if (!bachelor_decays && res_chain_map.count(bachelor))
                        chain.resonance_chains.push_back(res_chain_map[bachelor]);

                    // If neither daughter decays further, emit and finish
                    if (!intermediate_decays && !bachelor_decays) {
                        chain.legend_template = ch_legend.empty()
                            ? std::vector<std::string>{intermediate, " ", bachelor}
                            : ch_legend;
                        chain.symmetrize = ch_symmetrize;
                        decay_chains_.push_back(chain);
                        continue;
                    }

                    // --- BFS: recursively resolve intermediates ---
                    struct BFSItem { std::string name; bool is_first; bool bf, pb; };
                    std::queue<BFSItem> queue;

                    if (intermediate_decays) {
                        const IntDecay* ifm = getDecayMode(intermediate,
                            bachelor_drives_modes ? 0 : mi);
                        bool step2_is_bf   = has_ch_is_bf   ? ch_is_bf2   : (ifm ? ifm->is_bf : true);
                        bool step2_p_break = has_ch_p_break ? ch_p_break2 : (ifm ? ifm->p_break : false);
                        queue.push({intermediate, !bachelor_drives_modes, step2_is_bf, step2_p_break});
                    }
                    if (bachelor_decays) {
                        const IntDecay* bm = getDecayMode(bachelor,
                            bachelor_drives_modes ? mi : 0);
                        queue.push({bachelor, bachelor_drives_modes,
                            bm ? bm->is_bf : true,
                            bm ? bm->p_break : false});
                    }

                    while (!queue.empty()) {
                        auto item = queue.front(); queue.pop();
                        const IntDecay* mode = getDecayMode(item.name,
                            item.is_first ? mi : 0);
                        if (!mode) continue;

                        DecayStep substep;
                        substep.mother = item.name;
                        substep.daughters = {mode->d1, mode->d2};
                        substep.is_bf   = item.bf;
                        substep.p_break = item.pb;
                        chain.decay_steps.push_back(substep);

                        // Resonance chain for this intermediate
                        if (res_chain_map.count(item.name))
                            chain.resonance_chains.push_back(res_chain_map[item.name]);

                        // Enqueue any daughter that is itself an intermediate
                        auto enqueue = [&](const std::string& d) {
                            if (int_decay_modes.count(d)) {
                                const auto* dm = getDecayMode(d, 0);
                                queue.push({d, false,
                                    dm ? dm->is_bf   : true,
                                    dm ? dm->p_break : false});
                            }
                        };
                        enqueue(mode->d1);
                        enqueue(mode->d2);
                    }

                    // --- symmetrize: auto-detect ---
                    {
                        bool sym_explicit = (ch.size() >= 3 && ch[2].IsMap() &&
                                             ch[2]["symmetrize"]);
                        if (sym_explicit) {
                            chain.symmetrize = ch_symmetrize;
                        } else {
                            // Use symmetrize from block-level default
                            chain.symmetrize = symmetrize;
                            // Also auto-detect: any step has identical daughters?
                            for (const auto& step : chain.decay_steps) {
                                const Particle *pd1 = nullptr, *pd2 = nullptr;
                                for (const auto& p : particles_) {
                                    if (p.name == step.daughters[0]) pd1 = &p;
                                    if (p.name == step.daughters[1]) pd2 = &p;
                                }
                                if (pd1 && pd2 &&
                                    !pd1->identical_group.empty() &&
                                    pd1->identical_group == pd2->identical_group) {
                                    chain.symmetrize = true;
                                    break;
                                }
                            }
                        }
                    }

                    // --- Legend ---
                    if (!ch_legend.empty())
                        chain.legend_template = ch_legend;
                    else
                        chain.legend_template = {intermediate, " ", bachelor};

                    decay_chains_.push_back(chain);
                } // for each mode
                ++ch_idx;
            }

        } else {
            // ============================================================
            // Original format (unchanged)
            // ============================================================
            DecayChainConfig chain;
            chain.name = chain_name;

            // 解析衰变步骤
            if (chain_data["decay"]) {
                for (const auto &step_node : chain_data["decay"]) {
                    for (const auto &decay_pair : step_node) {
                        DecayStep step;
                        step.mother = decay_pair.first.as<std::string>();

                        if (decay_pair.second.IsSequence()) {
                            step.daughters =
                                decay_pair.second.as<std::vector<std::string>>();
                        } else if (decay_pair.second.IsMap()) {
                            step.daughters =
                                decay_pair.second["daughters"]
                                    .as<std::vector<std::string>>();
                            if (decay_pair.second["is_bf"])
                                step.is_bf = decay_pair.second["is_bf"].as<bool>();
                            if (decay_pair.second["p_break"])
                                step.p_break =
                                    decay_pair.second["p_break"].as<bool>();
                        }
                        chain.decay_steps.push_back(step);
                    }
                }
            }

            // 解析共振态链
            for (const auto &res_node : chain_data) {
                std::string key = res_node.first.as<std::string>();
                if (key != "decay" && key != "legend" && key != "symmetrize") {
                    ResonanceChainConfig res_chain;
                    res_chain.intermediate = key;

                    for (const auto &spin_node : res_node.second) {
                        for (const auto &spin_pair : spin_node) {
                            SpinChainConfig spin_chain;

                            if (spin_pair.first.IsSequence()) {
                                for (const auto &jp : spin_pair.first) {
                                    if (jp["J"]) {
                                        std::string j_str =
                                            jp["J"].as<std::string>();
                                        spin_chain.spin_parity.push_back(
                                            2 * transJValue(j_str) + 1);
                                    }
                                    if (jp["P"])
                                        spin_chain.spin_parity.push_back(
                                            jp["P"].as<int>());
                                }
                            }

                            spin_chain.resonances =
                                spin_pair.second.as<std::vector<std::string>>();
                            res_chain.spin_chains.push_back(spin_chain);
                        }
                    }
                    chain.resonance_chains.push_back(res_chain);
                }
            }

            if (chain_data["legend"])
                chain.legend_template =
                    chain_data["legend"].as<std::vector<std::string>>();

            if (chain_data["symmetrize"])
                chain.symmetrize = chain_data["symmetrize"].as<bool>();

            decay_chains_.push_back(chain);
        }
    }
}

void ConfigParser::parseResonances(const YAML::Node &node)
{
    for (const auto &res_node : node) {
        std::string name = res_node.first.as<std::string>();
        const auto &props = res_node.second;

        ResonanceConfig res;
        res.name = name;
        // res.J = props["J"].as<int>();
        res.J =
            static_cast<int>(2 * transJValue(props["J"].as<std::string>()) + 1);
        res.P = props["P"].as<int>();
        res.type = props["model"].as<std::string>();
        if (props["parameters"])
            res.parameters = props["parameters"].as<std::vector<double>>();

        // 模型选项（Hist: file/bins/range/extrapolate）
        if (props["file"])        res.options["file"] = props["file"].as<std::string>();
        if (props["bins"])        res.options["bins"] = props["bins"].as<std::string>();
        if (props["range"]) {
            auto r = props["range"].as<std::vector<double>>();
            std::string s;
            for (size_t i = 0; i < r.size(); ++i) {
                if (i) s += ",";
                s += std::to_string(r[i]);
            }
            res.options["range"] = s;
        }
        if (props["extrapolate"]) res.options["extrapolate"] = props["extrapolate"].as<std::string>();
        // Custom 模型: params(参数名列表) 和 expr(表达式)
        if (props["params"]) {
            auto pl = props["params"].as<std::vector<std::string>>();
            std::string s;
            for (size_t i = 0; i < pl.size(); ++i) {
                if (i) s += ",";
                s += pl[i];
            }
            res.options["params"] = s;
        }
        if (props["expr"]) res.options["expr"] = props["expr"].as<std::string>();

        // 解析 channels 字段（仅 Flatte 使用）
        if (props["channels"]) {
            for (const auto& ch : props["channels"]) {
                res.channels.push_back(ch.as<std::vector<double>>());
            }
        }

        // 处理tex字段，可能是字符串或字符串数组；缺失时默认用共振态名字
        if (props["tex"]) {
            if (props["tex"].IsSequence()) {
                // tex是数组
                for (const auto &tex_part : props["tex"]) {
                    res.tex.push_back(tex_part.as<std::string>());
                }
            } else {
                // 单个字符串，放入向量中
                res.tex.push_back(props["tex"].as<std::string>());
            }
        } else {
            res.tex.push_back(name);   // 默认: 用共振态名字显示
        }

        // 解析free字段: [0,1] 扫params[0]和params[1]; [-1] 全扫
        if (props["free"]) {
            res.free = props["free"].as<std::vector<int>>();
        }

        // 解析free_range: 每个free参数对应的 [lower, upper]
        if (props["free_range"]) {
            for (const auto& range_node : props["free_range"]) {
                res.free_range.push_back(range_node.as<std::vector<double>>());
            }
        }

        resonances_[name] = res;
    }
}

void ConfigParser::parseConstraints(const YAML::Node &node)
{
    constraints_.clear();

    // 解析全局 maxL
    if (node["maxL"]) {
        global_maxL_ = node["maxL"].as<int>();
    }

    // 解析全局 barrier factor d
    if (node["bf_d"]) {
        global_bf_d_ = node["bf_d"].as<double>();
    }

    // 解析全同粒子分组: identical: [[pi01, pi02], [Ks1, Ks2]]
    if (node["identical"]) {
        int group_idx = 1;
        for (const auto& group : node["identical"]) {
            auto names = group.as<std::vector<std::string>>();
            std::string group_name = "identical" + std::to_string(group_idx++);
            for (auto& p : particles_) {
                for (const auto& name : names) {
                    if (p.name == name) {
                        p.identical_group = group_name;
                    }
                }
            }
        }
        // 验证: 同一分组内粒子自旋一致
        auto groups = getIdenticalGroups();
        for (const auto& [gname, pnames] : groups) {
            int ref_spin = -1;
            for (const auto& p : particles_) {
                if (p.name == pnames[0]) { ref_spin = p.spin; break; }
            }
            for (const auto& nm : pnames) {
                for (const auto& p : particles_) {
                    if (p.name == nm && p.spin != ref_spin) {
                        std::cerr << "Warning: identical group \"" << gname
                                  << "\" contains particles with different spins" << std::endl;
                    }
                }
            }
        }
    }

    // 解析 trans 约束 — 三种格式:
    //   矩阵:  - [A, B]: [[-1, -1]]           → 复数矩阵
    //   列表:  - [A, B]: [1]  或  [-1]       → 实数标量列表
    //   标量:  - [A, B]: 1    或  -1          → 单个实数
    if (node["trans"]) {
        for (const auto &constraint_list : node["trans"]) {
            for (const auto &pair : constraint_list) {
                ConstraintConfig constraint;
                constraint.type = "trans";
                constraint.names = pair.first.as<std::vector<std::string>>();
                const YAML::Node &values = pair.second;

                if (values.IsSequence()) {
                    if (values.size() > 0 && values[0].IsSequence()) {
                        // Matrix format: [[-1, -1], [0, 1], ...]
                        for (const auto &row : values) {
                            if (row.IsSequence() && row.size() >= 2) {
                                constraint.values.push_back(
                                    std::complex<double>(
                                        row[0].as<double>(),
                                        row[1].as<double>()));
                            }
                        }
                    } else {
                        // Simple list format: [1] or [-1, 1, ...]
                        for (const auto &v : values) {
                            constraint.values.push_back(
                                std::complex<double>(v.as<double>(), 0.0));
                        }
                    }
                } else if (values.IsScalar()) {
                    // Scalar format: 1 or -1
                    constraint.values.push_back(
                        std::complex<double>(values.as<double>(), 0.0));
                }

                constraints_.push_back(constraint);
            }
        }
    }
}

void ConfigParser::parsePlotConfig(const YAML::Node &node)
{
    plot_configs_.clear();

    // 解析mass图配置
    if (node["mass"]) {
        for (const auto &plot_item : node["mass"]) {
            PlotConfig config;
            config.type = "mass";

            // 解析particles
            if (plot_item["input"]) {
                const YAML::Node &particles_node = plot_item["input"];
                if (particles_node.IsSequence()) {
                    // 检查是否是一维序列
                    bool is_2d = false;
                    for (const auto &elem : particles_node) {
                        if (elem.IsSequence()) {
                            is_2d = true;
                            break;
                        }
                    }
                    if (is_2d) {
                        // 二维序列
                        for (const auto &group : particles_node) {
                            config.particles.push_back(
                                group.as<std::vector<std::string>>());
                        }
                    } else {
                        // 一维序列，包装成二维
                        config.particles.push_back(
                            particles_node.as<std::vector<std::string>>());
                    }
                }
            }

            // bins: 单个整数
            config.bins = {plot_item["bins"].as<int>()};

            // range: 一维数组，转换为二维数组
            std::vector<double> range =
                plot_item["range"].as<std::vector<double>>();
            config.ranges = {range};

            // display: 一维数组（两个字符串）
            config.display =
                plot_item["display"].as<std::vector<std::string>>();

            plot_configs_.push_back(config);
        }
    }

    // 解析cosbeta图配置
    if (node["cosbeta"]) {
        for (const auto &plot_item : node["cosbeta"]) {
            PlotConfig config;
            config.type = "cosbeta";

            // 解析particles（二维列表）
            if (plot_item["input"]) {
                const YAML::Node &particles_node = plot_item["input"];
                for (const auto &group : particles_node) {
                    config.particles.push_back(
                        group.as<std::vector<std::string>>());
                }
            }

            // bins: 单个整数
            config.bins = {plot_item["bins"].as<int>()};

            // range: 一维数组，转换为二维数组
            std::vector<double> range =
                plot_item["range"].as<std::vector<double>>();
            config.ranges = {range};

            // display: 一维数组（两个字符串）
            config.display =
                plot_item["display"].as<std::vector<std::string>>();

            plot_configs_.push_back(config);
        }
    }

    // 解析dalitz图配置
    if (node["dalitz"]) {
        for (const auto &plot_item : node["dalitz"]) {
            PlotConfig config;
            config.type = "dalitz";

            // 解析particles（二维列表，包含两个粒子组）
            if (plot_item["input"]) {
                const YAML::Node &particles_node = plot_item["input"];
                for (const auto &group : particles_node) {
                    config.particles.push_back(
                        group.as<std::vector<std::string>>());
                }
            }

            // bins: 二维数组（两个整数）
            config.bins = plot_item["bins"].as<std::vector<int>>();

            // range: 二维数组
            const YAML::Node &range_node = plot_item["range"];
            for (const auto &range : range_node) {
                config.ranges.push_back(range.as<std::vector<double>>());
            }

            // display: 一维数组（两个字符串）
            config.display =
                plot_item["display"].as<std::vector<std::string>>();

            plot_configs_.push_back(config);
        }
    }
}
