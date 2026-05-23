#include <Config.cuh>
#include <algorithm>
#include <fstream>
#include <iostream>
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

        if (config["DecayChains"])
            parseDecayChains(config["DecayChains"]);

        if (config["Resonances"])
            parseResonances(config["Resonances"]);

        if (config["Constraints"])
            parseConstraints(config["Constraints"]);

        if (config["Plot"])
            parsePlotConfig(config["Plot"]);
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
}

void ConfigParser::parseDecayChains(const YAML::Node &node)
{
    for (const auto &chain_node : node) {
        std::string chain_name = chain_node.first.as<std::string>();
        const auto &chain_data = chain_node.second;

        DecayChainConfig chain;
        chain.name = chain_name;

        // 解析衰变步骤
        if (chain_data["decay"]) {
            for (const auto &step_node : chain_data["decay"]) {
                for (const auto &decay_pair : step_node) {
                    DecayStep step;
                    step.mother = decay_pair.first.as<std::string>();

                    if (decay_pair.second.IsSequence()) {
                        // 紧凑格式: mother: [daughter1, daughter2]
                        step.daughters =
                            decay_pair.second.as<std::vector<std::string>>();
                    } else if (decay_pair.second.IsMap()) {
                        // 扩展格式: mother: { daughters: [...], is_bf: bool, p_break: bool }
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
            if (key != "decay" && key != "legend") {
                ResonanceChainConfig res_chain;
                res_chain.intermediate = key;

                for (const auto &spin_node : res_node.second) {
                    for (const auto &spin_pair : spin_node) {
                        SpinChainConfig spin_chain;

                        // 解析自旋宇称 [J, P]
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

                        // 解析共振态列表
                        spin_chain.resonances =
                            spin_pair.second.as<std::vector<std::string>>();
                        res_chain.spin_chains.push_back(spin_chain);
                    }
                }
                chain.resonance_chains.push_back(res_chain);
            }
        }

        // 解析legend模板
        if (chain_data["legend"])
            chain.legend_template =
                chain_data["legend"].as<std::vector<std::string>>();

        // 解析全同粒子对称化标记
        if (chain_data["symmetrize"])
            chain.symmetrize = chain_data["symmetrize"].as<bool>();

        decay_chains_.push_back(chain);
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
        res.parameters = props["parameters"].as<std::vector<double>>();

        // 解析 channels 字段（仅 Flatte 使用）
        if (props["channels"]) {
            for (const auto& ch : props["channels"]) {
                res.channels.push_back(ch.as<std::vector<double>>());
            }
        }

        // 处理tex字段，可能是字符串或字符串数组
        if (props["tex"].IsSequence()) {
            // tex是数组
            for (const auto &tex_part : props["tex"]) {
                res.tex.push_back(tex_part.as<std::string>());
            }
        } else {
            // 单个字符串，放入向量中
            res.tex.push_back(props["tex"].as<std::string>());
        }

        // 解析scan字段: [0,1] 扫params[0]和params[1]; [-1] 全扫
        if (props["scan"]) {
            res.scan = props["scan"].as<std::vector<int>>();
        }

        // 解析scan_range: 每个scan参数对应的 [lower, upper]
        if (props["scan_range"]) {
            for (const auto& range_node : props["scan_range"]) {
                res.scan_range.push_back(range_node.as<std::vector<double>>());
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

    // 解析 full 约束（同时包含实部和虚部）
    if (node["trans"]) {
        for (const auto &constraint_list : node["trans"]) {
            for (const auto &pair : constraint_list) {
                ConstraintConfig constraint;
                constraint.type = "trans";

                // 解析链名列表
                constraint.names = pair.first.as<std::vector<std::string>>();

                // 解析约束值
                const YAML::Node &values = pair.second;
                if (values.IsSequence()) {
                    for (const auto &row : values) {
                        if (row.IsSequence() && row.size() >= 2) {
                            double real_val = row[0].as<double>();
                            double imag_val = row[1].as<double>();
                            // constraint.constraints.emplace_back(real_val,
                            // imag_val);
                            constraint.values.push_back(
                                std::complex<double>(real_val, imag_val));
                        }
                    }
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
