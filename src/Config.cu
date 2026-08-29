#include "Config.cuh"
#include <complex>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <queue>
#include <set>
#include <sstream>
#include <string>

// ---- 统一观测表达式解析: "M(p,pbar)" / "CosAngle(p; [p,pbar])" / "Phi(p)" ----
static std::string trimStr(const std::string &s)
{
    size_t b = s.find_first_not_of(" \t\r\n");
    size_t e = s.find_last_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    return s.substr(b, e - b + 1);
}

static std::vector<std::string> splitComma(const std::string &s)
{
    std::vector<std::string> out;
    std::string cur;
    for (char ch : s) {
        if (ch == ',') { out.push_back(trimStr(cur)); cur.clear(); }
        else cur += ch;
    }
    out.push_back(trimStr(cur));
    return out;
}

static int mapObsFunc(const std::string &name)
{
    std::string n;
    for (char c : name) n += (char)std::toupper(c);
    if (n == "M")  return OBS_M;
    if (n == "M2") return OBS_M2;
    if (n == "P")  return OBS_P;
    if (n == "E")  return OBS_E;
    if (n == "PERP" || n == "PT") return OBS_PERP;
    if (n == "THETA") return OBS_THETA;
    if (n == "PHI")   return OBS_PHI;
    if (n == "COSTHETA" || n == "COS_THETA") return OBS_COSTHETA;
    if (n == "ANGLE") return OBS_ANGLE;
    if (n == "COSANGLE" || n == "COS_ANGLE") return OBS_COSANGLE;
    return -1;
}

static ObsSpec parseObsExpr(const std::string &expr)
{
    ObsSpec spec;
    size_t lp = expr.find('(');
    size_t rp = expr.rfind(')');
    if (lp == std::string::npos || rp == std::string::npos || rp < lp) {
        std::cerr << "Warning: 观测表达式缺少括号, 跳过: " << expr << std::endl;
        return spec;
    }
    std::string fname = trimStr(expr.substr(0, lp));
    std::string inner = trimStr(expr.substr(lp + 1, rp - lp - 1));

    // 语法: Func([主体系统], [帧系统], [轴系统])  (";" 与 "," 等价, 帧/轴可省)
    //   第1组=主体, 第2组=帧(缺省=顶层母粒子静系), 第3组=轴(缺省=顶层母粒子; 角度类)
    //   例: M([p,pbar]) / M(p,pbar) / CosAngle([p], [p,pbar], [psip]) / CosAngle(p; [p,pbar])
    std::vector<std::string> groups;
    std::string cur;
    char sep = 0;  // 0=未定, ',' 或 ';'
    int depth = 0; // 括号深度: 深度内的 , 不分割 ([p,pbar] 内部逗号是粒子分隔)
    for (char ch : inner) {
        if (ch == '[') ++depth;
        else if (ch == ']') --depth;
        if ((ch == ';' || ch == ',') && depth == 0) {
            if (sep != 0 && ch != sep) continue;  // 只认第一个分隔符
            sep = ch;
            groups.push_back(trimStr(cur));
            cur.clear();
        } else {
            cur += ch;
        }
    }
    groups.push_back(trimStr(cur));

    // 识别显式 [...] 组
    auto isBracket = [](const std::string &g) {
        return g.size() >= 2 && g.front() == '[' && g.back() == ']';
    };
    auto flatNames = [](const std::string &g, std::vector<std::string> &out) {
        std::string s = g;
        if (!s.empty() && s.front() == '[' && s.back() == ']')
            s = s.substr(1, s.size() - 2);
        for (auto &t : splitComma(s))
            if (!t.empty()) out.push_back(t);
    };

    std::vector<std::string> brack;
    std::vector<std::string> loose;
    for (const auto &g : groups) {
        if (!g.empty() && isBracket(g)) brack.push_back(g);
        else if (!g.empty()) loose.push_back(g);
    }
    if (brack.size() >= 1) flatNames(brack[0], spec.args);  // 主体
    for (const auto &g : loose) spec.args.push_back(g);       // 无括号名并入主体
    if (brack.size() >= 2) flatNames(brack[1], spec.boost);  // 帧
    if (brack.size() >= 3) flatNames(brack[2], spec.axis);   // 轴

    spec.func = mapObsFunc(fname);
    if (spec.func < 0) {
        std::cerr << "Warning: 未知观测函数 \"" << fname
                  << "\" (支持: M/M2/P/E/Perp/Pt/Theta/Phi/CosTheta/Angle/CosAngle), "
                  << "跳到 M" << std::endl;
        spec.func = OBS_M;
    }
    return spec;
}

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
    config_file_ = config_file;
    try {
        YAML::Node config = YAML::LoadFile(config_file);

        if (config["Particles"])
            parseParticles(config["Particles"]);

        if (config["Data"])
            parseData(config["Data"]);

        // Parse identical groups from Constraints BEFORE DecayChains
        // so that symmetrization auto-detection can see them
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

        // 链过滤器（Constraints.chains）早提：需在 DecayChains 解析前读取
        if (config["Constraints"] && config["Constraints"]["chains"]) {
            const auto& ch = config["Constraints"]["chains"];
            if (ch.IsScalar()) {
                // 外部文件：每行一个子串（跳过空行与 # 注释），相对 config.yml 所在目录
                std::filesystem::path filter_path =
                    std::filesystem::path(config_file).parent_path() /
                    ch.as<std::string>();
                std::ifstream f(filter_path);
                std::string line;
                while (std::getline(f, line)) {
                    // 去首尾空白
                    size_t b = line.find_first_not_of(" \t\r\n");
                    if (b == std::string::npos) continue;
                    size_t e = line.find_last_not_of(" \t\r\n");
                    std::string pat = line.substr(b, e - b + 1);
                    if (pat.empty() || pat[0] == '#') continue;
                    chain_filter_.push_back(pat);
                }
                if (chain_filter_.empty())
                    std::cerr << "Warning: chain filter file \"" << filter_path
                              << "\" empty or not found" << std::endl;
            } else if (ch.IsSequence()) {
                chain_filter_ = ch.as<std::vector<std::string>>();
            }
        }

        if (config["Resonances"]) {
            if (config["Resonances"].IsScalar()) {
                // 外部共振态文件: Resonances 栏写文件名 → 从该文件读取（相对 config.yml 所在目录）
                std::filesystem::path res_path =
                    std::filesystem::path(config_file).parent_path() /
                    config["Resonances"].as<std::string>();
                try {
                    YAML::Node res_node = YAML::LoadFile(res_path.string());
                    // 容错: 外部文件若带顶层 Resonances 键则取其子节点
                    if (res_node["Resonances"])
                        res_node = res_node["Resonances"];
                    parseResonances(res_node);
                } catch (const YAML::Exception &e) {
                    std::cerr << "Warning: failed to load external resonance file \""
                              << res_path << "\": " << e.what() << std::endl;
                }
            } else {
                parseResonances(config["Resonances"]);
            }
        }

        if (config["DecayChains"])
            parseDecayChains(config["DecayChains"]);

        if (config["Constraints"])
            parseConstraints(config["Constraints"]);

        if (config["Plot"])
            parsePlotConfig(config["Plot"]);

        // 势垒因子三级作用域决议（per-step > ResonanceConfig > Constraints 全局 > 默认）
        resolveStepBF();

        // 用户请求的精度（auto=不检查，显式 float/double 时与 .so 比对）
        if (config["precision"])
            precision_ = config["precision"].as<std::string>();

        // 链过滤器: 子串匹配剔除不想要的衰变链（确保 legends/Info/Analysis 一致）。
        // 匹配链名 → 整链保留（原语义）; 匹配"组合级完整路径串"→ 只保留该共振态
        // 组合（该 intermediate 内只留匹配共振态, 其余 intermediate 是链结构、一律
        // 保留）——即"指定具体衰变链, 每步衰变到哪个共振态"。
        // 路径串 = 每步 [mother, d1, d2] 平铺、中间态名用选中共振态替换,
        // 如 psip_gamma_chic1_chic1_eta_f2_1525_f2_1525_Kp_Km（= h_ 波名无前缀）。
        if (!chain_filter_.empty() && !decay_chains_.empty()) {
            // 组合级完整路径串: 每个 intermediate 选一个共振态的笛卡尔积组合,
            // 返回 (路径串, 该组合的 intermediate→共振态 映射)。
            auto combPaths = [](const DecayChainConfig& dc) {
                std::vector<std::pair<std::string, std::vector<std::string>>> choices;
                for (const auto& rc : dc.resonance_chains) {
                    std::vector<std::string> res;
                    for (const auto& sc : rc.spin_chains)
                        for (const auto& r : sc.resonances) res.push_back(r);
                    if (!res.empty()) choices.push_back({rc.intermediate, std::move(res)});
                }
                std::vector<std::map<std::string, std::string>> combos = {{}};
                for (const auto& [iname, res] : choices) {
                    std::vector<std::map<std::string, std::string>> next;
                    next.reserve(combos.size() * res.size());
                    for (const auto& c : combos)
                        for (const auto& r : res) {
                            auto c2 = c;
                            c2[iname] = r;
                            next.push_back(std::move(c2));
                        }
                    combos = std::move(next);
                }
                std::vector<std::pair<std::string, std::map<std::string, std::string>>> out;
                out.reserve(combos.size());
                for (const auto& combo : combos) {
                    auto sub = [&](const std::string& n) {
                        auto it = combo.find(n);
                        return it != combo.end() ? it->second : n;
                    };
                    std::string p;
                    for (const auto& step : dc.decay_steps) {
                        p += sub(step.mother) + "_";
                        for (const auto& d : step.daughters) p += sub(d) + "_";
                    }
                    out.push_back({std::move(p), combo});
                }
                return out;
            };

            std::vector<DecayChainConfig> kept;
            for (auto& dc : decay_chains_) {
                bool keep_whole = false;
                // intermediate → 要保留的共振态（匹配组合的并集）
                std::map<std::string, std::set<std::string>> keep_pairs;
                const auto paths = combPaths(dc);
                for (const auto& pat : chain_filter_) {
                    if (dc.name.find(pat) != std::string::npos) {
                        keep_whole = true;
                        break;
                    }
                    for (const auto& [path, combo] : paths) {
                        if (path.find(pat) != std::string::npos) {
                            for (const auto& [iname, r] : combo)
                                keep_pairs[iname].insert(r);
                        }
                    }
                }
                if (!keep_whole && keep_pairs.empty()) continue; // 无匹配 → 剔除
                if (keep_whole) { kept.push_back(dc); continue; }

                // 只过滤"参与匹配组合"的 intermediate: 其内只留匹配的共振态分波,
                // 剔除空的 spin_chain; 其余 intermediate 的共振态全部保留。
                for (auto& rc : dc.resonance_chains) {
                    auto it = keep_pairs.find(rc.intermediate);
                    if (it == keep_pairs.end()) continue;
                    const auto& keep = it->second;
                    for (auto sit = rc.spin_chains.begin();
                         sit != rc.spin_chains.end();) {
                        std::vector<std::string> kept_r;
                        for (const auto& r : sit->resonances)
                            if (keep.count(r)) kept_r.push_back(r);
                        sit->resonances = kept_r;
                        if (sit->resonances.empty())
                            sit = rc.spin_chains.erase(sit);
                        else
                            ++sit;
                    }
                }
                // 若匹配涉及的 intermediate 全部共振态被过滤掉 → 链不可构建
                bool viable = true;
                for (const auto& rc : dc.resonance_chains) {
                    if (keep_pairs.count(rc.intermediate) && rc.spin_chains.empty()) {
                        viable = false;
                        break;
                    }
                }
                if (viable) kept.push_back(dc);
            }
            std::cout << "Chain filter: " << kept.size() << "/" << decay_chains_.size()
                      << " chains selected" << std::endl;
            decay_chains_ = std::move(kept);
        }
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

        // 处理tex字段，可能是字符串或字符串数组；缺失时默认用粒子名字
        // （与 Resonance 的 tex 处理一致；不写 tex 也应能通过解析）
        if (props["tex"]) {
            if (props["tex"].IsSequence()) {
                // tex是数组
                for (const auto &tex_part : props["tex"]) {
                    particle.tex.push_back(tex_part.as<std::string>());
                }
            } else {
                // 单个字符串，放入向量中
                particle.tex.push_back(props["tex"].as<std::string>());
            }
        } else {
            particle.tex.push_back(name);
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
                    // Must be a sequence of [bachelor, intermediate, {opts}] pairs
                    if (kv.second.size() > 0 &&
                        kv.second[0].IsSequence() &&
                        kv.second[0].size() >= 2) {
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
            // 本块展开链的起点（链名统一加完整中间态路径时只处理本块）
            size_t start_ci = decay_chains_.size();

            // --- Parse intermediate decays ---
            // Single mode:  R_KK: [Kp, Km]  or  [Kp, Km, {opts}]
            // Multi mode:   R1: [[R2, R3], [R4, R5, {p_break: true}]]
            struct IntDecay {
                std::string d1, d2;
                bool has_bf = true;
                bool has_bf_explicit = false; // YAML 是否显式给出 has_bf
                double bf_d = NAN;
                bool p_break = false;
                std::vector<std::vector<int>> sl_filter; // 允许的 [S, L] 分波; 空 = 全允许
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
                        if (dopts["has_bf"]) {
                            id.has_bf = dopts["has_bf"].as<bool>();
                            id.has_bf_explicit = true;
                        }
                        if (dopts["bf_d"]) id.bf_d = dopts["bf_d"].as<double>();
                        if (dopts["p_break"]) id.p_break = dopts["p_break"].as<bool>();
                        if (dopts["sl"]) {
                            // 支持扁平 [S, L] 或嵌套 [[S1, L1], [S2, L2], ...]
                            // config 层 S 为物理自旋（可半整数如 0.5），
                            // 内部 SL.S 用 2S+1 记号 → 解析时转换
                            std::vector<std::vector<double>> raw;
                            if (dopts["sl"][0].IsSequence())
                                raw = dopts["sl"].as<std::vector<std::vector<double>>>();
                            else
                                raw.push_back(dopts["sl"].as<std::vector<double>>());
                            for (const auto& row : raw)
                                if (row.size() >= 2)
                                    id.sl_filter.push_back(
                                        {(int)lround(2.0 * row[0] + 1.0), (int)row[1]});
                        }
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
                                    if (rit->second.J < 0) {
                                        // Resonances 段未写 J/P：auto-detect 无法推断量子数，
                                        // 必须用显式 [J: .., P: ..] 写法
                                        std::cerr << "Error: resonance '" << rname
                                                  << "' has no J/P in Resonances section; "
                                                  << "auto-detect form requires J/P. Use explicit "
                                                  << "[J: .., P: ..]: [" << rname << "] instead."
                                                  << std::endl;
                                        throw std::runtime_error(
                                            "resonance without J/P requires explicit [J,P] form");
                                    }
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

            // --- legends: one per channel, in order ---
            std::vector<std::vector<std::string>> all_legends;
            if (chain_data["legends"] && chain_data["legends"].IsSequence()) {
                for (const auto& leg : chain_data["legends"])
                    all_legends.push_back(leg.as<std::vector<std::string>>());
            }

            // --- Expand each channel: for multi-mode intermediates, generate one chain per mode ---
            size_t legend_rule_idx = 0;  // 展开后的链序号: 每条链按序取一条 legend 规则(该块共用的 legends 列表)
            for (const auto& ch : channels) {
                std::string bachelor = ch[0].as<std::string>();
                std::string intermediate = ch[1].as<std::string>();

                // Parse per-channel constraints (3rd element) — overrides all defaults
                bool ch_has_bf1 = true, ch_has_bf2 = true;
                bool ch_has_bf1_explicit = false, ch_has_bf2_explicit = false;
                double ch_bf_d1 = NAN, ch_bf_d2 = NAN;
                bool ch_p_break1 = false, ch_p_break2 = false;
                bool has_ch_p_break = false;
                std::vector<std::vector<int>> ch_sl_filter; // 第一步 (mother) 的分波白名单
                std::vector<std::string> ch_legend;   // 本行显式 legend 覆盖 (opts["legend"])
                bool ch_legend_from_opts = false;      // opts 显式给出时整行共用、不消耗规则

                if (ch.size() >= 3 && ch[2].IsMap()) {
                    const auto& opts = ch[2];
                    if (opts["legend"]) {
                        ch_legend = opts["legend"].as<std::vector<std::string>>();
                        ch_legend_from_opts = true;
                    }
                    if (opts["has_bf"]) {
                        if (opts["has_bf"].IsSequence()) {
                            ch_has_bf1 = opts["has_bf"][0].as<bool>();
                            ch_has_bf2 = opts["has_bf"][1].as<bool>();
                        } else {
                            ch_has_bf1 = ch_has_bf2 = opts["has_bf"].as<bool>();
                        }
                        ch_has_bf1_explicit = ch_has_bf2_explicit = true;
                    }
                    if (opts["bf_d"]) {
                        if (opts["bf_d"].IsSequence()) {
                            ch_bf_d1 = opts["bf_d"][0].as<double>();
                            ch_bf_d2 = opts["bf_d"][1].as<double>();
                        } else {
                            ch_bf_d1 = ch_bf_d2 = opts["bf_d"].as<double>();
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
                    if (opts["sl"]) {
                        // 支持扁平 [S, L] 或嵌套 [[S1, L1], [S2, L2], ...]（作用于第一步）
                        // config 层 S 为物理自旋（可半整数如 0.5），
                        // 内部 SL.S 用 2S+1 记号 → 解析时转换
                        std::vector<std::vector<double>> raw;
                        if (opts["sl"][0].IsSequence())
                            raw = opts["sl"].as<std::vector<std::vector<double>>>();
                        else
                            raw.push_back(opts["sl"].as<std::vector<double>>());
                        for (const auto& row : raw)
                            if (row.size() >= 2)
                                ch_sl_filter.push_back(
                                    {(int)lround(2.0 * row[0] + 1.0), (int)row[1]});
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

                    // Legend 规则分配: 无显式 opts 覆盖时, 每条展开链按顺序取一条 legend 规则,
                    // 使多模式中间态(如 chic1→eta+R_KK / Kp+R_Keta / Km+R_Keta)各得其规则。
                    if (!ch_legend_from_opts) {
                        ch_legend = (legend_rule_idx < all_legends.size())
                            ? all_legends[legend_rule_idx]
                            : std::vector<std::string>{};
                    }

                    DecayChainConfig chain;
                    chain.name = chain_name;  // 占位; 完整中间态路径名在块末统一生成

                    // --- Step 1: mother → bachelor + intermediate ---
                    DecayStep step1;
                    step1.mother = mother;
                    step1.daughters = {bachelor, intermediate};
                    step1.has_bf = ch_has_bf1;
                    step1.has_bf_explicit = ch_has_bf1_explicit;
                    step1.bf_d = ch_bf_d1;
                    step1.p_break = ch_p_break1;
                    step1.sl_filter = ch_sl_filter;
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
                        decay_chains_.push_back(chain);
                        continue;
                    }

                    // --- BFS: recursively resolve intermediates ---
                    // used: 链中已出现的粒子(行级 bachelor + 已展开模式的子粒子)。
                    // 深层多模式中间态选模式时避开 used——与行级 multi-mode filter
                    // (模式不得含 bachelor, 否则同一粒子重复出现) 同一约束的递归推广。
                    struct BFSItem {
                        std::string name;
                        bool is_first;
                        bool has_bf, has_bf_explicit, pb;
                        double bf_d;
                        std::vector<std::vector<int>> sl; // 该步的分波白名单
                        std::vector<std::string> used;    // 链中已出现的粒子
                    };
                    std::queue<BFSItem> queue;

                    if (intermediate_decays) {
                        const IntDecay* ifm = getDecayMode(intermediate,
                            bachelor_drives_modes ? 0 : mi);
                        bool step2_has_bf = ch_has_bf2_explicit ? ch_has_bf2
                            : (ifm && ifm->has_bf_explicit ? ifm->has_bf : true);
                        bool step2_has_bf_explicit = ch_has_bf2_explicit
                            || (ifm && ifm->has_bf_explicit);
                        double step2_bf_d = !std::isnan(ch_bf_d2) ? ch_bf_d2
                            : (ifm && !std::isnan(ifm->bf_d) ? ifm->bf_d : NAN);
                        bool step2_p_break = has_ch_p_break ? ch_p_break2 : (ifm ? ifm->p_break : false);
                        queue.push({intermediate, !bachelor_drives_modes, step2_has_bf,
                            step2_has_bf_explicit, step2_p_break, step2_bf_d,
                            ifm ? ifm->sl_filter : std::vector<std::vector<int>>{},
                            std::vector<std::string>{bachelor}});
                    }
                    if (bachelor_decays) {
                        const IntDecay* bm = getDecayMode(bachelor,
                            bachelor_drives_modes ? mi : 0);
                        queue.push({bachelor, bachelor_drives_modes,
                            bm && bm->has_bf_explicit ? bm->has_bf : true,
                            bm && bm->has_bf_explicit,
                            bm ? bm->p_break : false,
                            (bm && !std::isnan(bm->bf_d)) ? bm->bf_d : NAN,
                            bm ? bm->sl_filter : std::vector<std::vector<int>>{},
                            std::vector<std::string>{intermediate}});
                    }

                    while (!queue.empty()) {
                        auto item = queue.front(); queue.pop();

                        // 模式选择: 首层中间态用外层模式 mi (行级 filter 已保证其
                        // 不含行 bachelor); 深层中间态避开链中已出现的粒子。
                        size_t mode_idx = 0;
                        if (item.is_first) {
                            mode_idx = mi;
                        } else if (!item.used.empty()) {
                            const auto& modes = int_decay_modes[item.name];
                            bool all_conflict = true;
                            for (size_t k = 0; k < modes.size(); ++k) {
                                bool conflict = false;
                                for (const auto& u : item.used) {
                                    if (modes[k].d1 == u || modes[k].d2 == u) {
                                        conflict = true;
                                        break;
                                    }
                                }
                                if (!conflict) {
                                    mode_idx = k;
                                    all_conflict = false;
                                    break;
                                }
                            }
                            if (mode_idx > 0)
                                std::cerr << "Note: intermediate '" << item.name
                                          << "' uses decay mode " << mode_idx
                                          << " (avoids particles already in chain)"
                                          << std::endl;
                            else if (all_conflict)
                                std::cerr << "Warning: intermediate '" << item.name
                                          << "': all decay modes contain a particle "
                                          << "already in the chain; using mode 0 (duplicate "
                                          << "particle may appear)" << std::endl;
                        }
                        const IntDecay* mode = getDecayMode(item.name, mode_idx);
                        if (!mode) continue;

                        DecayStep substep;
                        substep.mother = item.name;
                        substep.daughters = {mode->d1, mode->d2};
                        substep.has_bf = item.has_bf;
                        substep.has_bf_explicit = item.has_bf_explicit;
                        substep.bf_d = item.bf_d;
                        substep.p_break = item.pb;
                        substep.sl_filter = item.sl;
                        chain.decay_steps.push_back(substep);

                        // Resonance chain for this intermediate
                        if (res_chain_map.count(item.name))
                            chain.resonance_chains.push_back(res_chain_map[item.name]);

                        // Enqueue any daughter that is itself an intermediate.
                        // 本模式的子粒子已出现在链中, 传给后代的 used。
                        std::vector<std::string> child_used = item.used;
                        child_used.push_back(mode->d1);
                        child_used.push_back(mode->d2);
                        auto enqueue = [&](const std::string& d) {
                            if (int_decay_modes.count(d)) {
                                const auto* dm = getDecayMode(d, 0);
                                queue.push({d, false,
                                    dm && dm->has_bf_explicit ? dm->has_bf : true,
                                    dm && dm->has_bf_explicit,
                                    dm ? dm->p_break : false,
                                    (dm && !std::isnan(dm->bf_d)) ? dm->bf_d : NAN,
                                    dm ? dm->sl_filter : std::vector<std::vector<int>>{},
                                    child_used});
                            }
                        };
                        enqueue(mode->d1);
                        enqueue(mode->d2);
                    }

                    // --- Legend ---
                    if (!ch_legend.empty())
                        chain.legend_template = ch_legend;
                    else
                        chain.legend_template = {intermediate, " ", bachelor};

                    decay_chains_.push_back(chain);
                    if (!ch_legend_from_opts) ++legend_rule_idx;
                } // for each mode

            // ============================================================
            // 链名: 完整中间态路径（去重保序）+ 同名 0-based 序号
            // 旧实现只含第一级中间态 + 行模式序号 (decay1_R_chic1_0/1/2),
            // 二级中间态不同 (R_KK vs R_Keta) 时名字无法区分。新名字:
            //   decay1_R_chic1_R_KK / decay1_R_chic1_R_Keta_0 /
            //   decay1_R_chic1_R_Keta_1
            // （Constraints.trans/chains 按"中间态名_序号"匹配, 不受影响;
            //   链过滤器按完整路径串或新链名子串匹配）
            // ============================================================
            auto chainBaseName = [&](const DecayChainConfig& ch) -> std::string {
                std::string base = chain_name;
                std::set<std::string> seen_m;
                // step1 的中间态子（非已知粒子 daughter, 如 R_chic1）
                for (const auto& d : ch.decay_steps[0].daughters)
                    if (!particle_names.count(d) && seen_m.insert(d).second)
                        base += "_" + d;
                // 后续衰变步的中间态（R_KK/R_Keta 等, 与 step1 去重）
                for (size_t si = 1; si < ch.decay_steps.size(); ++si)
                    if (seen_m.insert(ch.decay_steps[si].mother).second)
                        base += "_" + ch.decay_steps[si].mother;
                return base;
            };
            std::map<std::string, int> name_count;
            for (size_t ci = start_ci; ci < decay_chains_.size(); ++ci)
                name_count[chainBaseName(decay_chains_[ci])]++;
            std::map<std::string, int> name_seen;
            for (size_t ci = start_ci; ci < decay_chains_.size(); ++ci) {
                auto& ch = decay_chains_[ci];
                std::string base = chainBaseName(ch);
                int k = name_seen[base]++;
                ch.name = (name_count[base] > 1)
                    ? base + "_" + std::to_string(k) : base;
            }
            } // if (is_compact)
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
                            if (decay_pair.second["has_bf"]) {
                                step.has_bf = decay_pair.second["has_bf"].as<bool>();
                                step.has_bf_explicit = true;
                            }
                            if (decay_pair.second["bf_d"])
                                step.bf_d = decay_pair.second["bf_d"].as<double>();
                            if (decay_pair.second["p_break"])
                                step.p_break =
                                    decay_pair.second["p_break"].as<bool>();
                            if (decay_pair.second["sl"]) {
                                // 支持扁平 [S, L] 或嵌套 [[S1, L1], [S2, L2], ...]
                                if (decay_pair.second["sl"][0].IsSequence())
                                    step.sl_filter = decay_pair.second["sl"]
                                        .as<std::vector<std::vector<int>>>();
                                else
                                    step.sl_filter.push_back(
                                        decay_pair.second["sl"]
                                            .as<std::vector<int>>());
                            }
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
        // J/P 可选：不写时 J=-1（哨兵）、P=0（哨兵）——由 intermediates 的
        // [J, P] 决定量子数（CP 共轭对等场景：同一共振 N 在两条链中宇称相反）。
        // auto-detect（intermediates 不带 [J,P] 只写名字）依赖此字段分组，
        // 哨兵值在该路径下会被报错跳过（见下方 jp_groups 检查）。
        if (props["J"])
            res.J = static_cast<int>(2 * transJValue(props["J"].as<std::string>()) + 1);
        else
            res.J = -1;
        res.P = props["P"] ? props["P"].as<int>() : 0;
        res.type = props["model"].as<std::string>();
        // 参数值: paramValues（新规范名）或 parameters（旧名，兼容）
        if (props["paramValues"])
            res.parameters = props["paramValues"].as<std::vector<double>>();
        else if (props["parameters"])
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
        // Custom 模型: paramNames(参数名列表) 和 expr(表达式)
        // （旧名 params 兼容）
        YAML::Node pnames = props["paramNames"] ? props["paramNames"]
                                                : props["params"];
        if (pnames) {
            auto pl = pnames.as<std::vector<std::string>>();
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

        // 势垒因子（此共振态作为中间态时）: has_bf / bf_d
        if (props["has_bf"]) {
            res.has_bf = props["has_bf"].as<bool>();
            res.has_bf_explicit = true;
        }
        if (props["bf_d"]) {
            res.bf_d = props["bf_d"].as<double>();
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

void ConfigParser::resolveStepBF()
{
    // 三级作用域: per-step(已在解析时写入) > ResonanceConfig(母粒子) > Constraints 全局 > 默认
    // 母粒子名 M 查找 ResonanceConfig 的规则:
    //   1. 直接按名字找（Resonances 段以中间态名作键，如 "R_KK"）
    //   2. 否则在该链的 resonance_chains 中找 intermediate == M 的条目，
    //      取其第一个共振态名的配置（多共振态共享同一中间态时取第一个；
    //      目标节点自身的 Bf 由各共振态自己的 DSL 决定，不受此限制）
    for (auto& chain : decay_chains_) {
        for (auto& step : chain.decay_steps) {
            const ResonanceConfig* cfg = nullptr;
            auto it = resonances_.find(step.mother);
            if (it != resonances_.end()) {
                cfg = &it->second;
            } else {
                for (const auto& rc : chain.resonance_chains) {
                    if (rc.intermediate != step.mother || rc.spin_chains.empty())
                        continue;
                    const auto& rnames = rc.spin_chains[0].resonances;
                    if (!rnames.empty()) {
                        auto rit = resonances_.find(rnames[0]);
                        if (rit != resonances_.end()) cfg = &rit->second;
                    }
                    break;
                }
            }
            if (!step.has_bf_explicit) {
                if (cfg && cfg->has_bf_explicit) step.has_bf = cfg->has_bf;
                else step.has_bf = global_has_bf_;
            }
            if (std::isnan(step.bf_d)) {
                if (cfg && !std::isnan(cfg->bf_d)) step.bf_d = cfg->bf_d;
                else step.bf_d = global_bf_d_;
            }
        }
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

    // 解析全局势垒开关 has_bf（默认开启）
    if (node["has_bf"]) {
        global_has_bf_ = node["has_bf"].as<bool>();
    }

    // ---- 命名变量约束（theta 参数，按 "resName_paramName" 匹配）----
    // fix_var: {rho_770_mass: 0.775, ...} — 固定参数值
    if (node["fix_var"]) {
        for (const auto& kv : node["fix_var"]) {
            fix_var_[kv.first.as<std::string>()] = kv.second.as<double>();
        }
    }
    // free_var: [name, ...] — 取消 fix_var 的固定
    if (node["free_var"]) {
        for (const auto& nm : node["free_var"]) {
            free_var_.insert(nm.as<std::string>());
        }
    }
    // var_range: {name: [lower, upper], ...} — 覆盖拟合范围
    if (node["var_range"]) {
        for (const auto& kv : node["var_range"]) {
            const auto& range = kv.second;
            if (range.IsSequence() && range.size() >= 2) {
                var_range_[kv.first.as<std::string>()] =
                    { range[0].as<double>(), range[1].as<double>() };
            } else {
                std::cerr << "Warning: var_range 项 \"" << kv.first.as<std::string>()
                          << "\" 应为 [lower, upper]，已忽略" << std::endl;
            }
        }
    }
    // var_equal: [[n1, n2, ...], ...] — 一组参数共享（owner = 组内第一个有槽的名字）
    if (node["var_equal"]) {
        for (const auto& group : node["var_equal"]) {
            std::vector<std::string> names;
            for (const auto& nm : group) names.push_back(nm.as<std::string>());
            if (names.size() >= 2) var_equal_.push_back(std::move(names));
        }
    }
    // gauss_constr: {name: sigma, ...} — 高斯罚项 Σ(x-μ)²/(2σ²)，μ = 初始值
    if (node["gauss_constr"]) {
        for (const auto& kv : node["gauss_constr"]) {
            gauss_constr_[kv.first.as<std::string>()] = kv.second.as<double>();
        }
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

    // ---- 统一形式: Plot 直接是观测列表（1d=一个 expr 字符串, 2d=两个字符串）----
    // 例:
    //   Plot:
    //     - expr: "M(p,pbar)"
    //       bins: [60]; ranges: [[1.8, 3.0]]
    //     - expr: ["M(p,pbar)", "CosAngle(p; [p,pbar])"]
    //       bins: [60, 50]; ranges: [[1.8, 3.0], [-1, 1]]
    if (node.IsSequence()) {
        int idx = 0;
        for (const auto &plot_item : node) {
            PlotConfig config;
            config.type = "obs";
            if (plot_item["name"])
                config.name = plot_item["name"].as<std::string>();

            const YAML::Node &expr_node = plot_item["expr"]
                ? plot_item["expr"] : plot_item["expression"];  // expression 为别名(与 Custom 一致)
            std::vector<std::string> exprs;
            if (!expr_node) {
                std::cerr << "Warning: Plot 项缺少 expr/expression, 跳过" << std::endl;
                continue;
            }
            if (expr_node.IsScalar())
                exprs.push_back(expr_node.as<std::string>());
            else
                exprs = expr_node.as<std::vector<std::string>>();
            for (const auto &e : exprs)
                config.obs.push_back(parseObsExpr(e));
            if (config.obs.empty() || config.obs.size() > 2) {
                std::cerr << "Warning: expr 需要 1 (1d) 或 2 (2d) 个表达式, 实际 "
                          << config.obs.size() << ", 跳过" << std::endl;
                continue;
            }

            if (plot_item["bins"].IsScalar())
                config.bins = { plot_item["bins"].as<int>() };
            else
                config.bins = plot_item["bins"].as<std::vector<int>>();
            if (config.bins.size() != config.obs.size()) {
                std::cerr << "Warning: bins 数量(" << config.bins.size()
                          << ") != 维度(" << config.obs.size()
                          << "), 补到一样长" << std::endl;
                while ((int)config.bins.size() < (int)config.obs.size())
                    config.bins.push_back(config.bins.back());
            }

            if (plot_item["ranges"]) {
                for (const auto &r : plot_item["ranges"])
                    config.ranges.push_back(r.as<std::vector<double>>());
            } else if (plot_item["range"]) {   // 单数 range 为别名 (与旧格式一致)
                const auto &rn = plot_item["range"];
                if (rn.IsSequence() && rn[0].IsSequence()) {
                    for (const auto &r : rn)
                        config.ranges.push_back(r.as<std::vector<double>>());
                } else {
                    config.ranges.push_back(rn.as<std::vector<double>>());
                }
            }
            if (config.ranges.empty()) {
                std::cerr << "Warning: Plot 项 '"
                          << (config.name.empty() ? "obs" + std::to_string(idx) : config.name)
                          << "' 缺少 range/ranges (检查拼写 rages?), 跳过" << std::endl;
                continue;
            }
            if ((int)config.ranges.size() != (int)config.obs.size()) {
                std::cerr << "Warning: Plot 项 '"
                          << (config.name.empty() ? "obs" + std::to_string(idx) : config.name)
                          << "' ranges 数量(" << config.ranges.size()
                          << ") != 维度(" << config.obs.size() << "), 跳过" << std::endl;
                continue;
            }
            if (plot_item["display"])
                config.display =
                    plot_item["display"].as<std::vector<std::string>>();
            if (config.name.empty())
                config.name = "obs" + std::to_string(idx);
            ++idx;
            plot_configs_.push_back(config);
        }
        return;
    }

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
