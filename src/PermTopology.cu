// src/PermTopology.cu — 全同粒子置换拓扑（纯 host，无 cuda 依赖）
#include <PermTopology.h>

#include <algorithm>
#include <iostream>
#include <set>

namespace ctpwa {

PermTopology buildPermTopology(const PermTopologyInputs& in)
{
    PermTopology out;
    out.maps.clear();
    out.signs.clear();
    out.signs.push_back(1.0);  // σ=0：恒等

    if (in.identicalGroups.empty()) return out;

    std::string root = in.decayChain[0].mother;

    // 链中粒子集合（校验用）
    std::set<std::string> in_chain;
    for (const auto& node : in.decayChain) {
        in_chain.insert(node.mother);
        in_chain.insert(node.daug1);
        in_chain.insert(node.daug2);
    }

    // 一个成员的子树：path（相对该成员的"左/右"走向序列）→ 粒子名 + 逆映射
    struct Subtree {
        std::string name;
        std::map<std::vector<int>, std::string> paths;      // path → name
        std::map<std::string, std::vector<int>> name_path;  // name → path
    };

    // 每组生成 coset（每个成员 i 的槽位接收成员 σ(i) 的内容）
    for (const auto& [members, is_boson] : in.identicalGroups) {
        std::vector<Subtree> subs;
        for (const auto& m : members) {
            if (m == root || in_chain.find(m) == in_chain.end()) continue;
            Subtree s;
            s.name = m;
            subs.push_back(s);
        }
        if (subs.size() < 2) continue;  // 单成员组无置换

        // 子树路径传播（父→子，循环直至不动点）
        for (auto& s : subs) {
            std::vector<int> empty;
            s.paths[empty] = s.name;
            s.name_path[s.name] = empty;
            bool changed = true;
            while (changed) {
                changed = false;
                for (const auto& node : in.decayChain) {
                    auto it = s.name_path.find(node.mother);
                    if (it == s.name_path.end()) continue;
                    std::vector<int> p1 = it->second, p2 = it->second;
                    p1.push_back(1);
                    p2.push_back(2);
                    if (s.paths.find(p1) == s.paths.end()) {
                        s.paths[p1] = node.daug1;
                        s.name_path[node.daug1] = p1;
                        changed = true;
                    }
                    if (s.paths.find(p2) == s.paths.end()) {
                        s.paths[p2] = node.daug2;
                        s.name_path[node.daug2] = p2;
                        changed = true;
                    }
                }
            }
        }

        // 一致性：所有成员子树结构必须相同（全同粒子必须有相同的衰变结构）
        const auto& ref_paths = subs[0].paths;
        for (const auto& s : subs) {
            if (s.paths.size() != ref_paths.size()) {
                std::cerr << "Error: identical group members have different decay structures ("
                          << subs[0].name << " vs " << s.name << ")" << std::endl;
                return out;  // 安全失败：不做对称化
            }
            for (const auto& [p, n] : ref_paths)
                if (s.paths.find(p) == s.paths.end()) {
                    std::cerr << "Error: identical group members have different decay structures ("
                              << subs[0].name << " vs " << s.name << ")" << std::endl;
        return out;
                }
        }

        // 嵌套检查：一个成员的子树内不能有同组另一成员
        for (size_t i = 0; i < subs.size(); ++i)
            for (size_t j = 0; j < subs.size(); ++j)
                if (i != j && subs[i].name_path.count(subs[j].name)) {
                    std::cerr << "Error: identical group members nested inside each other ("
                              << subs[i].name << " contains " << subs[j].name << ")" << std::endl;
        return out;
                }

        // 稳定子：同一步的两个子粒子互为同组成员 → 该换位是稳定子
        std::vector<std::pair<int, int>> stab_pairs;
        for (const auto& node : in.decayChain) {
            int i1 = -1, i2 = -1;
            for (size_t i = 0; i < subs.size(); ++i) {
                if (subs[i].name == node.daug1) i1 = static_cast<int>(i);
                if (subs[i].name == node.daug2) i2 = static_cast<int>(i);
            }
            if (i1 >= 0 && i2 >= 0) stab_pairs.emplace_back(i1, i2);
        }
        // 稳定子元素 = 各换位的任意乘积（不相交 → 直接枚举子集）
        int n = static_cast<int>(subs.size());
        std::vector<int> id_perm(n);
        for (int i = 0; i < n; ++i) id_perm[i] = i;
        std::vector<std::vector<int>> stab_elts = { id_perm };
        for (const auto& [a, b] : stab_pairs) {
            size_t cur = stab_elts.size();
            for (size_t k = 0; k < cur; ++k) {
                std::vector<int> t = stab_elts[k];
                std::swap(t[a], t[b]);
                stab_elts.push_back(t);
            }
        }

        // 枚举 S_n 全部置换，按稳定子取陪集代表元（canonical = 轨道内字典序最小）
        auto canonical = [&](const std::vector<int>& p) {
            std::vector<int> best = p;
            for (const auto& t : stab_elts) {
                std::vector<int> q(n);
                for (int i = 0; i < n; ++i) q[i] = p[t[i]];
                if (q < best) best = q;
            }
            return best;
        };

        std::vector<std::vector<int>> reps;
        std::set<std::vector<int>> seen;
        std::vector<int> perm = id_perm;
        do {
            auto c = canonical(perm);
            if (seen.insert(c).second) reps.push_back(c);
        } while (std::next_permutation(perm.begin(), perm.end()));

        // 每组生成自己的 σ≥1 重建动量映射 + 符号。
        // tf-pwa 约定：全同粒子振幅和 = Σ_σ A(x_σ)，对全置换群求和（不做陪集约化）。
        // 因此陪集代表元 r 要乘上全部稳定子元素 s（r∘s），得到全群元素；
        // 稳定子元素（如 f 链的 Ks1↔Ks2 交换）必须计入——它给出 A_f(x) 的加倍项。
        std::vector<std::map<std::string, int>> group_maps;
        std::vector<double> group_signs;
        std::set<std::vector<int>> seen_group;
        for (const auto& rep : reps) {
            for (const auto& st : stab_elts) {
                std::vector<int> g(n);
                for (int i = 0; i < n; ++i) g[i] = rep[st[i]];
                if (g == id_perm) continue;  // 纯恒等 = σ=0，已在 out.signs[0]
                if (!seen_group.insert(g).second) continue;  // 全群元素去重
                // 符号：费米子 sgn(g)（逆序数奇偶），玻色子 +1
                double sg = 1.0;
                if (!is_boson) {
                    int inv = 0;
                    for (int i = 0; i < n; ++i)
                        for (int j = i + 1; j < n; ++j)
                            if (g[i] > g[j]) ++inv;
                    sg = (inv % 2 == 0) ? 1.0 : -1.0;
                }
                // 重建动量映射：位置 i 的子树内容换成成员 g[i] 的子树（路径对应）。
                // 映射语义（与 convertToDeviceMomenta 一致）：mapping[name] = 动量新槽位。
                // 必须覆盖全部粒子（恒等底 + 组内交换）——convertToDeviceMomenta 用 map 尺寸分配数组
                std::map<std::string, int> pmap = in.particleToIndex;
                for (int i = 0; i < n; ++i) {
                    int j = g[i];  // 成员 subs[j] 的内容移到位置 i
                    for (const auto& [path, name] : subs[i].paths) {
                        auto it = subs[j].paths.find(path);
                        if (it == subs[j].paths.end()) continue;  // 已校验一致，不会发生
                        pmap[it->second] = in.particleToIndex.at(name);
                    }
                }
                group_maps.push_back(pmap);
                group_signs.push_back(sg);
            }
        }

        // 与已有组叠加（多组笛卡尔积；槽位不相交）
        if (group_maps.empty()) continue;
        if (out.maps.empty()) {
            out.maps = group_maps;
            out.signs = { 1.0 };
            for (auto sg : group_signs) out.signs.push_back(sg);
        } else {
            std::vector<std::map<std::string, int>> new_maps;
            std::vector<double> new_signs;
            for (size_t a = 0; a < out.maps.size(); ++a)
                for (size_t b = 0; b < group_maps.size(); ++b) {
                    std::map<std::string, int> merged = group_maps[b];
                    for (const auto& [k, v] : out.maps[a]) merged[k] = v;
                    new_maps.push_back(merged);
                    new_signs.push_back(out.signs[a + 1] * group_signs[b]);
                }
            out.maps = std::move(new_maps);
            out.signs.clear();
            out.signs.push_back(1.0);
            for (auto sg : new_signs) out.signs.push_back(sg);
        }

        // 仅当存在非平凡陪集（reps.size()>1，成员交换产生新拓扑）时输出；
        // 纯稳定子组（如 f0→Ks Ks 的兄弟交换，仅一个陪集）视为不交换，不打印。
        // （注意：纯稳定子组的换位项仍会叠加进 out.maps 参与求和——tf-pwa 约定
        //   Σ_σ 对全置换群求和，A(x) 加倍项必须保留，这里只控制输出。）
        if (reps.size() <= 1) continue;
        std::cout << "Identical group (" << (is_boson ? "boson" : "fermion") << "): [";
        for (size_t i = 0; i < subs.size(); ++i) {
            if (i) std::cout << ", ";
            std::cout << subs[i].name;
        }
        std::cout << "] in chain " << in.chainName << std::endl;
    }

    if (out.signs.size() > 33)
        std::cerr << "Warning: " << out.signs.size() - 1
                  << " permutation topologies — memory usage scales with this!" << std::endl;

    return out;
}

}  // namespace ctpwa
