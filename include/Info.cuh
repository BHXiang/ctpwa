#ifndef INFO_CUH
#define INFO_CUH

#include <Config.cuh>
#include <Parameters.cuh>
#include <Resonance.cuh>
#include <map>
#include <string>
#include <vector>

// Built by DecayInfo, consumed by Analysis::calculateAmplitudes
struct ChainInfo {
    std::string name;
    std::map<std::pair<std::string, std::vector<int>>, std::vector<Resonance>>
        intermediate_resonance_map;
    std::vector<std::vector<Particle>> intermediate_combs;
    std::vector<std::string> amplitude_names;  // 该链的振幅名(_LS 格式)
};

// 用户侧链视图（分层浏览用）: 一条展开链的只读结构 + 按需取完整链串/波名
struct ChainView {
    std::string name;                          // 链名(与 chains 子串过滤一致)
    std::string topology;                      // 拓扑串 "Jpsi→R+γ_R→d1+d2_..."(与旧 print 一致)
    std::vector<std::string> steps;            // 每步 "mother->d1+d2"(含 sl/ls 白名单文本)
    std::vector<std::string> intermediates;    // 每行 "int: [J,P]: res1 res2 ..."
    std::vector<std::string> amplitude_names;  // 该链波名(_LS 格式)
    int n_resonances = 0;                      // 该链共振态总数
    std::vector<std::string> exact_chain_strings; // 全部完整链串(与 chains_exact 同一实现, 预生成)

    // 完整链串: containing 非空时只返回含它的串
    std::vector<std::string> exactchains(const std::string &containing = "") const;
    std::vector<std::string> amplitudes() const { return amplitude_names; }
    // counts = {中间态数, 共振态总数, 完整链串数, 振幅数}
    std::vector<int> counts() const;
    void print() const;
};

class DecayInfo {
public:
    DecayInfo(const std::string& config_file = "config.yml");
    DecayInfo(const ConfigParser& parser);  // share existing parser

    bool isValid() const { return initialized_; }

    // Particles
    const std::vector<Particle>& particles() const { return particles_; }

    // Decay chains
    const std::vector<ChainInfo>& chainInfos() const { return chains_info_; }

    // Amplitude names (old-style, per-resonance×SL)
    const std::vector<std::string>& amplitudeNames() const { return amplitude_names_; }
    int nAmplitudes() const { return static_cast<int>(amplitude_names_.size()); }

    // Resonance names
    const std::vector<std::string>& resonanceNames() const { return resonance_names_; }

    // Coupling matrix parameter names (chain + step params)
    const std::vector<std::string>& paramNames() const { return param_names_; }
    int nFreeParams() const { return static_cast<int>(param_names_.size()); }

    // Coupling matrix details
    const CouplingMatrixResult& couplingMatrix() const { return coupling_matrix_; }
    bool hasCouplingMatrix() const { return use_coupling_matrix_; }

    // Constraints
    const std::vector<ConstraintConfig>& constraints() const { return constraints_; }

    // Resonance param info: {init_value, lower, upper} per free theta param
    const std::vector<std::string>& resonanceParamNames() const { return resonance_param_names_; }

    // Number of SL combinations per partial wave
    const std::vector<int>& nSLvectors() const { return nsl_vectors_; }

    // Print summary
    void print(int level = 1) const;   // 0=总览 1=链概览(默认) 2=链明细+完整链串 3=全部振幅
    void summary() const;              // 只打总览

    // 分层浏览（层级对象, 对应 Python: d.chains[i].exactchains(...)）
    std::vector<ChainView> chains() const;
    // 完整链串: chain<0 = 全部链; containing 非空时只返回包含它的串
    std::vector<std::string> exactchains(int chain = -1,
                                         const std::string &containing = "") const;
    // 扁平打印全部完整链串(一行一条, 首行 # 计数; containing 非空时过滤)
    // 输出可直接保存为 chains_exact 外部文件(自动跳过 # 注释)
    void printExactChains(const std::string &containing = "") const;
    // 扁平打印全部拟合参数名(chain×step 耦合段: 链参数+步参数; 末尾共振态 θ 段;
    // 下标与拟合参数向量/parameters.txt 一致; containing 非空时只打印含它的参数)
    void printParamNames(const std::string &containing = "") const;
    // 波名: chain<0 = 全部链; resonance 非空时只返回名字含该子串的
    std::vector<std::string> amplitudes(int chain = -1,
                                        const std::string &resonance = "") const;

private:
    void initialize(const std::string& config_file);
    void buildDecayChains(const std::vector<DecayChainConfig>& chains,
                          const std::map<std::string, ResonanceConfig>& config_resonances,
                          int global_max_l);

    ConfigParser config_parser_;
    std::vector<Particle> particles_;
    std::vector<ChainInfo> chains_info_;
    std::vector<std::string> amplitude_names_;
    std::vector<std::string> resonance_names_;
    std::vector<std::string> param_names_;
    std::vector<std::string> resonance_param_names_;
    std::vector<int> nsl_vectors_;
    std::vector<ConstraintConfig> constraints_;
    CouplingMatrixResult coupling_matrix_;
    CouplingMatrixBuilder coupling_matrix_builder_;
    std::map<std::string, std::string> chain_display_map_;
    bool use_coupling_matrix_ = false;
    int n_chain_free_after_trans_ = 0;
    bool initialized_ = false;
};

#endif // INFO_CUH
