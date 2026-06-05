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
    void print() const;

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
