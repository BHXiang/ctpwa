#ifndef INFO_CUH
#define INFO_CUH

#include <Config.cuh>
#include <Resonance.cuh>
#include <Parameters.cuh>
#include <map>
#include <string>
#include <vector>

class DecayInfo {
public:
    DecayInfo(const std::string& config_file = "config.yml");

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

    // Print summary
    void print() const;

private:
    void initialize(const std::string& config_file);

    ConfigParser config_parser_;
    std::vector<Particle> particles_;
    std::vector<ChainInfo> chains_info_;
    std::vector<std::string> amplitude_names_;
    std::vector<std::string> resonance_names_;
    std::vector<std::string> param_names_;
    std::vector<std::string> resonance_param_names_;
    std::vector<ConstraintConfig> constraints_;
    CouplingMatrixResult coupling_matrix_;
    CouplingMatrixBuilder coupling_matrix_builder_;
    bool use_coupling_matrix_ = false;
    bool initialized_ = false;
};

#endif // INFO_CUH
