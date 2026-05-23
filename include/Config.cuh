#ifndef CONFIG_CUH
#define CONFIG_CUH

#include <AmpGen.cuh>
#include <map>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

struct Particle {
  std::string name;
  int spin;       // 2J+1
  int parity;
  double mass;
  std::string identical_group; // non-empty if this particle belongs to an identical group
  std::vector<int> polarization_2m; // allowed 2m values; empty = all states
  std::vector<std::string> tex;
  bool is_fermion() const { return (spin - 1) % 2 != 0; } // half-integer spin -> fermion
  bool is_polarized() const { return !polarization_2m.empty(); }
};

struct ResonanceConfig {
  std::string name;
  int J;
  int P;
  std::string type;
  std::vector<double> parameters;
  std::vector<std::vector<double>> channels; // Flatte: [[m_a, m_b], ...]; 非Flatte为空
  std::vector<std::string> tex;
  std::vector<int> scan; // 需要scan的参数下标; {-1}=全扫; 空=不扫
};

struct SpinChainConfig {
  std::vector<int> spin_parity;
  std::vector<std::string> resonances;
};

struct ResonanceChainConfig {
  std::string intermediate;
  std::vector<SpinChainConfig> spin_chains;
};

struct DecayStep {
  std::string mother;
  std::vector<std::string> daughters;
  bool is_bf = true;    // 是否施加势垒因子
  bool p_break = false; // 是否宇称破缺（如弱衰变）
};

struct DecayChainConfig {
  std::string name;
  std::vector<DecayStep> decay_steps;
  std::vector<ResonanceChainConfig> resonance_chains;
  std::vector<std::string> legend_template;
  bool symmetrize = false; // enable identical particle symmetrization for this chain
};

struct ConstraintConfig {
  std::vector<std::string> names;
  std::vector<std::complex<double>> values;
  std::string type;
};

struct PlotConfig {
  std::vector<std::vector<std::string>> particles;
  std::vector<int> bins;
  std::vector<std::vector<double>> ranges;
  std::vector<std::string> display;
  std::string type; // "mass", "cosbeta", "dalitz"
};

class ConfigParser {
public:
  ConfigParser() = default;
  ConfigParser(const std::string &config_file);

  bool isValid() const { return !particles_.empty(); }

  const std::vector<Particle> &getParticles() const { return particles_; }
  const std::vector<DecayChainConfig> &getDecayChains() const {
    return decay_chains_;
  }
  const std::map<std::string, ResonanceConfig> &getResonances() const {
    return resonances_;
  }
  const std::map<std::string, std::vector<std::string>> &getDataFiles() const {
    return data_files_;
  }
  const std::vector<std::string> &getDataOrder() const { return data_order_; }
  const std::vector<ConstraintConfig> &getConstraints() const {
    return constraints_;
  }
  int getGlobalMaxL() const { return global_maxL_; }
  double getBfD() const { return global_bf_d_; }
  const std::vector<PlotConfig> &getPlotConfigs() const {
    return plot_configs_;
  }

  std::vector<std::string> getLegends() const;
  std::string generateLegend(const std::vector<std::string> &particles) const;
  // 返回全同粒子分组: group_name -> {particle_name, ...}
  std::map<std::string, std::vector<std::string>> getIdenticalGroups() const;

private:
  // 解析函数
  void parseParticles(const YAML::Node &node);
  void parseData(const YAML::Node &node);
  void parseDecayChains(const YAML::Node &node);
  void parseResonances(const YAML::Node &node);
  void parseConstraints(const YAML::Node &node);
  void parsePlotConfig(const YAML::Node &node);

  std::vector<Particle> particles_;
  std::vector<DecayChainConfig> decay_chains_;
  std::map<std::string, ResonanceConfig> resonances_;
  std::map<std::string, std::vector<std::string>> data_files_;
  std::vector<std::string> data_order_;
  std::vector<ConstraintConfig> constraints_;
  std::vector<PlotConfig> plot_configs_;
  int global_maxL_ = -1; // -1 = no limit; set via Constraints.maxL
  double global_bf_d_ = 3.0; // barrier factor d; set via Constraints.bf_d
};

#endif // CONFIG_CUH