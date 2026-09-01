#ifndef CONFIG_CUH
#define CONFIG_CUH

#include <map>
#include <set>
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
  std::vector<int> free; // 需要拟合的参数下标; {-1}=全部; 空=不拟合
  std::vector<std::vector<double>> free_range; // 每个free参数的 [lower, upper]; 空=使用默认
  std::map<std::string, std::string> options; // 模型选项 (Hist: file/bins/range/extrapolate)
  // 势垒因子（此共振态作为中间态时）: has_bf 是否施加；bf_d 势垒半径
  // （NAN = 未显式设置 → 由三级作用域决议: per-step > resonance > Constraints 全局 > 默认）
  bool has_bf = true;
  bool has_bf_explicit = false;
  double bf_d = NAN;
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
  // 势垒因子: has_bf 是否施加；bf_d 势垒半径（NAN = 未显式设置）。
  // parse() 末尾 resolveStepBF 按三级作用域决议出最终值:
  // per-step > ResonanceConfig(母粒子) > Constraints 全局 > 默认 (has_bf=true, bf_d=3.0)
  bool has_bf = true;
  bool has_bf_explicit = false; // YAML 是否显式给出 has_bf
  double bf_d = NAN;
  bool p_break = false; // 是否宇称破缺（如弱衰变）
  std::vector<std::vector<int>> sl_filter; // 允许的 [S, L] 分波（S 为 2S+1）; 空 = 全允许
};

struct DecayChainConfig {
  std::string name;
  std::vector<DecayStep> decay_steps;
  std::vector<ResonanceChainConfig> resonance_chains;
  std::vector<std::string> legend_template;
};


struct ConstraintConfig {
  std::vector<std::string> names;
  std::vector<std::complex<double>> values;
  std::string type;
};

// 统一观测函数（函数名与 LorentzVector/TLorentzVector 方法一致）
enum ObsFunc {
    OBS_M = 0,        // M(...): 任意粒子子集 4-矢量和取 M()
    OBS_M2,           // M2(...)
    OBS_P,            // P(p)
    OBS_E,            // E(p)
    OBS_PERP,         // Perp(p) = Pt(p)
    OBS_PT,           // Pt(p)
    OBS_THETA,        // Theta(p): boost 系内极角
    OBS_PHI,          // Phi(p): boost 系内方位角
    OBS_COSTHETA,     // CosTheta(p)
    OBS_ANGLE,        // Angle(a): 与母粒子夹角（boost 系内）
    OBS_COSANGLE,     // CosAngle(a): 与母粒子夹角余弦
};

struct ObsSpec {
    int func = OBS_M;
    std::vector<std::string> args;   // 主体: 粒子名（求和为系统 4-矢量）
    std::vector<std::string> boost;  // 帧: boost 子系粒子名（空 = 顶层母粒子静系）
    std::vector<std::string> axis;   // 轴: 角度类函数的参考方向粒子（空 = 顶层母粒子）
};

struct PlotConfig {
  std::vector<std::vector<std::string>> particles;
  std::vector<int> bins;
  std::vector<std::vector<double>> ranges;
  std::vector<std::string> display;
  std::string type; // "mass", "cosbeta", "dalitz", "obs"(统一 expr 形式)
  std::string name;
  std::vector<ObsSpec> obs;          // 1 个 = 1d, 2 个 = 2d
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
  bool getHasBf() const { return global_has_bf_; }
  // 用户请求的精度: "auto" | "float" | "double"（与 .so 编译精度比对用）
  const std::string &getPrecision() const { return precision_; }
  const std::vector<PlotConfig> &getPlotConfigs() const {
    return plot_configs_;
  }

  // ---------- 命名变量约束 (Constraints: fix_var/free_var/var_range/var_equal/gauss_constr) ----------
  // 仅作用于 theta 参数（共振态 mass/width/r/g），按 "resName_paramName" 名字匹配
  const std::map<std::string, double> &getFixVar() const { return fix_var_; }
  const std::set<std::string> &getFreeVar() const { return free_var_; }
  const std::map<std::string, std::pair<double, double>> &getVarRange() const {
    return var_range_;
  }
  const std::vector<std::vector<std::string>> &getVarEqual() const {
    return var_equal_;
  }
  const std::map<std::string, double> &getGaussConstr() const {
    return gauss_constr_;
  }
  // Constraints.free_phsp_amplitudes: true = phsp 振幅不驻留（构造期分块流式
  // 建 phsp 矩阵、拟合期零 phsp 驻留、writeResult 按批重算）；false = 现状（驻留）。
  // 仅当所有共振态质量/宽度固定（无 free 参数）时生效，否则警告并忽略。
  bool getFreePhspAmplitudes() const { return free_phsp_amplitudes_; }

  std::vector<std::string> getLegends() const;
  std::string generateLegend(const std::vector<std::string> &particles) const;
  // 返回全同粒子分组: group_name -> {particle_name, ...}
  std::map<std::string, std::vector<std::string>> getIdenticalGroups() const;
  // 完整链串生成（与 chains_exact 过滤器同一实现; 用户抄串用）
  //   getExactChainStrings(dc): 该级联全部共振态组合的 "[a->b+c, ...]" 串
  //   getExactChainStrings(chain_idx): 指定展开链(decay_chains_[idx]); <0 返回空
  //   containing 非空时只返回包含该子串的串
  std::vector<std::string> getExactChainStrings(
      const DecayChainConfig &dc, const std::string &containing = "") const;
  std::vector<std::string> getExactChainStrings(
      int chain_idx, const std::string &containing = "") const;

private:
  // 解析函数
  void parseParticles(const YAML::Node &node);
  void parseData(const YAML::Node &node);
  void parseDecayChains(const YAML::Node &node);
  void parseResonances(const YAML::Node &node);
  void parseConstraints(const YAML::Node &node);
  void parsePlotConfig(const YAML::Node &node);
  // 势垒因子三级作用域决议: per-step > ResonanceConfig(母粒子) > Constraints 全局 > 默认
  void resolveStepBF();

  std::vector<Particle> particles_;
  std::vector<DecayChainConfig> decay_chains_;
  std::string config_file_;                    // 配置文件路径（供解析相对路径）
  std::vector<std::string> chain_filter_;      // Constraints.chains 子串过滤; 空 = 全跑
  std::vector<std::string> chain_exact_filter_; // Constraints.chains_exact: TFPWA 式精确整链串过滤; 空 = 不启用
  std::map<std::string, ResonanceConfig> resonances_;
  std::map<std::string, std::vector<std::string>> data_files_;
  std::vector<std::string> data_order_;
  std::vector<ConstraintConfig> constraints_;
  std::vector<PlotConfig> plot_configs_;
  int global_maxL_ = -1; // -1 = no limit; set via Constraints.maxL
  double global_bf_d_ = 3.0; // barrier factor d; set via Constraints.bf_d
  bool global_has_bf_ = true; // 全局势垒开关; set via Constraints.has_bf
  std::string precision_ = "auto"; // 用户请求精度: "auto" | "float" | "double"（auto=跟随 .so 编译精度）

  // ---- 命名变量约束（Constraints 下）----
  std::map<std::string, double> fix_var_;       // name → 固定值
  std::set<std::string> free_var_;              // 取消 fix_var 的名字
  std::map<std::string, std::pair<double, double>> var_range_; // name → [lower, upper]
  std::vector<std::vector<std::string>> var_equal_;  // [[n1, n2, ...], ...] 共享参数
  std::map<std::string, double> gauss_constr_;  // name → sigma
  bool free_phsp_amplitudes_ = false;  // Constraints.free_phsp_amplitudes
};

#endif // CONFIG_CUH