#ifndef RESONANCE_CUH
#define RESONANCE_CUH

#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include <cuComplex.h>
#include <thrust/complex.h>

// 共振模型类型枚举
enum class ResModelType : int
{
    BWR = 0,
    BW = 1,
    ONE = 2,
    Flatte = 3
};

// 设备端共振结构体（轻量，固定大小）
struct DeviceResonance
{
    ResModelType type;       // 模型类型
    int particle_idx;        // 粒子索引
    int J;                   // 自旋
    int P;                   // 宇称
    int param_count;         // 自由参数个数（在 d_all_params 中的长度）
    int param_offset;        // 在 d_all_params 中的起始位置
    int n_channels;          // Flatte channel 数（非 Flatte 为 0）
    int channel_offset;      // 在 d_all_channels 中的起始位置
};

// 共振态类
class Resonance
{
  public:
    Resonance(const std::string& name, const std::string& tag, int J, int P,
              const std::string& modelTypeStr,
              const std::vector<double>& params,
              const std::vector<std::pair<double, double>>& channels = {});

    static ResModelType modelTypeFromString(const std::string& modelStr);
    double getParam(const std::string& paramName);
    const std::map<std::string, double>& getParams() const { return params_; }
    const std::vector<std::string>& getOrderedParamNames() const { return param_names_; }

    std::string getName() const { return name_; }
    std::string getTag() const { return tag_; }
    int getJ() const { return J_; }
    int getP() const { return P_; }
    ResModelType getModelType() const { return modelType_; }

    void setConjugatePartner(const std::string& partnerName)
    {
        conjugate_partner_ = partnerName;
    }
    std::string getConjugatePartner() const { return conjugate_partner_; }
    bool hasConjugatePartner() const { return !conjugate_partner_.empty(); }

    // 按规范顺序返回自由参数值（与 d_all_params 中的排列一致）
    std::vector<double> getOrderedParams() const;
    // 返回 channel masses（仅 Flatte 有效）
    const std::vector<std::pair<double, double>>& getChannels() const { return channels_; }
    // 返回规范顺序的参数名列表
    static std::vector<std::string> paramNamesForType(ResModelType type);
    // 参数名 → params[] 下标的映射
    static int paramIndexForType(ResModelType type, const std::string& paramName);

  private:
    void setParamsByModelType(const std::vector<double>& params);

    std::string name_;
    std::string tag_;
    int J_; // 自旋
    int P_; // 宇称
    ResModelType modelType_;
    std::string conjugate_partner_;
    std::map<std::string, double> params_;
    std::vector<std::string> param_names_;          // insertion order for iteration
    std::vector<std::pair<double, double>> channels_;
};

#endif // RESONANCE_H
