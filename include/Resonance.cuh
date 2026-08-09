#ifndef RESONANCE_CUH
#define RESONANCE_CUH

#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include "ComplexType.h"
#include <thrust/complex.h>

// 共振模型类型枚举
enum class ResModelType : int
{
    BWR = 0,
    BW = 1,
    ONE = 2,
    Flatte = 3,
    Hist = 4,     // 直方图形状模型（查表，无自由参数）
    Custom = 5    // 用户自定义表达式模型（DSL 字节码解释）
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
    int aux_offset = 0;      // 模型专属辅助数据在 d_all_aux 中的偏移（Hist 形状表 / Custom 字节码；未用为 0）
    int aux_size = 0;        // 辅助数据长度（未用为 0）
};

// ============================================================================
// 模型注册表（Model Registry）— 类驱动
// 每个模型一个 ResonanceModel 子类，负责参数元数据 + 模型专属逻辑
// （aux 数据构建等）。新增模型 = 写一个类 + registerModel。
// ============================================================================

// 单个参数描述
struct ParamSpec
{
    std::string name;    // 参数名（进入 d_all_params / θ 参数名）
    double init;         // 默认值（spec 用；实际值来自 config 的 parameters 列表）
    bool optional = false; // true 时允许 config 不提供（如 BWR 的 r）
};

// 模型接口：每个模型一个实现类
class ResonanceModel
{
  public:
    virtual ~ResonanceModel() = default;

    virtual std::string name() const = 0;                 // 配置字符串（"BWR"/"Hist"）
    virtual ResModelType type() const = 0;                // 设备分派枚举
    virtual std::vector<ParamSpec> paramSpecs() const = 0; // 固定参数（按此顺序进 d_all_params）
    virtual std::string extraPrefix() const { return ""; } // 附加参数前缀（Flatte: "g" → g1,g2,...）

    // 构建模型专属辅助数据（Hist: 读直方图文件 → [m_min,m_max,n_bins,values...]；非 aux 模型返回空）
    virtual std::vector<double> buildAuxData(
        const std::map<std::string, std::string>& options) const { return {}; }
};

// 注册表：按名字/类型查询模型；模型实例为注册表所有（单例），调用方只持有指针
class ModelRegistry
{
  public:
    // 按名字查询（大小写不敏感；找不到返回 nullptr）
    static const ResonanceModel* find(const std::string& name);
    // 按类型查询（必须存在）
    static const ResonanceModel& get(ResModelType type);
    // 运行时注册（Hist/Custom 等动态模型使用）；覆盖同名条目
    static void registerModel(std::unique_ptr<ResonanceModel> model);
    // 全部已注册模型
    static const std::map<std::string, std::unique_ptr<ResonanceModel>>& all();

  private:
    static std::map<std::string, std::unique_ptr<ResonanceModel>>& table();
    static std::map<ResModelType, const ResonanceModel*>& typeMap();
};

// 共振态类
class Resonance
{
  public:
    Resonance(const std::string& name, const std::string& tag, int J, int P,
              const std::string& modelTypeStr,
              const std::vector<double>& params,
              const std::vector<std::pair<double, double>>& channels = {},
              const std::map<std::string, std::string>& options = {});

    static ResModelType modelTypeFromString(const std::string& modelStr);
    double getParam(const std::string& paramName);
    const std::map<std::string, double>& getParams() const { return params_; }
    const std::vector<std::string>& getOrderedParamNames() const { return param_names_; }

    std::string getName() const { return name_; }
    std::string getTag() const { return tag_; }
    int getJ() const { return J_; }
    int getP() const { return P_; }
    ResModelType getModelType() const { return modelType_; }
    const ResonanceModel& getModel() const { return *model_; }

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
    // 返回模型辅助数据（Hist: [m_min, m_max, n_bins, values...]；非 aux 模型为空）
    // 与 channels 一起进入 d_all_channels 辅助段，aux_offset 指向段内偏移
    const std::vector<double>& getAuxData() const { return aux_data_; }
    // 返回规范顺序的参数名列表
    static std::vector<std::string> paramNamesForType(ResModelType type);
    // 参数名 → params[] 下标的映射
    static int paramIndexForType(ResModelType type, const std::string& paramName);

  private:
    void setParamsFromModel(const std::vector<double>& params);

    std::string name_;
    std::string tag_;
    int J_; // 自旋
    int P_; // 宇称
    ResModelType modelType_;
    const ResonanceModel* model_ = nullptr;    // 注册表单例指针（不拥有）
    std::map<std::string, std::string> options_; // 模型选项（Hist: file/bins/range）
    std::string conjugate_partner_;
    std::map<std::string, double> params_;
    std::vector<std::string> param_names_;          // insertion order for iteration
    std::vector<std::pair<double, double>> channels_;
    std::vector<double> aux_data_;                  // 模型辅助数据（Hist 形状表）
};

#endif // RESONANCE_H
