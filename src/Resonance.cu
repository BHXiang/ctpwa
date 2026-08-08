#include <Resonance.cuh>
#include <algorithm>
#include <fstream>
#include <sstream>

// ============================================================================
// 模型类实现（每个模型一个 ResonanceModel 子类）
// ============================================================================

namespace {

class BWRModel : public ResonanceModel
{
  public:
    std::string name() const override { return "BWR"; }
    ResModelType type() const override { return ResModelType::BWR; }
    std::vector<ParamSpec> paramSpecs() const override {
        return {{"mass", 0.0, false}, {"width", 0.0, false}, {"r", 3.0, true}};
    }
};

class BWModel : public ResonanceModel
{
  public:
    std::string name() const override { return "BW"; }
    ResModelType type() const override { return ResModelType::BW; }
    std::vector<ParamSpec> paramSpecs() const override {
        return {{"mass", 0.0, false}, {"width", 0.0, false}};
    }
};

class ONEModel : public ResonanceModel
{
  public:
    std::string name() const override { return "ONE"; }
    ResModelType type() const override { return ResModelType::ONE; }
    std::vector<ParamSpec> paramSpecs() const override {
        return {{"mass", 0.0, false}};
    }
};

class FlatteModel : public ResonanceModel
{
  public:
    std::string name() const override { return "Flatte"; }
    ResModelType type() const override { return ResModelType::Flatte; }
    std::vector<ParamSpec> paramSpecs() const override {
        return {{"mass", 0.0, false}};
    }
    std::string extraPrefix() const override { return "g"; }  // g1, g2, ...
};

// Hist：直方图形状模型。无固定参数（可选 mass 用于 q0 归一化）。
// aux 数据 = 读直方图文件 → [m_min, m_max, n_bins, values...]
class HistModel : public ResonanceModel
{
  public:
    std::string name() const override { return "Hist"; }
    ResModelType type() const override { return ResModelType::Hist; }
    std::vector<ParamSpec> paramSpecs() const override {
        return {{"mass", 1.0, true}};   // 可选质量（用于 q0 归一化）
    }
    std::vector<double> buildAuxData(
        const std::map<std::string, std::string>& options) const override
    {
        auto opt = [&](const std::string& k) -> std::string {
            auto it = options.find(k);
            return (it != options.end()) ? it->second : std::string();
        };

        std::string file = opt("file");
        std::string bins_str = opt("bins");
        std::string range_str = opt("range");
        if (file.empty() || bins_str.empty() || range_str.empty()) {
            throw std::runtime_error("Hist model requires file, bins and range options");
        }
        int n_bins = std::stoi(bins_str);
        if (n_bins < 2)
            throw std::runtime_error("Hist model needs bins >= 2");

        double m_min = 0.0, m_max = 0.0;
        {
            std::stringstream ss(range_str);
            std::string tok;
            std::vector<double> r;
            while (std::getline(ss, tok, ',')) {
                try { r.push_back(std::stod(tok)); } catch (...) {}
            }
            if (r.size() != 2 || r[1] <= r[0])
                throw std::runtime_error("Hist model needs range lo,hi with hi > lo");
            m_min = r[0]; m_max = r[1];
        }

        std::ifstream f(file);
        if (!f.is_open())
            throw std::runtime_error("Hist model: cannot open file " + file);
        std::vector<double> values;
        double v;
        while (f >> v) values.push_back(v);
        if ((int)values.size() != n_bins)
            throw std::runtime_error("Hist model: file " + file + " has " +
                std::to_string(values.size()) + " values, expected " +
                std::to_string(n_bins));

        std::vector<double> aux;
        aux.push_back(m_min);
        aux.push_back(m_max);
        aux.push_back((double)n_bins);
        aux.insert(aux.end(), values.begin(), values.end());
        return aux;
    }
};

}  // namespace

// ============================================================================
// ModelRegistry 实现
// ============================================================================

std::map<std::string, std::unique_ptr<ResonanceModel>>& ModelRegistry::table()
{
    // initializer_list 会拷贝元素（unique_ptr 不可拷贝）→ 用 lambda 构造
    static auto t = [] {
        std::map<std::string, std::unique_ptr<ResonanceModel>> m;
        m.emplace("BWR", std::make_unique<BWRModel>());
        m.emplace("BW", std::make_unique<BWModel>());
        m.emplace("ONE", std::make_unique<ONEModel>());
        m.emplace("Flatte", std::make_unique<FlatteModel>());
        m.emplace("Hist", std::make_unique<HistModel>());
        return m;
    }();
    return t;
}

std::map<ResModelType, const ResonanceModel*>& ModelRegistry::typeMap()
{
    static std::map<ResModelType, const ResonanceModel*> m;
    static bool initialized = false;
    if (!initialized) {
        for (const auto& [name, model] : table())
            m[model->type()] = model.get();
        initialized = true;
    }
    return m;
}

const ResonanceModel* ModelRegistry::find(const std::string& name)
{
    // 大小写不敏感查找（"BWR"/"bwr"/"Hist"/"hist" 均有效）
    auto& t = table();
    auto it = t.find(name);
    if (it != t.end()) return it->second.get();
    std::string lower = name;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    for (const auto& [k, model] : t) {
        std::string kl = k;
        std::transform(kl.begin(), kl.end(), kl.begin(), ::tolower);
        if (kl == lower) return model.get();
    }
    return nullptr;
}

const ResonanceModel& ModelRegistry::get(ResModelType type)
{
    auto& m = typeMap();
    auto it = m.find(type);
    if (it == m.end())
        throw std::runtime_error("ModelRegistry: unknown model type " +
                                 std::to_string(static_cast<int>(type)));
    return *it->second;
}

void ModelRegistry::registerModel(std::unique_ptr<ResonanceModel> model)
{
    if (model == nullptr) return;
    table()[model->name()] = std::move(model);
    // 新增类型时同步反查表
    typeMap();
}

const std::map<std::string, std::unique_ptr<ResonanceModel>>& ModelRegistry::all()
{
    return table();
}

// ============================================================================
// Resonance 类实现
// ============================================================================

Resonance::Resonance(const std::string& name, const std::string& tag, int J,
    int P, const std::string& modelTypeStr,
    const std::vector<double>& params,
    const std::vector<std::pair<double, double>>& channels,
    const std::map<std::string, std::string>& options)
    : name_(name), tag_(tag), J_(J), P_(P), channels_(channels), options_(options)
{
    model_ = ModelRegistry::find(modelTypeStr);
    if (model_ == nullptr)
        throw std::runtime_error("Unknown model type: " + modelTypeStr);
    modelType_ = model_->type();
    setParamsFromModel(params);
    // 模型专属辅助数据（Hist 形状表；非 aux 模型为空）
    aux_data_ = model_->buildAuxData(options_);
}

ResModelType Resonance::modelTypeFromString(const std::string& modelStr)
{
    const ResonanceModel* model = ModelRegistry::find(modelStr);
    if (model == nullptr)
        throw std::runtime_error("Unknown model type: " + modelStr);
    return model->type();
}

double Resonance::getParam(const std::string& paramName)
{
    auto it = params_.find(paramName);
    if (it == params_.end()) {
        throw std::runtime_error("Parameter " + paramName +
            " not found for resonance " + name_);
    }
    return it->second;
}

// 从模型参数定义构建参数表：paramSpecs 的固定参数 + extraPrefix 的附加参数
// （Flatte 的 g1,g2,...）。顺序 = param_names_ 插入序 = d_all_params 布局。
void Resonance::setParamsFromModel(const std::vector<double>& params)
{
    params_.clear();
    param_names_.clear();

    const auto& specs = model_->paramSpecs();

    // 必需参数数量校验
    size_t required = 0;
    for (const auto& ps : specs)
        if (!ps.optional) ++required;
    if (params.size() < required) {
        throw std::runtime_error(
            "Model '" + model_->name() + "' requires at least " +
            std::to_string(required) + " parameter(s), got " +
            std::to_string(params.size()));
    }

    // 固定参数
    for (size_t i = 0; i < specs.size() && i < params.size(); ++i) {
        params_[specs[i].name] = params[i];
        param_names_.push_back(specs[i].name);
    }

    // 附加参数（Flatte: g1, g2, ...）
    std::string prefix = model_->extraPrefix();
    if (!prefix.empty()) {
        for (size_t i = specs.size(); i < params.size(); ++i) {
            std::string gname = prefix + std::to_string(i);
            params_[gname] = params[i];
            param_names_.push_back(gname);
        }
    }
}

std::vector<std::string> Resonance::paramNamesForType(ResModelType type)
{
    const auto& model = ModelRegistry::get(type);
    std::vector<std::string> names;
    for (const auto& ps : model.paramSpecs()) names.push_back(ps.name);
    return names;
}

int Resonance::paramIndexForType(ResModelType type, const std::string& paramName)
{
    const auto& model = ModelRegistry::get(type);
    const auto& specs = model.paramSpecs();
    for (size_t i = 0; i < specs.size(); ++i)
        if (specs[i].name == paramName) return static_cast<int>(i);
    // 附加参数: prefix + 序号 → 下标 = 固定参数数 + (序号 - prefix 长度起点)
    // 例如 Flatte: g1 → index 1, g2 → index 2, ...
    std::string prefix = model.extraPrefix();
    if (!prefix.empty() && paramName.size() > prefix.size()) {
        std::string num = paramName.substr(prefix.size());
        bool all_digits = !num.empty() &&
            std::all_of(num.begin(), num.end(), ::isdigit);
        if (all_digits) {
            int n = std::stoi(num);
            return static_cast<int>(specs.size()) + (n - 1);
        }
    }
    return -1;
}

std::vector<double> Resonance::getOrderedParams() const
{
    std::vector<double> result;
    for (const auto& name : param_names_) {
        auto it = params_.find(name);
        result.push_back(it != params_.end() ? it->second : 0.0);
    }
    return result;
}
