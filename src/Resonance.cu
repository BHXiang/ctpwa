#include <Resonance.cuh>
#include <CustomExpr.cuh>
#include <SymbolicDiff.cuh>  // buildModelAST
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
    // 符号微分 aux 在 addBlock 构建（需要 decay 结构的 q0 依赖信息）
    std::vector<double> buildAuxData(const std::map<std::string,std::string>&,
        const std::vector<std::pair<double,double>>&) const override { return {}; }
};

class BWModel : public ResonanceModel
{
  public:
    std::string name() const override { return "BW"; }
    ResModelType type() const override { return ResModelType::BW; }
    std::vector<ParamSpec> paramSpecs() const override {
        return {{"mass", 0.0, false}, {"width", 0.0, false}};
    }
    std::vector<double> buildAuxData(const std::map<std::string,std::string>&,
        const std::vector<std::pair<double,double>>&) const override { return {}; }
};

class ONEModel : public ResonanceModel
{
  public:
    std::string name() const override { return "ONE"; }
    ResModelType type() const override { return ResModelType::ONE; }
    std::vector<ParamSpec> paramSpecs() const override {
        return {{"mass", 0.0, false}};
    }
    std::vector<double> buildAuxData(const std::map<std::string,std::string>&,
        const std::vector<std::pair<double,double>>&) const override { return {}; }
};

class FlatteModel : public ResonanceModel
{
  public:
    std::string name() const override { return "Flatte"; }
    ResModelType type() const override { return ResModelType::Flatte; }
    std::vector<ParamSpec> paramSpecs() const override {
        return {{"mass", 0.0, false}};
    }
    std::string extraPrefix() const override { return "g"; }
    std::vector<double> buildAuxData(const std::map<std::string,std::string>&,
        const std::vector<std::pair<double,double>>&) const override { return {}; }
};

class GSModel : public ResonanceModel
{
  public:
    std::string name() const override { return "GS"; }
    ResModelType type() const override { return ResModelType::GS; }
    std::vector<ParamSpec> paramSpecs() const override {
        return {{"mass", 0.0, false}, {"width", 0.0, false}};
    }
    std::vector<double> buildAuxData(const std::map<std::string,std::string>&,
        const std::vector<std::pair<double,double>>&) const override { return {}; }
};

// Interp: 统一插值模型（hist/linear/spline）。无固定参数。
// aux = [method, N, x_min, dx, y_0...y_{N-1}] (等距 bin)
class InterpModel : public ResonanceModel
{
  public:
    std::string name() const override { return "Interp"; }
    ResModelType type() const override { return ResModelType::Interp; }
    std::vector<ParamSpec> paramSpecs() const override { return {}; }
    std::vector<double> buildAuxData(const std::map<std::string,std::string>& opts,
        const std::vector<std::pair<double,double>>&) const override
    {
        // method: hist=0, linear=1, spline=2
        int method = 1;  // default: linear
        auto it = opts.find("method");
        if (it != opts.end()) {
            if (it->second == "hist") method = 0;
            else if (it->second == "spline") method = 2;
        }
        // 读取数据文件
        it = opts.find("file");
        if (it == opts.end()) return {};  // no file → empty
        std::ifstream f(it->second);
        if (!f) return {};
        std::vector<double> x, y;
        double a, b;
        while (f >> a >> b) { x.push_back(a); y.push_back(b); }
        if (x.size() < 2) return {};
        // 检查是否等距
        double dx = x[1] - x[0];
        bool uniform = true;
        for (size_t i = 1; i < x.size(); ++i) {
            if (fabs(x[i] - x[i-1] - dx) > 1e-10) { uniform = false; break; }
        }
        std::vector<double> aux;
        if (uniform) {
            aux.push_back((double)method);   // method
            aux.push_back((double)x.size()); // N
            aux.push_back(x[0]);             // x_min
            aux.push_back(dx);               // dx
            aux.insert(aux.end(), y.begin(), y.end());
        } else {
            aux.push_back((double)-(method+1)); // negative = non-uniform
            aux.push_back((double)x.size());
            aux.insert(aux.end(), x.begin(), x.end());
            aux.insert(aux.end(), y.begin(), y.end());
        }
        return aux;
    }
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
        const std::map<std::string, std::string>& options,
        const std::vector<std::pair<double, double>>&) const override
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

// Custom：用户自定义表达式模型（DSL）
// options: expr(表达式), params(逗号分隔的参数名列表)
// aux 数据 = 编译后的字节码段（compileCustomExpr 输出）
class CustomModel : public ResonanceModel
{
  public:
    std::string name() const override { return "Custom"; }
    ResModelType type() const override { return ResModelType::Custom; }

    std::vector<ParamSpec> paramSpecs() const override {
        // 参数名在构造时由 options["params"] 决定；这里返回空（setParamsFromModel 特殊处理）
        return {};
    }

    // 参数名列表（由 options["params"] 解析）
    static std::vector<std::string> paramNamesFromOptions(
        const std::map<std::string, std::string>& options)
    {
        std::vector<std::string> names;
        auto it = options.find("params");
        if (it != options.end()) {
            std::string s = it->second;
            size_t pos = 0;
            while (pos < s.size()) {
                size_t comma = s.find(',', pos);
                std::string name = (comma == std::string::npos)
                    ? s.substr(pos) : s.substr(pos, comma - pos);
                // 去空白
                size_t b = name.find_first_not_of(" \t");
                size_t e = name.find_last_not_of(" \t");
                if (b != std::string::npos)
                    names.push_back(name.substr(b, e - b + 1));
                if (comma == std::string::npos) break;
                pos = comma + 1;
            }
        }
        return names;
    }

    std::vector<double> buildAuxData(
        const std::map<std::string, std::string>& options,
        const std::vector<std::pair<double, double>>&) const override
    {
        auto expr_it = options.find("expr");
        if (expr_it == options.end())
            throw std::runtime_error("Custom model requires expr option");
        auto names = paramNamesFromOptions(options);
        if (names.size() > 16)
            throw std::runtime_error("Custom model supports at most 16 free params");
        return compileCustomExpr(expr_it->second, names);
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
        m.emplace("GS", std::make_unique<GSModel>());
        m.emplace("Interp", std::make_unique<InterpModel>());
        m.emplace("Hist", std::make_unique<HistModel>());
        m.emplace("Custom", std::make_unique<CustomModel>());
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
    aux_data_ = model_->buildAuxData(options_, channels_);
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

    // Custom 模型: 参数名来自 options["params"]（paramSpecs 为空）
    std::vector<ParamSpec> specs;
    if (modelType_ == ResModelType::Custom) {
        for (const auto& n : CustomModel::paramNamesFromOptions(options_))
            specs.push_back({n, 0.0, false});
    } else {
        specs = model_->paramSpecs();
    }

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
