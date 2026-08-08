
#include <Resonance.cuh>
#include <algorithm>
#include <fstream>
#include <sstream>

// ============================================================================
// ModelRegistry 实现
// ============================================================================

std::map<std::string, ModelSpec>& ModelRegistry::table()
{
    // 内置模型注册（一次初始化）
    static std::map<std::string, ModelSpec> t = {
        {"BWR", {ResModelType::BWR, "BWR",
                 {{"mass", 0.0, false}, {"width", 0.0, false}, {"r", 3.0, true}},
                 ""}},
        {"BW", {ResModelType::BW, "BW",
                {{"mass", 0.0, false}, {"width", 0.0, false}},
                ""}},
        {"ONE", {ResModelType::ONE, "ONE",
                 {{"mass", 0.0, false}},
                 ""}},
        {"Flatte", {ResModelType::Flatte, "Flatte",
                    {{"mass", 0.0, false}},
                    "g"}},
        {"Hist", {ResModelType::Hist, "Hist",
                  {},   // 无固定参数（形状由直方图文件决定）
                  ""}},
    };
    return t;
}

std::map<ResModelType, std::string>& ModelRegistry::typeNameMap()
{
    static std::map<ResModelType, std::string> m;
    // 惰性初始化：从注册表反查（保证与 table() 一致）
    static bool initialized = false;
    if (!initialized) {
        for (const auto& [name, spec] : table())
            m[spec.type] = name;
        initialized = true;
    }
    return m;
}

const ModelSpec* ModelRegistry::find(const std::string& name)
{
    // 大小写不敏感查找（"BWR"/"bwr"/"Hist"/"hist" 均有效）
    auto& t = table();
    auto it = t.find(name);
    if (it != t.end()) return &it->second;
    std::string lower = name;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    for (const auto& [k, spec] : t) {
        std::string kl = k;
        std::transform(kl.begin(), kl.end(), kl.begin(), ::tolower);
        if (kl == lower) return &spec;
    }
    return nullptr;
}

const ModelSpec& ModelRegistry::get(ResModelType type)
{
    auto& m = typeNameMap();
    auto it = m.find(type);
    if (it == m.end())
        throw std::runtime_error("ModelRegistry: unknown model type " +
                                 std::to_string(static_cast<int>(type)));
    return get(it->second);
}

const ModelSpec& ModelRegistry::get(const std::string& name)
{
    const ModelSpec* spec = find(name);
    if (spec == nullptr)
        throw std::runtime_error("ModelRegistry: unknown model '" + name + "'");
    return *spec;
}

void ModelRegistry::registerModel(const ModelSpec& spec)
{
    table()[spec.name] = spec;
    // 新增类型时同步反查表（动态模型可复用现有枚举或扩展）
    typeNameMap()[spec.type] = spec.name;
}

const std::map<std::string, ModelSpec>& ModelRegistry::all()
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
    : name_(name), tag_(tag), J_(J), P_(P), channels_(channels)
{
    modelType_ = modelTypeFromString(modelTypeStr);
    spec_ = ModelRegistry::get(modelType_);
    // 合并模型选项（Hist: file/bins/range/extrapolate）
    for (const auto& [k, v] : options) spec_.options[k] = v;
    setParamsFromSpec(spec_, params);
    // Hist: 加载直方图 → 形状表 [m_min, m_max, n_bins, values...]
    if (modelType_ == ResModelType::Hist)
        loadHistAuxData();
}

ResModelType Resonance::modelTypeFromString(const std::string& modelStr)
{
    const ModelSpec* spec = ModelRegistry::find(modelStr);
    if (spec == nullptr)
        throw std::runtime_error("Unknown model type: " + modelStr);
    return spec->type;
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

// 从 ModelSpec 构建参数表：spec.params 的固定参数 + extra_prefix 的附加参数
// （Flatte 的 g1,g2,...）。顺序 = param_names_ 插入序 = d_all_params 布局。
void Resonance::setParamsFromSpec(const ModelSpec& spec, const std::vector<double>& params)
{
    params_.clear();
    param_names_.clear();

    // 必需参数数量校验
    size_t required = 0;
    for (const auto& ps : spec.params)
        if (!ps.optional) ++required;
    if (params.size() < required) {
        throw std::runtime_error(
            "Model '" + spec.name + "' requires at least " +
            std::to_string(required) + " parameter(s), got " +
            std::to_string(params.size()));
    }

    // 固定参数
    for (size_t i = 0; i < spec.params.size() && i < params.size(); ++i) {
        params_[spec.params[i].name] = params[i];
        param_names_.push_back(spec.params[i].name);
    }

    // 附加参数（Flatte: g1, g2, ...）
    if (!spec.extra_prefix.empty()) {
        for (size_t i = spec.params.size(); i < params.size(); ++i) {
            std::string gname = spec.extra_prefix + std::to_string(i);
            params_[gname] = params[i];
            param_names_.push_back(gname);
        }
    }
}

std::vector<std::string> Resonance::paramNamesForType(ResModelType type)
{
    const auto& spec = ModelRegistry::get(type);
    std::vector<std::string> names;
    for (const auto& ps : spec.params) names.push_back(ps.name);
    return names;
}

int Resonance::paramIndexForType(ResModelType type, const std::string& paramName)
{
    const auto& spec = ModelRegistry::get(type);
    for (size_t i = 0; i < spec.params.size(); ++i)
        if (spec.params[i].name == paramName) return static_cast<int>(i);
    // 附加参数: prefix + 序号 → 下标 = 固定参数数 + (序号 - prefix 长度起点)
    // 例如 Flatte: g1 → index 1, g2 → index 2, ...
    if (!spec.extra_prefix.empty() && paramName.size() > spec.extra_prefix.size()) {
        std::string num = paramName.substr(spec.extra_prefix.size());
        bool all_digits = !num.empty() &&
            std::all_of(num.begin(), num.end(), ::isdigit);
        if (all_digits) {
            int n = std::stoi(num);
            return static_cast<int>(spec.params.size()) + (n - 1);
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

// ============================================================================
// Hist 模型：加载直方图文件（dat 文本，每行一个 bin 值）
// options: file(必填), bins(必填), range(必填, "lo,hi"), extrapolate(可选, 默认 true)
// aux_data_ = [m_min, m_max, n_bins, values[0..n_bins-1]]
// ============================================================================
void Resonance::loadHistAuxData()
{
    auto opt = [&](const std::string& k) -> std::string {
        auto it = spec_.options.find(k);
        return (it != spec_.options.end()) ? it->second : std::string();
    };

    std::string file = opt("file");
    std::string bins_str = opt("bins");
    std::string range_str = opt("range");
    if (file.empty() || bins_str.empty() || range_str.empty()) {
        throw std::runtime_error("Hist model '" + name_ +
            "' requires file, bins and range options");
    }
    int n_bins = std::stoi(bins_str);
    if (n_bins < 2) {
        throw std::runtime_error("Hist model '" + name_ + "' needs bins >= 2");
    }
    // range: "lo,hi"
    double m_min = 0.0, m_max = 0.0;
    {
        std::stringstream ss(range_str);
        std::string tok;
        std::vector<double> r;
        while (std::getline(ss, tok, ',')) {
            try { r.push_back(std::stod(tok)); } catch (...) {}
        }
        if (r.size() != 2 || r[1] <= r[0]) {
            throw std::runtime_error("Hist model '" + name_ +
                "' needs range lo,hi with hi > lo");
        }
        m_min = r[0]; m_max = r[1];
    }

    // 读取直方图值
    std::ifstream f(file);
    if (!f.is_open()) {
        throw std::runtime_error("Hist model '" + name_ +
            "': cannot open file " + file);
    }
    std::vector<double> values;
    double v;
    while (f >> v) values.push_back(v);
    if ((int)values.size() != n_bins) {
        throw std::runtime_error("Hist model '" + name_ + "': file " + file +
            " has " + std::to_string(values.size()) + " values, expected " +
            std::to_string(n_bins));
    }

    aux_data_.clear();
    aux_data_.push_back(m_min);
    aux_data_.push_back(m_max);
    aux_data_.push_back((double)n_bins);
    aux_data_.insert(aux_data_.end(), values.begin(), values.end());
}
