
#include <Resonance.cuh>

// Resonance 类实现
Resonance::Resonance(const std::string& name, const std::string& tag, int J,
    int P, const std::string& modelTypeStr,
    const std::vector<double>& params,
    const std::vector<std::pair<double, double>>& channels)
    : name_(name), tag_(tag), J_(J), P_(P), channels_(channels)
{
    modelType_ = modelTypeFromString(modelTypeStr);
    setParamsByModelType(params);
}

ResModelType Resonance::modelTypeFromString(const std::string& modelStr)
{
    static const std::map<std::string, ResModelType> modelMap = {
        {"BWR", ResModelType::BWR},
        {"BW", ResModelType::BW},
        {"ONE", ResModelType::ONE},
        {"Flatte", ResModelType::Flatte} };

    auto it = modelMap.find(modelStr);
    if (it != modelMap.end()) {
        return it->second;
    }
    throw std::runtime_error("Unknown model type: " + modelStr);
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

void Resonance::setParamsByModelType(const std::vector<double>& params)
{
    switch (modelType_) {
    case ResModelType::BWR:
        if (params.size() < 2) {
            throw std::runtime_error(
                "BWR model requires at least mass and width parameters");
        }
        params_ = { {"mass", params[0]}, {"width", params[1]} };
        if (params.size() > 2) {
            params_["r"] = params[2];
        }
        break;

    case ResModelType::BW:
        if (params.size() < 2) {
            throw std::runtime_error(
                "BW model requires at least mass and width parameters");
        }
        params_ = { {"mass", params[0]}, {"width", params[1]} };
        break;

    case ResModelType::ONE:
        if (params.size() < 1) {
            throw std::runtime_error(
                "One parameter model requires mass parameter");
        }
        params_ = { {"mass", params[0]} };
        break;

    case ResModelType::Flatte: {
        if (params.size() < 2) {
            throw std::runtime_error(
                "Flatte model requires at least mass and one coupling parameter");
        }
        params_ = { {"mass", params[0]} };
        for (size_t i = 1; i < params.size(); ++i) {
            params_["g" + std::to_string(i)] = params[i];
        }
        break;
    }
    }
}

std::vector<std::string> Resonance::paramNamesForType(ResModelType type)
{
    switch (type) {
    case ResModelType::BWR:   return {"mass", "width", "r"};
    case ResModelType::BW:    return {"mass", "width"};
    case ResModelType::ONE:   return {"mass"};
    case ResModelType::Flatte: return {}; // Flatte 参数名动态生成，不使用此函数
    default: return {};
    }
}

int Resonance::paramIndexForType(ResModelType type, const std::string& paramName)
{
    auto names = paramNamesForType(type);
    for (size_t i = 0; i < names.size(); ++i) {
        if (names[i] == paramName) return static_cast<int>(i);
    }
    // Flatte: g1 → index 1, g2 → index 2, ...
    if (type == ResModelType::Flatte) {
        if (paramName == "mass") return 0;
        if (paramName.size() > 1 && paramName[0] == 'g') {
            int n = std::stoi(paramName.substr(1));
            return n; // g1 → 1, g2 → 2, ...
        }
    }
    return -1;
}

std::vector<double> Resonance::getOrderedParams() const
{
    // Flatte 自由参数: [mass, g1, g2, ...] （不含 channel masses）
    if (modelType_ == ResModelType::Flatte) {
        std::vector<double> result;
        auto it = params_.find("mass");
        if (it != params_.end()) result.push_back(it->second);
        for (size_t i = 1; ; ++i) {
            auto gi = params_.find("g" + std::to_string(i));
            if (gi == params_.end()) break;
            result.push_back(gi->second);
        }
        return result;
    }

    auto names = paramNamesForType(modelType_);
    std::vector<double> result;
    for (const auto& name : names) {
        auto it = params_.find(name);
        if (it != params_.end()) {
            result.push_back(it->second);
        } else {
            result.push_back(0.0);
        }
    }
    return result;
}
