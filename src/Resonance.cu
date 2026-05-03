
#include <Resonance.cuh>

// Resonance 类实现
Resonance::Resonance(const std::string& name, const std::string& tag, int J,
    int P, const std::string& modelTypeStr,
    const std::vector<double>& params)
    : name_(name), tag_(tag), J_(J), P_(P)
{
    modelType_ = modelTypeFromString(modelTypeStr);
    setParamsByModelType(params);
}

ResModelType Resonance::modelTypeFromString(const std::string& modelStr)
{
    static const std::map<std::string, ResModelType> modelMap = {
        {"BWR", ResModelType::BWR},
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
            params_["r"] = params[2]; // Blatt-Weisskopf半径
        }
        break;

    case ResModelType::ONE:
        if (params.size() < 1) {
            throw std::runtime_error(
                "One parameter model requires mass parameter");
        }
        params_ = { {"mass", params[0]} };
        break;
    case ResModelType::Flatte:
        if (params.size() < 3) {
            throw std::runtime_error(
                "Flatte model requires mass, g_pi, and g_K parameters");
        }
        params_ = {
            {"mass", params[0]}, {"g_pi", params[1]}, {"g_K", params[2]} };
        break;
    }
}

