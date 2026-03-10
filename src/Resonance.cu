
#include <Resonance.cuh>

// Resonance 类实现
Resonance::Resonance(const std::string &name, const std::string &tag, int J, int P,
                     const std::string &modelTypeStr, const std::vector<double> &params)
    : name_(name), tag_(tag), J_(J), P_(P)
{
    modelType_ = modelTypeFromString(modelTypeStr);
    setParamsByModelType(params);
}

ResModelType Resonance::modelTypeFromString(const std::string &modelStr)
{
    static const std::map<std::string, ResModelType> modelMap = {
        {"BWR", ResModelType::BWR},
        {"ONE", ResModelType::ONE}};

    auto it = modelMap.find(modelStr);
    if (it != modelMap.end())
    {
        return it->second;
    }
    throw std::runtime_error("Unknown model type: " + modelStr);
}

double Resonance::getParam(const std::string &paramName)
{
    auto it = params_.find(paramName);
    if (it == params_.end())
    {
        throw std::runtime_error("Parameter " + paramName + " not found for resonance " + name_);
    }
    return it->second;
}

void Resonance::setParamsByModelType(const std::vector<double> &params)
{
    switch (modelType_)
    {
    case ResModelType::BWR:
        if (params.size() < 2)
        {
            throw std::runtime_error("BWR model requires at least mass and width parameters");
        }
        params_ = {{"mass", params[0]}, {"width", params[1]}};
        if (params.size() > 2)
        {
            params_["r"] = params[2]; // Blatt-Weisskopf半径
        }
        break;

    case ResModelType::ONE:
        if (params.size() < 1)
        {
            throw std::runtime_error("One parameter model requires mass parameter");
        }
        params_ = {{"mass", params[0]}};
        break;
    }
}

// int Resonance::setParamsByModelType(const std::vector<double> &params, const int site, const std::vector<int> &free)
// {
//     switch (modelType_)
//     {
//     case ResModelType::BWR:
//         if (params.size() < 2)
//         {
//             throw std::runtime_error("BWR model requires at least mass and width parameters");
//         }
//         params_ = {{"mass", params[0]}, {"width", params[1]}};
//         if (params.size() > 2)
//         {
//             params_["r"] = params[2]; // Blatt-Weisskopf半径
//         }

//         // 设置自由参数
//         if (free[0] == -1)
//         {
//             free_params_ = {{"mass", site}, {"width", site + 1}};
//             return 2;
//         }
//         else
//         {
//             int num_params = 0;
//             for (const auto &f : free)
//             {
//                 if (f == 0)
//                 {
//                     free_params_["mass"] = site + num_params;
//                     num_params++;
//                 }
//                 else if (f == 1)
//                 {
//                     free_params_["width"] = site + num_params;
//                     num_params++;
//                 }
//                 else if (f == 2)
//                 {
//                     free_params_["r"] = site + num_params;
//                     num_params++;
//                 }
//             }
//             return num_params;
//         }

//         break;

//     case ResModelType::ONE:
//         if (params.size() < 1)
//         {
//             throw std::runtime_error("One parameter model requires mass parameter");
//         }
//         params_ = {{"mass", params[0]}};
//         free_params_ = {{"mass", site}};
//         return 1;
//         break;
//     }
// }

// void Resonance::setFreeParams(const int site, const std::vector<int> &free)
// {
//     if (modelType_ == ResModelType::BWR)
//     {
//         if (free[0] == -1)
//         {
//             free_params_ = {{"mass", site + 1}, {"width", site + 2}};
//             site += 2;
//         }
//         else
//         {
//             for (const auto &f : free)
//             {
//                 if (f == 0)
//                 {
//                     free_params_["mass"] = site + 1;
//                     site += 1;
//                 }
//                 else if (f == 1)
//                 {
//                     free_params_["width"] = site + 2;
//                     site += 1;
//                 }
//                 else if (f == 2)
//                 {
//                     free_params_["r"] = site + 3;
//                     site += 1;
//                 }
//             }
//         }
//     }
//     else if (modelType_ == ResModelType::ONE)
//     {
//         free_params_ = {{"mass", site + 1}};
//         site += 1;
//     }
// }

// void Resonance::UpdateFreeParams(const std::vector<double> &newParams)
// {
//     // 更新自由参数
//     for (const auto &param : free_params_)
//     {
//         if (param.second < newParams.size())
//         {
//             params_[param.first] = newParams[param.second];
//         }
//     }
// }

// 设备端函数实现
__device__ inline double BlattWeisskopf(int L, double q, double q0)
{
    const double d = 3.0;
    if (q0 <= 0)
        return 1.0; // 防止除以零
    const double z = q * d;
    const double z0 = q0 * d;

    switch (L)
    {
    case 0:
        return 1.0;
    case 1:
        return sqrt((1.0 + z0 * z0) / (1.0 + z * z));
    case 2:
        return sqrt((9.0 + 3.0 * z0 * z0 + z0 * z0 * z0 * z0) / (9.0 + 3.0 * z * z + z * z * z * z));
    case 3:
        return sqrt((pow(z0, 6) + 6.0 * pow(z0, 4) + 45.0 * z0 * z0 + 225.0) / (pow(z, 6) + 6.0 * pow(z, 4) + 45.0 * z * z + 225.0));
    case 4:
        return sqrt((pow(z0, 8) + 10.0 * pow(z0, 6) + 135.0 * pow(z0, 4) + 1575.0 * z0 * z0 + 11025.0) /
                    (pow(z, 8) + 10.0 * pow(z, 6) + 135.0 * pow(z, 4) + 1575.0 * z * z + 11025.0));
    case 5:
        return sqrt((pow(z0, 10) + 15.0 * pow(z0, 8) + 315.0 * pow(z0, 6) + 6300.0 * pow(z0, 4) + 99225.0 * z0 * z0 + 893025.0) /
                    (pow(z, 10) + 15.0 * pow(z, 8) + 315.0 * pow(z, 6) + 6300.0 * pow(z, 4) + 99225.0 * z * z + 893025.0));
    case 6:
        return sqrt((pow(z0, 12) + 21.0 * pow(z0, 10) + 630.0 * pow(z0, 8) + 17325.0 * pow(z0, 6) + 363825.0 * pow(z0, 4) + 6185025.0 * z0 * z0 + 540326025.0) /
                    (pow(z, 12) + 21.0 * pow(z, 10) + 630.0 * pow(z, 8) + 17325.0 * pow(z, 6) + 363825.0 * pow(z, 4) + 6185025.0 * z * z + 540326025.0));
    default:
        return 1.0; // 更高角动量返回1.0
    }
}

__device__ thrust::complex<double> BWR(double m, double m0, double gamma0, int L, double q, double q0)
{
    // 计算能量依赖的宽度
    const double gamma = gamma0 * pow(q / q0, 2 * L + 1) * (m0 / m) * pow(BlattWeisskopf(L, q, q0), 2);
    double x = m0 * m0 - m * m;
    double y = m0 * gamma;
    double s = x * x + y * y;

    return thrust::complex<double>(x / s, y / s);
}
