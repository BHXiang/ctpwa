#ifndef RESONANCE_CUH
#define RESONANCE_CUH

// #include <vector>
#include <map>
#include <set>
#include <string>
// #include <AmpGen.cuh>
#include <cuComplex.h>
// #include <stdexcept>
// #include <iostream>
// #include <thrust/device_vector.h>
// #include <thrust/host_vector.h>
#include <thrust/complex.h>
// #include <fstream>
// #include <array>

// 共振模型类型枚举
enum class ResModelType : int
{
    BWR = 0, // Breit-Wigner
    ONE = 1,
    FLATTE = 2
};

// 设备端共振结构体
struct DeviceResonance
{
    ResModelType type; // 模型类型
    int particle_idx;  // 粒子索引
    int J;             // 自旋
    int P;             // 宇称
    int param_count;   // 参数个数
    double params[8];  // 参数数组（固定大小）
};

// 共振态类
class Resonance
{
public:
    Resonance(const std::string &name, const std::string &tag, int J, int P,
              const std::string &modelTypeStr, const std::vector<double> &params);

    static ResModelType modelTypeFromString(const std::string &modelStr);
    double getParam(const std::string &paramName);
    const std::map<std::string, double> &getParams() const { return params_; }

    std::string getName() const { return name_; }
    std::string getTag() const { return tag_; }
    int getJ() const { return J_; }
    int getP() const { return P_; }
    ResModelType getModelType() const { return modelType_; }

    void setConjugatePartner(const std::string &partnerName) { conjugate_partner_ = partnerName; }
    std::string getConjugatePartner() const { return conjugate_partner_; }
    bool hasConjugatePartner() const { return !conjugate_partner_.empty(); }

    void setFreeParams(const std::vector<double> &freeParams);
    const std::map<std::string, double> &getFreeParams() const { return free_params_; }

private:
    void setParamsByModelType(const std::vector<double> &params);
    void UpdateFreeParams(const std::vector<double> &newParams);

    std::string name_;
    std::string tag_;
    int J_; // 自旋
    int P_; // 宇称
    ResModelType modelType_;
    std::string conjugate_partner_;
    std::map<std::string, double> params_;
    std::map<std::string, double> free_params_;
};

// 设备端函数声明
__device__ double BlattWeisskopf(int L, double q, double q0);

__device__ thrust::complex<double> BreitWigner(double m, double m0, double gamma0, int L, double q, double q0);

// __global__ void computeAmpsKernel(cuComplex *amplitudes, const DeviceMomenta *d_momenta, const SL *slCombinations, const thrust::complex<double> *slamps, const DeviceResonance resonance, const DecayNode *decayChain, int decayChain_size, int nEvents, int nSLComb, int nPolar)

#endif // RESONANCE_H