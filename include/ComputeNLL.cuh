#ifndef COMPUTENLL_CUH
#define COMPUTENLL_CUH

#include "ComplexType.h"
#include <cuda_runtime.h>

// void computeWeightResult(const ctComplex *d_matrix, const ctComplex
// *d_vector, double *d_final_result, double *d_row_results, int M, int N); void
// computeWeightResult(const ctComplex *d_matrix, const ctComplex *d_vector,
// double *d_final_result, double *d_row_results, int nEvents, int ngls, int
// npolar); void computeWeightResult(const ctComplex *d_matrix, const ctComplex
// *d_vector, double *d_final_result, int nEvents, int ngls, int npolar);

void computeNll(const ctComplex *d_matrix, const ctComplex *d_vector,
                const double *d_weights, ctComplex *d_S, double *d_Q,
                double *d_final_result, int nlength, int ngls, int npolar,
                double phsp_factor);
void computePHSPfactor(const ctComplex *d_matrix, const ctComplex *d_vector,
                       ctComplex *d_B, double *d_final_result, int M, int N);

// 从原始振幅矩阵计算NLL值和梯度（使用cuBLAS batch + CUB多级规约）
// d_amp: nEvents × n_polar × n_amplitudes, 每个事件的振幅矩阵(行主序)
// d_vector: n_amplitudes, 拟合参数向量v
// d_P_vec: n_amplitudes, phsp投影向量
// d_grad_out: n_amplitudes, 输出梯度 ∂NLL/∂v
// 返回: NLL = -Σlog(|A_k·v|²) + nEvents·log(phsp_factor)
double computeFactorNLL(const ctComplex* d_amp, const ctComplex* d_vector, ctComplex* d_grad_out, int nEvents, int n_polar, int n_amplitudes, const double* d_weights = nullptr, ctComplex* d_w_out = nullptr);

// 就地共轭复数数组
void conjugateComplexArray(ctComplex* data, int N);

// 向量axpy: y[i] += alpha * x[i], n较小(<~200)，单block完成
void axpyComplex(ctComplex* y, const ctComplex* x, ctComplex alpha, int n);

// 返回 Σ_ev f_ev = Σ |A·v|²（double 累加）——phsp 均值的高精度来源
double computePhspMeanSum(const ctComplex* d_amp, const ctComplex* d_vector,
    int nEvents, int n_polar, int n_amplitudes);

// free-θ 常驻模式: 一次 phsp 扫描同时给出 phsp_sum（语义同 computePhspMeanSum）
// 与 v 梯度 phsp 项 d_P_vec = Σ_ev,p (w_ev/W)·A_ev,p·conj(S_ev,p)（每 GPU 局部和，
// 调用方负责跨 GPU 求和）。数学身份 d_P_vec ≡ B̄·v̄（B̄=Σ(w/W)AA^H），
// 免去 free-θ 下每 forward 的 7.2GB B̄ 加权副本重建。
double computePhspMeanSumAndGradP(const ctComplex* d_amp, const ctComplex* d_vector,
    const double* d_w, int nEvents, int n_polar, int n_amplitudes,
    double inv_W_total, ctComplex* d_P_out);

// 自定义核计算二次型 v^H·M·v，同时输出d_P_vec = M * v
// M: n×n Hermitian矩阵(行主序), v: n维向量(已共轭), n: 维度(<~200)
void computeQuadraticForm(const ctComplex* d_M, const ctComplex* d_v,
    ctComplex* d_P_vec, ctPhspReal* d_phsp_r, ctPhspReal* d_phsp_i, int n);

// 双精度 phsp 归一化因子: Re(v^H · M_double · v)（M_double 为 cuDoubleComplex n×n，
// 由流式分块构建，替代 computePhspMeanSum 的原始振幅扫描）。当前设备须为调用方指定。
double computeDoublePhspSum(const cuDoubleComplex* d_M, const ctComplex* d_v, int n);

// ===========================================================================
// 混合精度（config precision:float，double .so 内）float 变体:
//   A 驻留 float2（8B），耦合 v 恒 double（调用内 downcast 一份 float 副本喂 Sgemv），
//   S/w 缓冲 float2，逐事件 factor 用 double 累加（float2 读入 → double 平方和），
//   -log/跨事件累加恒 double —— 与 V1/V2 nvcc 验证算法逐字一致。
// 纯新增：double 路径不调用，调用点按 float_amps_ 分派。
// ===========================================================================

// 语义同 computeFactorNLL；d_amp 为 float2 布局。d_grad_out / d_w_out 可选。
// d_w_out（float2，仅共振梯度消费方需要）为 null 表示不输出。
double computeFactorNLLF(const float2* d_amp, const ctComplex* d_vector,
    ctComplex* d_grad_out, int nEvents, int n_polar, int n_amplitudes,
    const double* d_weights = nullptr, float2* d_w_out = nullptr);

// 语义同 computePhspMeanSum（Σ|A·v|² double 累加）；d_amp 为 float2 布局。
double computePhspMeanSumF(const float2* d_amp, const ctComplex* d_vector,
    int nEvents, int n_polar, int n_amplitudes);

// 语义同 computePhspMeanSumAndGradP；d_amp 为 float2 布局。
// d_P_out 仍为 double（ctComplex）输出——float matvec 局部和 → double 累加。
double computePhspMeanSumAndGradPF(const float2* d_amp, const ctComplex* d_vector,
    const double* d_w, int nEvents, int n_polar, int n_amplitudes,
    double inv_W_total, ctComplex* d_P_out);

#endif
