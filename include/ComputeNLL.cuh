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

// 自定义核计算二次型 v^H·M·v，同时输出d_P_vec = M * v
// M: n×n Hermitian矩阵(行主序), v: n维向量(已共轭), n: 维度(<~200)
void computeQuadraticForm(const ctComplex* d_M, const ctComplex* d_v,
    ctComplex* d_P_vec, ctPhspReal* d_phsp_r, ctPhspReal* d_phsp_i, int n);

// 双精度 phsp 归一化因子: Re(v^H · M_double · v)（M_double 为 cuDoubleComplex n×n，
// 由流式分块构建，替代 computePhspMeanSum 的原始振幅扫描）。当前设备须为调用方指定。
double computeDoublePhspSum(const cuDoubleComplex* d_M, const ctComplex* d_v, int n);

#endif
