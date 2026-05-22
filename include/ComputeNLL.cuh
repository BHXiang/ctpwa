#ifndef COMPUTENLL_CUH
#define COMPUTENLL_CUH

#include <cuComplex.h>
#include <cuda_runtime.h>

// void computeWeightResult(const cuComplex *d_matrix, const cuComplex
// *d_vector, double *d_final_result, double *d_row_results, int M, int N); void
// computeWeightResult(const cuComplex *d_matrix, const cuComplex *d_vector,
// double *d_final_result, double *d_row_results, int nEvents, int ngls, int
// npolar); void computeWeightResult(const cuComplex *d_matrix, const cuComplex
// *d_vector, double *d_final_result, int nEvents, int ngls, int npolar);

void computeNll(const cuComplex *d_matrix, const cuComplex *d_vector,
                const double *d_weights, cuComplex *d_S, double *d_Q,
                double *d_final_result, int nlength, int ngls, int npolar,
                double phsp_factor);
void computePHSPfactor(const cuComplex *d_matrix, const cuComplex *d_vector,
                       cuComplex *d_B, double *d_final_result, int M, int N);

// 从原始振幅矩阵计算NLL值和梯度（使用cuBLAS batch + CUB多级规约）
// d_amp: nEvents × n_polar × n_amplitudes, 每个事件的振幅矩阵(行主序)
// d_vector: n_amplitudes, 拟合参数向量v
// d_P_vec: n_amplitudes, phsp投影向量
// d_grad_out: n_amplitudes, 输出梯度 ∂NLL/∂v
// 返回: NLL = -Σlog(|A_k·v|²) + nEvents·log(phsp_factor)
double computeFactorNLL(const cuComplex* d_amp, const cuComplex* d_vector, cuComplex* d_grad_out, int nEvents, int n_polar, int n_amplitudes, const double* d_weights = nullptr);

// 就地共轭复数数组
void conjugateComplexArray(cuComplex* data, int N);

// 向量axpy: y[i] += alpha * x[i], n较小(<~200)，单block完成
void axpyComplex(cuComplex* y, const cuComplex* x, cuComplex alpha, int n);

// 自定义核计算二次型 v^H·M·v，同时输出d_P_vec = M * v
// M: n×n Hermitian矩阵(行主序), v: n维向量(已共轭), n: 维度(<~200)
void computeQuadraticForm(const cuComplex* d_M, const cuComplex* d_v,
    cuComplex* d_P_vec, float* d_phsp_r, float* d_phsp_i, int n);

#endif
