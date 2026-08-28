#ifndef COMPUTEBF_CUH
#define COMPUTEBF_CUH

#include "ComplexType.h"
#include <cuda_runtime.h>

// 计算分波积分和散射矩阵(单次评估; 供 truth/phsp 积分与拟合分数使用)
// d_square_integral: 可选 [npartials] Σ|A_i|⁴（每事件 intensity² 累加），
//   用于效率的 MC 统计误差（tf-pwa add_int_error 同款）；不需要时传 nullptr。
void computeBranchingFractions(
    const ctComplex* d_matrix,
    const ctComplex* d_vector,
    double* d_partial_integral,
    double* d_scattering_matrix,
    double* d_total_integral,
    double* d_square_integral,
    int* d_nSLvectors,
    int npartials, int nEvents, int ngls, int npolar);

// 从Jacobian和协方差计算分波误差(纯host端)
// bf_errors[i] = sqrt(J_i @ cov @ J_i^T)
void computeBFErrors(
    const double* J,
    const double* cov,
    double* h_bf_errors,
    int npartials, int n2);

#endif
