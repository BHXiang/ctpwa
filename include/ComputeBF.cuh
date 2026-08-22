#ifndef COMPUTEBF_CUH
#define COMPUTEBF_CUH

#include "ComplexType.h"
#include <cuda_runtime.h>

// 计算分波积分和散射矩阵(单次评估; 供 truth/phsp 积分与拟合分数使用)
void computeBranchingFractions(
    const ctComplex* d_matrix,
    const ctComplex* d_vector,
    double* d_partial_integral,
    double* d_scattering_matrix,
    double* d_total_integral,
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
