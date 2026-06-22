#ifndef COMPUTEBF_CUH
#define COMPUTEBF_CUH

#include <cuComplex.h>
#include <cuda_runtime.h>

// 计算分支比所需的积分和散射矩阵(单次评估)
void computeBranchingFractions(
    const cuComplex* d_matrix,
    const cuComplex* d_vector,
    double* d_partial_integral,
    double* d_scattering_matrix,
    double* d_total_integral,
    int* d_nSLvectors,
    int npartials, int nEvents, int ngls, int npolar);

// 从已累积的积分计算BF值(纯host端公式)
// bf[i] = scattering[i*np+i] * dataIntegral / (phsp_i/truth_i * truth_total)
void computeBFfromIntegrals(
    const double* h_phsp_partial,
    const double* h_truth_partial,
    const double* h_scattering,
    double* h_bf,
    int npartials,
    double dataIntegral);

// 从Jacobian和协方差计算BF误差(纯host端)
// bf_errors[i] = sqrt(J_i @ cov @ J_i^T)
void computeBFErrors(
    const double* J,
    const double* cov,
    double* h_bf_errors,
    int npartials, int n2);

#endif
