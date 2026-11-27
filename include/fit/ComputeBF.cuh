#ifndef COMPUTEBF_CUH
#define COMPUTEBF_CUH

#include "ComplexType.h"
#include <cuda_runtime.h>

// 计算分波积分和散射矩阵(单次评估; 供 truth/phsp 积分与拟合分数使用)
// d_square_integral: 可选 [npartials] Σ w²|A_i|⁴（每事件 (w·intensity)² 累加），
//   用于效率的 MC 统计误差（tf-pwa add_integral error 同款：它也是 Σ(w·f)²）；
//   不需要时传 nullptr。
// d_event_weights: 可选 [nEvents] 逐事件权重（nullptr = 全部 1）。
//   partial/scattering/total 累加 w·f，square 累加 (w·f)² —— 与 NLL 的
//   Σ w / Σ w·f 归一化约定一致（w 是"事例数权重"，重要性权重 w∝1/|M_gen|² 时
//   同时给出正确的二阶矩）。
//
// d_complex_result / d_result_matrix: **调用方预分配并复用的 workspace**
//   （分别 ≥ nEvents*npolar 与 ≥ ngls*nEvents*npolar）。
//   旧实现在函数内部 cudaMalloc/cudaFree 且不检查返回值：大 phsp_truth 下
//   150 个 batch × (1+2·n_free) 次调用累积上千次 ~百 MB 分配，碎片化后分配失败
//   → nullptr 进 cuBLAS/kernel → 报成 illegal memory access（issue #3）。
//   分配失败/核函数出错现在直接抛异常（不再只 printf）。
void computeBranchingFractions(
    const ctComplex* d_matrix,
    const ctComplex* d_vector,
    double* d_partial_integral,
    double* d_scattering_matrix,
    double* d_total_integral,
    double* d_square_integral,
    int* d_nSLvectors,
    int npartials, int nEvents, int ngls, int npolar,
    ctComplex* d_complex_result,
    ctComplex* d_result_matrix,
    const double* d_event_weights = nullptr);

// 从Jacobian和协方差计算分波误差(纯host端)
// bf_errors[i] = sqrt(J_i @ cov @ J_i^T)
void computeBFErrors(
    const double* J,
    const double* cov,
    double* h_bf_errors,
    int npartials, int n2);

#endif
