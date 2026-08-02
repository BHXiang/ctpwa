#ifndef COMPUTEHESSIAN_CUH
#define COMPUTEHESSIAN_CUH

#include "ComplexType.h"
#include <cuda_runtime.h>

// 计算data/bkg事件的Hessian贡献（含权重）
// d_amp: nEvents×n_polar×n_amplitudes (行主序)
// d_vector: n_amplitudes 复数参数
// d_weights: nEvents权重 (nullptr=全1)
// d_hessian: 2n×2n double矩阵（就地累加，调用方先清零）
void computeDataHessianContrib(
    const ctComplex* d_amp, const ctComplex* d_vector,
    const double* d_weights,
    double* d_hessian,
    int nEvents, int n_polar, int n_amplitudes);

// 计算phsp项的Hessian贡献: weight * [2*tildeP/T - 4*(tildeP*u)*(tildeP*u)^T/T²]
// d_phsp_matrix: n×n Hermitian (A_phsp^H * A_phsp, 未除N_phsp)
// d_vector: n 复数参数
// phsp_factor: v^H*M*v
// weight: N_data - W_bkg (phsp项权重)
// d_hessian: 2n×2n（就地累加）
void computePhspHessian(
    const ctComplex* d_phsp_matrix, const ctComplex* d_vector,
    double phsp_factor, double weight,
    double* d_hessian, int n);

// 将扩展Hessian (2·ext_n × 2·ext_n) 按约束Jacobian投影到原始空间 (2n × 2n)
// d_hessian_ext: 输入 2·n_ext × 2·n_ext, d_hessian_out: 输出 2·n_orig × 2·n_orig
void reduceHessianWithConstraints(
    const double* d_hessian_ext,
    double* d_hessian_out,
    const int* d_origin_ids, const int* d_ext_ids,
    const double* d_re_ratios, const double* d_im_ratios,
    int n_constraints, int n_orig, int n_ext);

// Reorder vv block from interleaved [Re0,Im0,Re1,Im1,...] to [Re0..Re_n, Im0..Im_n]
// H: [2*nv × 2*nv], stride = n2 (for full matrix: n2 = total, for vv-only: n2 = 2*nv)
void reorderVVBlockInterleavedToGrouped(
    double* H, int nv, int stride);
#endif
