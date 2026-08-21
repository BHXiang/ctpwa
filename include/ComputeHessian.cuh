#ifndef COMPUTEHESSIAN_CUH
#define COMPUTEHESSIAN_CUH

#include "ComplexType.h"
#include <cuda_runtime.h>
#include <AmpGen.cuh>

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

// 完整 Hessian 快速路径（与 computeDataHessianContrib 数学完全一致，~4× 快）：
//   H = Σ_e 4w·Bu·Bu^T/I² − 2·tildeB(A^H diag(w/I) A)
// 不逐事件 CGEMM：tildeB 项聚合为一次加权 Gram 矩阵（两个普通 CGEMM），
// 外积项用 block-reduce kernel（共享内存累加 + 单次 atomicAdd）。
// 负权重（bkg）自动做 ± 拆分。n_polar ∈ [2,10] 走模板 kernel，其余回退到原路径。
void computeDataHessianContribFast(
    const ctComplex* d_amp, const ctComplex* d_vector,
    const double* d_weights,
    double* d_hessian,
    int nEvents, int n_polar, int n_amplitudes);

// Gauss-Newton 近似（丢弃 -2w·tildeB/I 项）：
//   H_GN = Σ_e 4w·Bu·Bu^T/I²   (~4.5× 快)
// 适合迭代式二阶优化器（Fisher-like 曲率）；需要精确协方差时用 Fast 版本。
void computeDataHessianContribGN(
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

// ============================================================================
// Device kernel declarations（rdc 多 TU 链接需要）
// ============================================================================

__global__ void computeSfromAmpsKernel(
    double* d_S_re, double* d_S_im, double* d_I,
    const ctComplex* d_amp, const ctComplex* d_v,
    int nEvents, int nPolar, int n_amp,
    double* d_phsp_t3_re, double* d_phsp_t3_im);

__global__ void computeCustomHessianKernel(
    const thrust::complex<double>* d_slamp_tab,
    const ctComplex* d_v,
    const DeviceMomenta* d_momenta,
    const DecayNode* d_decayNodes, int decayChain_size,
    const SL* d_slComb,
    const DeviceResonance* d_resonances,
    const double* d_all_params,
    const double* d_all_channels,
    const int* d_global_idx,
    const int* d_param_map,
    int Npr,
    double* d_hess, int hess_ld,
    int nEvents, int nSL, int nPolar, double default_weight,
    const double* d_event_weights,
    const double* d_S_re_full, const double* d_S_im_full,
    double* d_g_out,
    double* d_dS_re_out, double* d_dS_im_out,
    double* d_dF_re_out, double* d_dF_im_out,
    double* d_phsp_I, double* d_phsp_grad, double* d_phsp_hessA,
    const int* d_res_off,             // [Nres] 每共振态自由位置区间起始
    const int* d_res_cnt,             // [Nres] 每共振态自由位置数
    int Nres,
    int jit_target_node,              // JIT-full 物化节点下标（-1 → 解释器）
    int evt_offset,
    int nSigma,
    const DeviceMomenta* d_mom_tab,
    const double* d_sign_tab,
    const double* d_jit_out_full = nullptr);   // JIT 物化 F/dF/d2F（null → 解释器）

__global__ void hessianCrossBlockKernel(
    const double* d_g_A, const double* d_dS_re_A, const double* d_dS_im_A,
    const double* d_g_B,
    const double* d_dS_re_B, const double* d_dS_im_B,
    const double* d_I,
    const int* d_global_idx_A, const int* d_global_idx_B,
    int NTA, int NTB, int nEvents, int nPolar,
    double* d_hess, int hess_ld,
    double default_weight, const double* d_event_weights,
    double* d_phsp_hessA,
    int phsp_ld);

__global__ void hessianMixedBlockKernel(
    const double* d_S_re, const double* d_S_im,
    const double* d_I,
    const ctComplex* d_amp,
    const thrust::complex<double>* d_slamp_tab,
    const double* d_g, const double* d_dS_re, const double* d_dS_im,
    const double* d_dF_re, const double* d_dF_im,
    const int* d_global_idx,
    double* d_mixed, int mixed_ld,
    int nEvents, int nSL, int Npr, int nPolar, int n_amp_total, int site,
    int nTotal_slamp,
    double default_weight, const double* d_event_weights,
    double* d_phsp_sum,
    int evt_offset,
    int nSigma, const double* d_sign_tab);

__global__ void hessianCrossMixedKernel(
    const double* d_S_re, const double* d_S_im,
    const double* d_I,
    const ctComplex* d_amp,
    const double* d_g_B, const double* d_dS_re_B, const double* d_dS_im_B,
    const int* d_gidx_B, int NTb,
    int nSL_A, int site_A,
    int nEvents, int nPolar, int n_amp_total,
    double* d_mixed, int mixed_ld,
    double default_weight, const double* d_event_weights,
    double* d_phsp_sum,
    int evt_offset);

__global__ void negateWeightsKernel(double* out, const double* in, int n);

__global__ void scalePhspAmpsKernel(
    ctComplex* d_amp, const double* d_weights,
    int nEvents, int nPolar, int nAmp, double inv_W_total, int evt_offset);

#endif
