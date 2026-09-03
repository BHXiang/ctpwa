#include "ComplexType.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>

#include <ComputeHessian.cuh>
#include <CustomExpr.cuh>
#include "AmpGen.cuh"

// 就地共轭kernel
__global__ void conjKernel(ctComplex* data, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) data[i].y = -data[i].y;
}

// 合并Step A+B: 加载B和v到共享内存，计算S和Bu，然后计算per-event Hessian（无原子操作）
// 注意：d_B 已经过 conjKernel 共轭（sB = conj(A^H A) = R - iC），因此实际公式是
// 标准约定 tildeB = [[R,-C],[C,R]]：
//   Bu[0:n]  = R*vr - C*vi  = Re(B·v)
//   Bu[n:2n] = C*vr + R*vi  = Im(B·v)
// H = 4w·Bu·Bu^T/S² - 2w·tildeB/S（S = u^T·Bu = |Av|²）
__global__ void perEventHessianKernel(
    const ctComplex* __restrict__ d_B,       // chunk × n²
    const ctComplex* __restrict__ d_v,       // n
    const double* __restrict__ d_weights,    // chunk
    double* __restrict__ d_hessian_chunk,    // chunk × 4n² (output, per-event slot)
    int nEvents, int n)
{
    const int n2 = 2 * n;
    const int hess_sz = n2 * n2;
    const int evt = blockIdx.x;
    if (evt >= nEvents) return;

    // 动态共享内存（double 版 n=64 时 64×64×16B=64KB > 48KB 默认上限，
    // 调用方设置 cudaFuncAttributeMaxDynamicSharedMemorySize）
    extern __shared__ ctFloat hsmem[];
    ctComplex* sB  = reinterpret_cast<ctComplex*>(hsmem);          // n×n
    double*   svr  = reinterpret_cast<double*>(hsmem + n * n * (sizeof(ctComplex) / sizeof(ctFloat)));  // n
    double*   svi  = svr + n;                                      // n
    double*   sBu  = svi + n;                                      // 2n
    double*   sS   = sBu + 2 * n;                                  // 1

    // 协作加载 B_k → shared memory
    const ctComplex* Bk = d_B + evt * n * n;
    int nn = n * n;
    for (int idx = threadIdx.x; idx < nn; idx += blockDim.x)
        sB[idx] = Bk[idx];
    // 加载 v → shared memory (real/imag split)
    for (int idx = threadIdx.x; idx < n; idx += blockDim.x) {
        svr[idx] = d_v[idx].x;
        svi[idx] = d_v[idx].y;
    }
    __syncthreads();

    double weight = (d_weights != nullptr) ? d_weights[evt] : 1.0;

    // --- 计算 Bu = tildeB @ u ---
    // sB = conj(A^H A)（已共轭），设 B = A^H A = R + iC，则 sB.x = R, sB.y = -C。
    // 实际等价于标准约定 tildeB = [[R, -C], [C, R]]：
    //   Bu[0:n]   = R*vr - C*vi  = Re(B·v)
    //   Bu[n:2n]  = C*vr + R*vi  = Im(B·v)
    //（以下代码利用 sB.y = -C 直接写出，勿按字面 [[R,C],[-C,R]] 解读）
    for (int i = threadIdx.x; i < n2; i += blockDim.x) {
        double sum = 0.0;
        if (i < n) {
            for (int j = 0; j < n; ++j)
                sum += sB[i * n + j].x * svr[j] + sB[i * n + j].y * svi[j];
        } else {
            int ii = i - n;
            for (int j = 0; j < n; ++j)
                sum += -sB[ii * n + j].y * svr[j] + sB[ii * n + j].x * svi[j];
        }
        sBu[i] = sum;
    }
    __syncthreads();

    // --- 计算 S = u^T @ Bu = Σ_i (vr[i]*Bu[i] + vi[i]*Bu[n+i]) = |Av|² ---
    if (threadIdx.x == 0) {
        double S = 0.0;
        for (int i = 0; i < n; ++i)
            S += svr[i] * sBu[i] + svi[i] * sBu[n + i];
        if (S <= 0.0) S = 1e-30;
        *sS = S;
    }
    __syncthreads();

    double invS = 1.0 / *sS;
    double invS2 = invS * invS;
    double* hess_out = d_hessian_chunk + evt * hess_sz;

    // --- per-event Hessian: H = 4*w*Bu*Bu^T/S² - 2*w*tildeB/S ---
    // 同样基于共轭存储 sB（sB.x=R, sB.y=-C）：
    //   H_00 = -2wR/S, H_01 = +2wC/S, H_10 = -2wC/S, H_11 = -2wR/S
    //（即标准约定 [[R,-C],[C,R]] 的 tildeB 项）
    for (int idx = threadIdx.x; idx < hess_sz; idx += blockDim.x) {
        int i = idx / n2, j = idx % n2;
        double val = 4.0 * weight * sBu[i] * sBu[j] * invS2;

        int a, b;
        if (i < n && j < n) {
            a = i; b = j;
            val += -2.0 * weight * sB[a * n + b].x * invS;
        } else if (i < n && j >= n) {
            a = i; b = j - n;
            val += -2.0 * weight * sB[a * n + b].y * invS;
        } else if (i >= n && j < n) {
            a = i - n; b = j;
            val += 2.0 * weight * sB[a * n + b].y * invS;
        } else {
            a = i - n; b = j - n;
            val += -2.0 * weight * sB[a * n + b].x * invS;
        }
        hess_out[idx] = val;
    }
}

// Step C: 沿events维度归约 per-event Hessian → 单次atomicAdd到全局
__global__ void reduceHessianKernel(
    const double* __restrict__ d_hessian_chunk,  // nEvents × 4n²
    double* __restrict__ d_hessian_global,        // 4n²
    int nEvents, int n)
{
    const int n2 = 2 * n;
    const int hess_sz = n2 * n2;
    const int elem = blockIdx.x;   // 0..hess_sz-1, 每个block归约一个(i,j)
    if (elem >= hess_sz) return;

    __shared__ double s_partial[256];
    double sum = 0.0;
    for (int evt = threadIdx.x; evt < nEvents; evt += blockDim.x)
        sum += d_hessian_chunk[evt * hess_sz + elem];

    s_partial[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) s_partial[threadIdx.x] += s_partial[threadIdx.x + s];
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        unsigned long long* ptr = (unsigned long long*)(d_hessian_global + elem);
        unsigned long long old = *ptr, new_val;
        do {
            new_val = __double_as_longlong(__longlong_as_double(old) + s_partial[0]);
        } while (atomicCAS(ptr, old, new_val) != old);
    }
}

void computeDataHessianContrib(
    const ctComplex* d_amp, const ctComplex* d_vector,
    const double* d_weights,
    double* d_hessian,
    int nEvents, int n_polar, int n_amplitudes)
{
    const int n = n_amplitudes;
    const int n2 = 2 * n;
    const int stride_amp = n_polar * n;
    const int stride_B  = n * n;
    const int hess_sz   = n2 * n2;
    constexpr int kBlockSize = 256;

    // 估算chunk大小（取1/4空闲内存）
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    // per-event: B(n² complex) + localHess(4n² double)
    size_t per_event = stride_B * sizeof(ctComplex) + hess_sz * sizeof(double);
    int max_chunk = (free_mem / 4) / per_event;
    if (max_chunk > nEvents) max_chunk = nEvents;
    if (max_chunk < 1) max_chunk = 1;

    // 一次性分配chunk缓冲区 (B + hessian_chunk)
    ctComplex *d_B;
    double *d_hessian_chunk;
    cudaMalloc(&d_B, max_chunk * stride_B * sizeof(ctComplex));
    cudaMalloc(&d_hessian_chunk, max_chunk * hess_sz * sizeof(double));

    cublasHandle_t handle;
    cublasCreate(&handle);
    ctComplex alpha = ctMake(1.0f, 0.0f);
    ctComplex beta  = ctMake(0.0f, 0.0f);

    for (int off = 0; off < nEvents; off += max_chunk) {
        int chunk = (off + max_chunk <= nEvents) ? max_chunk : (nEvents - off);
        const ctComplex* A_chunk = d_amp + off * stride_amp;
        const double* w_chunk = (d_weights != nullptr) ? d_weights + off : nullptr;

        // 1. B_k = A^H * A (cublas col-major → 读为row-major = A^H A)
        CUBLAS_CGEMM_STRIDED_BATCHED(handle,
            CUBLAS_OP_N, CUBLAS_OP_C, n, n, n_polar,
            &alpha, A_chunk, n, stride_amp, A_chunk, n, stride_amp,
            &beta, d_B, n, stride_B, chunk);
        int grid = (chunk * stride_B + kBlockSize - 1) / kBlockSize;
        conjKernel<<<grid, kBlockSize>>>(d_B, chunk * stride_B);

        // 2. Per-event Hessian (含S和Bu计算；B 已共轭 → 标准约定 [[R,-C],[C,R]])
        {
            size_t shm = (size_t)(n * n) * sizeof(ctComplex)
                       + (size_t)(5 * n) * sizeof(double);
            cudaFuncSetAttribute(perEventHessianKernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm);
            perEventHessianKernel<<<chunk, kBlockSize, shm>>>(d_B, d_vector, w_chunk,
                d_hessian_chunk, chunk, n);
        }

        // 3. 归约 per-event → 全局（每chunk仅 4n² 次atomicAdd）
        reduceHessianKernel<<<hess_sz, kBlockSize>>>(d_hessian_chunk, d_hessian,
            chunk, n);
    }

    cublasDestroy(handle);
    cudaFree(d_B); cudaFree(d_hessian_chunk);
}

// ========== phsp Hessian ==========
// 关键：phsp_matrix_未经conjKernel修复（与data部分不同），存储为共轭: sP = conj(M), sP.y = -C_true
// 因此 Pu = P_stored * v (复数乘法) 恰好得到 tildeB_old @ u（old convention [[R,C],[-C,R]]）:
//   Re(P*v) = R_true*vr + C_true*vi  = (tildeB_old @ u)_top
//   Im(P*v) = R_true*vi - C_true*vr  = (tildeB_old @ u)_bot
// tildeP贡献同理用 [[R,C],[-C,R]] 约定（与Pu一致，公式自洽）:
//   H_00,H_11 += 2wR/T, H_01 += +2wC/T, H_10 += -2wC/T
__global__ void phspHessianKernel(
    const ctComplex* __restrict__ P,      // n×n (共轭存储)
    const ctComplex* __restrict__ v,      // n
    double invT, double weight,
    double* __restrict__ d_hessian,
    int n)
{
    const int n2 = 2 * n;
    // 动态共享内存（double 版 64×64×16B=64KB 超限）
    extern __shared__ ctFloat hsmem[];
    ctComplex* sP      = reinterpret_cast<ctComplex*>(hsmem);      // n×n
    double*    sPu_real = reinterpret_cast<double*>(hsmem + n * n * (sizeof(ctComplex) / sizeof(ctFloat)));  // 2n

    int nn = n * n;
    for (int idx = threadIdx.x; idx < nn; idx += blockDim.x) sP[idx] = P[idx];
    __syncthreads();

    // Pu = P * v (复数乘法), 利用共轭存储得到正确的tildeP_true @ u
    for (int a = threadIdx.x; a < n; a += blockDim.x) {
        ctComplex pu = ctMake(0, 0);
        for (int b = 0; b < n; ++b) {
            ctComplex pb = sP[a * n + b], vb = v[b];
            pu.x += pb.x * vb.x - pb.y * vb.y;
            pu.y += pb.x * vb.y + pb.y * vb.x;
        }
        sPu_real[a] = pu.x;
        sPu_real[a + n] = pu.y;
    }
    __syncthreads();

    double invT2 = invT * invT;
    for (int idx = threadIdx.x; idx < n2 * n2; idx += blockDim.x) {
        int i = idx / n2, j = idx % n2;
        double val = 0.0;

        // -4*w*Pu*Pu^T/T²
        val += -4.0 * weight * sPu_real[i] * sPu_real[j] * invT2;

        // +2*w*tildeP/T, [[R,C],[-C,R]] 约定（sP.x=R, sP.y=-C）
        if (i < n && j < n)
            val += 2.0 * weight * sP[i * n + j].x * invT;
        else if (i < n && j >= n)
            val += -2.0 * weight * sP[i * n + (j - n)].y * invT;
        else if (i >= n && j < n)
            val += 2.0 * weight * sP[(i - n) * n + j].y * invT;
        else
            val += 2.0 * weight * sP[(i - n) * n + (j - n)].x * invT;

        unsigned long long* ptr = (unsigned long long*)(d_hessian + i * n2 + j);
        unsigned long long old = *ptr, new_val;
        do {
            new_val = __double_as_longlong(__longlong_as_double(old) + val);
        } while (atomicCAS(ptr, old, new_val) != old);
    }
}

void computePhspHessian(
    const ctComplex* d_phsp_matrix, const ctComplex* d_vector,
    double phsp_factor, double weight,
    double* d_hessian, int n)
{
    double invT = 1.0 / phsp_factor;
    {
        size_t shm = (size_t)(n * n) * sizeof(ctComplex)
                   + (size_t)(2 * n) * sizeof(double);
        cudaFuncSetAttribute(phspHessianKernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm);
        phspHessianKernel<<<1, 256, shm>>>(d_phsp_matrix, d_vector, invT, weight, d_hessian, n);
    }
}

// ========== Hessian约束投影: H_orig = J^T * H_ext * J ==========
// J是n_ext×n_orig的对角块矩阵: J[2*eid][2*oid]=cr, J[2*eid+1][2*oid+1]=ci
// 与extendVectorWithConstraints的实/虚部分离比例约定一致
// d_H_in: 2·n_ext × 2·n_ext 扩展Hessian (只读), d_H_out: 2·n_orig × 2·n_orig 输出
__global__ void reduceHessianKernel(
    const double* __restrict__ H_in,             // 2·n_ext × 2·n_ext (只读)
    double* __restrict__ H_out,                  // 2·n_orig × 2·n_orig (输出)
    const int* __restrict__ d_origin_ids,        // n_constraints
    const int* __restrict__ d_ext_ids,           // n_constraints
    const double* __restrict__ d_re_ratios,      // n_constraints (real ratio)
    const double* __restrict__ d_im_ratios,      // n_constraints (imag ratio)
    int n_constraints, int n_orig, int n_ext)
{
    const int n2o = 2 * n_orig;
    const int n2e = 2 * n_ext;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n2o || j >= n2o) return;

    // 1. 读取origin-origin块 (identity mapping)
    double val = H_in[i * n2e + j];

    // 2. 右乘J: 每个constraint列 j_ext → j_orig
    for (int c = 0; c < n_constraints; ++c) {
        int oid = d_origin_ids[c];
        int eid = d_ext_ids[c];
        double cr = d_re_ratios[c];
        double ci = d_im_ratios[c];

        if (j == 2 * oid)
            val += H_in[i * n2e + 2 * eid] * cr;
        if (j == 2 * oid + 1)
            val += H_in[i * n2e + 2 * eid + 1] * ci;
    }

    // 3. 左乘J^T (J是对角阵，J^T=J): 每个constraint行 i_ext → i_orig
    double val2 = val;
    for (int c = 0; c < n_constraints; ++c) {
        int oid = d_origin_ids[c];
        int eid = d_ext_ids[c];
        double cr = d_re_ratios[c];
        double ci = d_im_ratios[c];

        if (i == 2 * oid)
            val2 += cr * H_in[2 * eid * n2e + j];
        if (i == 2 * oid + 1)
            val2 += ci * H_in[(2 * eid + 1) * n2e + j];
    }

    // 4. 交叉项: 行和列都是约束的情况
    for (int ci_idx = 0; ci_idx < n_constraints; ++ci_idx) {
        int oi = d_origin_ids[ci_idx], ei = d_ext_ids[ci_idx];
        double cri = d_re_ratios[ci_idx], cii = d_im_ratios[ci_idx];
        if (i != 2 * oi && i != 2 * oi + 1) continue;

        for (int cj = 0; cj < n_constraints; ++cj) {
            int oj = d_origin_ids[cj], ej = d_ext_ids[cj];
            double crj = d_re_ratios[cj], cij = d_im_ratios[cj];
            if (j != 2 * oj && j != 2 * oj + 1) continue;

            double contrib = 0.0;
            bool i_real = (i == 2 * oi), j_real = (j == 2 * oj);

            if (i_real && j_real)
                contrib = cri * H_in[2 * ei * n2e + 2 * ej] * crj;
            else if (i_real && !j_real)
                contrib = cri * H_in[2 * ei * n2e + 2 * ej + 1] * cij;
            else if (!i_real && j_real)
                contrib = cii * H_in[(2 * ei + 1) * n2e + 2 * ej] * crj;
            else
                contrib = cii * H_in[(2 * ei + 1) * n2e + 2 * ej + 1] * cij;

            val2 += contrib;
        }
    }

    H_out[i * n2o + j] = val2;
}

void reduceHessianWithConstraints(
    const double* d_hessian_ext,
    double* d_hessian_out,
    const int* d_origin_ids, const int* d_ext_ids,
    const double* d_re_ratios, const double* d_im_ratios,
    int n_constraints, int n_orig, int n_ext)
{
    const int n2o = 2 * n_orig;
    dim3 block(16, 16);
    dim3 grid((n2o + 15) / 16, (n2o + 15) / 16);
    reduceHessianKernel<<<grid, block>>>(
        d_hessian_ext, d_hessian_out,
        d_origin_ids, d_ext_ids,
        d_re_ratios, d_im_ratios,
        n_constraints, n_orig, n_ext);
}
/**
 * Two-stage Hessian kernels — verified formula from lab_hessian/test_hessian.cu
 *
 * Stage 1 (hessianStage1Kernel):
 *   Per-block: computes g_e, dS_e, I_e, same-resonance Hessian (Terms A+B+C).
 *   Outputs g_e and dS_e to temp buffers for cross-block stage.
 *
 * Stage 2 (hessianCrossBlockKernel):
 *   Cross-block: reads two blocks' temp buffers, computes cross-resonance
 *   Terms A (g·g^T) and B (-2/I·Re(conj(dS)·dS)), atomicAdd to d_hess.
 *
 * Formula (per event e):
 *   S_{e,p} = Σ_sl v_sl · slamps_{sl,e,p} · F_sl(θ)
 *   I_e = Σ_p |S_{e,p}|²
 *   g_ej = -2/I_e · Σ_p Re(conj(S_{e,p}) · dS_{e,j,p})
 *   H_ejk = g_j·g_k - 2/I_e·Re(conj(dS_k)·dS_j) - 2/I_e·Re(conj(S)·d²S_jk)
 *         = Term A        + Term B                    + Term C (same-res only)
 */

 // ============================================================
 // Pre-pass: compute full S[p]=Σ_a v[a]·amp[a,e,p] from raw amplitudes
 // amp: [nEv * nPol * n_amp] layout, v: [n_amp] interleaved
 // Outputs S_re[nEv*nPol], S_im[nEv*nPol], I[nEv]
 // ============================================================
__global__ void computeSfromAmpsKernel(
    double* d_S_re, double* d_S_im, double* d_I,
    const ctComplex* d_amp,   // [nEv * nPol * n_amp]
    const ctComplex* d_v,     // [n_amp]
    int nEvents, int nPolar, int n_amp,
    double* d_phsp_t3_re = nullptr,   // [n_amp]
    double* d_phsp_t3_im = nullptr)   // [n_amp]
{
    int evt = blockIdx.x * blockDim.x + threadIdx.x;
    if (evt >= nEvents) return;

    // S 全量存全局输出（nPolar 无上限，不用栈数组）；I 与 term3 消费
    // 读回同一线程刚写出的 d_S_re/d_S_im（同地址，L2/寄存器命中）
    double I_val = 0.0;
    for (int p = 0; p < nPolar; ++p) {
        double sre = 0.0, sim = 0.0;
        for (int a = 0; a < n_amp; ++a) {
            ctComplex amp_ap = d_amp[evt * nPolar * n_amp + p * n_amp + a];
            ctComplex v_a = d_v[a];
            sre += (double)v_a.x * (double)amp_ap.x - (double)v_a.y * (double)amp_ap.y;
            sim += (double)v_a.x * (double)amp_ap.y + (double)v_a.y * (double)amp_ap.x;
        }
        d_S_re[evt * nPolar + p] = sre;
        d_S_im[evt * nPolar + p] = sim;
        I_val += sre * sre + sim * sim;
    }
    d_I[evt] = I_val;

    // Accumulate term3 = conj(S) * amp_a for phsp mixed Hessian
    if (d_phsp_t3_re) {
        for (int a = 0; a < n_amp; ++a) {
            double t3_re = 0.0, t3_im = 0.0;
            for (int p = 0; p < nPolar; ++p) {
                ctComplex amp_ap = d_amp[evt * nPolar * n_amp + p * n_amp + a];
                double ar = (double)amp_ap.x, ai = (double)amp_ap.y;
                double Sr = d_S_re[evt * nPolar + p];
                double Si = d_S_im[evt * nPolar + p];
                t3_re += Sr * ar + Si * ai;
                t3_im += Sr * ai - Si * ar;
            }
            atomicAdd(&d_phsp_t3_re[a], t3_re);
            atomicAdd(&d_phsp_t3_im[a], t3_im);
        }
    }
}

// float-A 版 S = A^T·v（A float2 读入、v double、输出交错 ctComplex S——θ 段共振梯度
// phsp 用，与 CUBLAS_CGEMV 输出布局一致（double2 交错 [re,im]），免整段 double 上转）
__global__ void computeSFromFloatAmpsKernel(
    ctComplex* d_S,             // [nEv * nPol] double2 交错输出
    const float2* d_amp,        // [nEv * nPol * n_amp] float2
    const ctComplex* d_v,       // [n_amp] double
    int nEvents, int nPolar, int n_amp)
{
    int evt = blockIdx.x * blockDim.x + threadIdx.x;
    if (evt >= nEvents) return;
    for (int p = 0; p < nPolar; ++p) {
        double sre = 0.0, sim = 0.0;
        for (int a = 0; a < n_amp; ++a) {
            float2 amp_ap = d_amp[evt * nPolar * n_amp + p * n_amp + a];
            ctComplex v_a = d_v[a];
            sre += v_a.x * (double)amp_ap.x - v_a.y * (double)amp_ap.y;
            sim += v_a.x * (double)amp_ap.y + v_a.y * (double)amp_ap.x;
        }
        d_S[evt * nPolar + p] = ctMake(sre, sim);
    }
}

// ============================================================
// 单节点 Bf 的 q0 链对共振态质量参数的一阶/二阶对数导数
// O = Bf(L, q, q0, d), q0 = breakup(m0, md1, md2)；md1/md2 经回退等于某自由
// 共振态质量参数 θ_r 时（aux 中为 VAR，无导数）:
//   dlnO_j = dlnBf/dq0 · ∂q0/∂θ_j
//   d2lnO_jk = d2lnBf/dq0² · ∂q0/∂θ_j·∂q0/∂θ_k + dlnBf/dq0 · ∂²q0/∂θ_j∂θ_k
// 只作用于 mass 槽位（d_param_map == 0）；im 恒 0（Bf 实数）。
// 与梯度 kernel 的跨-Bf q0 项一致；∂²q0 混合项（两子粒子同为自由共振态）
// 经 breakup_d2q_dm1dm2 计入。
// ============================================================
__device__ void addBfQ0HessianTerms(
    const DecayNode& node, int L, double q0,
    double m0_q0, double md1_q0, double md2_q0,
    double* dln_re, double* dln_im, double* d2ln_re, double* d2ln_im,
    const DeviceResonance* d_resonances, int R,
    const int* res_off, const int* res_cnt, const int* pm)
{
    double lq0 = dlnBf_dq0(L, q0, node.bf_d);
    double l2q0 = d2lnBf_dq0q0(L, q0, node.bf_d);
    bool both_daug = (node.mass[1] <= 0 && node.mass[2] <= 0
                      && node.daug1_idx == node.daug2_idx);
    for (int r = 0; r < R; ++r) {
        const DeviceResonance& res = d_resonances[r];
        if (res.param_count <= 0) continue;
        double dq0 = 0.0, d2q0 = 0.0;
        if (node.mass[1] <= 0 && node.daug1_idx == res.particle_idx) {
            dq0 += breakup_dq_dm(1, m0_q0, md1_q0, md2_q0);
            d2q0 += breakup_d2q_dm2(1, m0_q0, md1_q0, md2_q0);
        }
        if (node.mass[2] <= 0 && node.daug2_idx == res.particle_idx) {
            dq0 += breakup_dq_dm(2, m0_q0, md1_q0, md2_q0);
            d2q0 += breakup_d2q_dm2(2, m0_q0, md1_q0, md2_q0);
        }
        if (both_daug && node.daug1_idx == res.particle_idx)
            d2q0 += 2.0 * breakup_d2q_dm1dm2(m0_q0, md1_q0, md2_q0);
        if (dq0 == 0.0 && d2q0 == 0.0) continue;
        int off = res_off[r], cnt = res_cnt[r];
        for (int j_loc = 0; j_loc < cnt; ++j_loc) {
            int p = pm ? pm[off + j_loc] : (off + j_loc);
            if (p != 0) continue;   // 仅 mass 参数受 q0 链影响
            int jj = off + j_loc;
            dln_re[jj] += lq0 * dq0;
            double v2 = l2q0 * dq0 * dq0 + lq0 * d2q0;
            d2ln_re[jj * 16 + jj] += v2;
            // 与其他共振态质量参数的交叉项（q0 混合二阶导）
            for (int r2 = 0; r2 < R; ++r2) {
                if (r2 == r || d_resonances[r2].param_count <= 0) continue;
                double dq0b = 0.0;
                if (node.mass[1] <= 0 && node.daug1_idx == d_resonances[r2].particle_idx)
                    dq0b += breakup_dq_dm(1, m0_q0, md1_q0, md2_q0);
                if (node.mass[2] <= 0 && node.daug2_idx == d_resonances[r2].particle_idx)
                    dq0b += breakup_dq_dm(2, m0_q0, md1_q0, md2_q0);
                if (dq0b == 0.0) continue;
                double d2q0_cross = 0.0;
                bool pair = (node.mass[1] <= 0 && node.daug1_idx == res.particle_idx
                             && node.mass[2] <= 0 && node.daug2_idx == d_resonances[r2].particle_idx)
                         || (node.mass[1] <= 0 && node.daug1_idx == d_resonances[r2].particle_idx
                             && node.mass[2] <= 0 && node.daug2_idx == res.particle_idx);
                if (pair) d2q0_cross = breakup_d2q_dm1dm2(m0_q0, md1_q0, md2_q0);
                double v2x = l2q0 * dq0 * dq0b + lq0 * d2q0_cross;
                int off2 = res_off[r2], cnt2 = res_cnt[r2];
                for (int k_loc = 0; k_loc < cnt2; ++k_loc) {
                    int q = pm ? pm[off2 + k_loc] : (off2 + k_loc);
                    if (q != 0) continue;
                    int kk = off2 + k_loc;
                    d2ln_re[jj * 16 + kk] += v2x;
                    d2ln_re[kk * 16 + jj] += v2x;   // 对称
                }
            }
            break;  // 每共振态最多一个 mass 槽位
        }
    }
}

// ============================================================
// Custom 模型 Hessian（标量路径，参数数 P 运行时无上限；多共振态）
// F_total = Π 共振态因子 × Π Bf；log-derivative 累积（与梯度 kernel R>1 一致）:
//   dln[j]  = Σ_n ∂ln F_n/∂θ_j
//   d2ln[j][k] = Σ_n (∂²F_n/∂θ_j∂θ_k/F_n − ∂lnF_n/∂θ_j·∂lnF_n/∂θ_k)
//   dF_t[j] = F_total·dln[j]
//   d2F_t[j][k] = F_total·(d2ln[j][k] + dln[j]·dln[k])
// 位置空间 = 块内自由参数位置 [0, Npr)（Npr = Σ_r res_dF_count[r]），
// 每共振态 r 占据区间 [res_off[r], res_off[r]+res_cnt[r])；d_param_map 把位置
// 映到该共振态自身参数下标（0=mass, 1=width, ...）。
// ============================================================
__global__ void computeCustomHessianKernel(
    const thrust::complex<double>* d_slamp_tab,
    const ctComplex* d_v,
    const DeviceMomenta* d_momenta,
    const DecayNode* d_decayNodes, int decayChain_size,
    const SL* d_slComb,
    const DeviceResonance* d_resonances,
    const double* d_all_params,
    const double* d_all_channels,
    const int* d_global_idx,          // [Npr] 自由位置 → 全局 slot 下标
    const int* d_param_map,           // [Npr] 位置 j → 该共振态参数下标（null → 恒等）
    int Npr,                          // 自由参数数 = Σ_r res_dF_count[r]（≤ 16）
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
    int evt_offset = 0,
    int nSigma = 1,
    const DeviceMomenta* d_mom_tab = nullptr,
    const double* d_sign_tab = nullptr,
    const double* d_jit_out_full) {  // JIT 物化 F/dF/d2F（null → 解释器）
    int evt = blockIdx.x * blockDim.x + threadIdx.x;
    if (evt >= nEvents) return;
    int evt_abs = evt + evt_offset;
    int nTotal = d_momenta->n_events * nPolar;
    double weight = d_event_weights ? d_event_weights[evt] : default_weight;

    int R = Nres;
    if (R > 8) R = 8;
    if (Npr < 0) Npr = 0;
    if (Npr > 16) Npr = 16;
    if (Npr < 1) return;
    // 位置 j → 参数下标（fix_var 固定 / var_equal 合并的参数不求导；null → 恒等）
    const int* pm = d_param_map;
    const double* aux = d_all_channels;

    double dFr[16], dFi[16], d2Fr[16 * 16], d2Fi[16 * 16];

    // 读取 full S 和 I（pre-pass）
    // d_S_re/d_S_im 是段内布局（每段 nEv·nPol 独立分配，见
    // computeUnifiedHessian 的 cudaMalloc），用段内 evt 索引（与模板版
    // hessianStage1Kernel 522 行一致）；evt_abs 会越界（data/bkg 段）。
    double I_val = 0.0;
    for (int p = 0; p < nPolar; ++p) {
        double Sr = d_S_re_full[evt * nPolar + p];
        double Si = d_S_im_full[evt * nPolar + p];
        I_val += Sr * Sr + Si * Si;
    }
    if (I_val < 1e-30) return;
    double inv_I = 1.0 / I_val;

    // dS / d2S 累加（σ 求和）。
    // nPolar 无上限：dS[j][p] 需跨 SL/σ 累加，故按 PCHUNK 外层分块，栈数组
    // 只存 [16][PCHUNK]；g/termB 用跨 chunk 小累加器。
    // d2S 只被 termC（Σ_p cwr·d2S_re − cwi·d2S_im）消费，cwr/cwi 不依赖
    // SL/σ → 直接在 p 内层按 (j,k) 折叠成标量。
    const int PCHUNK = 128;
    double g_acc[16], termB_acc[16 * 16];
    double d2S_acc_re[16 * 16], d2S_acc_im[16 * 16];
    for (int j = 0; j < Npr; ++j) g_acc[j] = 0.0;
    for (int j = 0; j < Npr; ++j)
        for (int k = 0; k < Npr; ++k) {
            termB_acc[j * 16 + k] = 0.0;
            d2S_acc_re[j * 16 + k] = 0.0;
            d2S_acc_im[j * 16 + k] = 0.0;
        }

    for (int pc = 0; pc < nPolar; pc += PCHUNK) {
        int plo = pc;
        int phi = (pc + PCHUNK < nPolar) ? pc + PCHUNK : nPolar;
        double dS_re[16][PCHUNK], dS_im[16][PCHUNK];
        for (int j = 0; j < Npr; ++j)
            for (int p = plo; p < phi; ++p) { dS_re[j][p - plo] = 0; dS_im[j][p - plo] = 0; }

    // 每个 SL 组合（本 block nSL 个；SL 与共振态共享，需遍历 SL 求导）
    // 注意：dS 需要 Σ_sl v_sl · dF_j · slamp（对所有 SL 求和）
    for (int sl_idx = 0; sl_idx < nSL; ++sl_idx) {
        ctComplex vv = d_v[sl_idx];

        // σ 求和
        for (int s = 0; s < nSigma; ++s) {
            const DeviceMomenta* dm = (s == 0 || !d_mom_tab) ? d_momenta : &d_mom_tab[s];
            double sg = (s == 0) ? 1.0 : d_sign_tab[s];
            // d_slamp_tab 是全局布局（按每 GPU 全部事件分配，见
            // computeSLAmps 的 d_slamp_tab_ 分配循环），行距 = nSL·nTotal
            // （与模板版 hessianStage1Kernel 535 行 slamp_row 一致）
            const thrust::complex<double>* slam = d_slamp_tab + (size_t)s * ((size_t)nSL * nTotal);

            // 节点循环（标量）→ F_total, dF_total, d2F_total（log-derivative 累积）
            double Ftr = 1.0, Fti = 0.0;
            double dln_re[64], dln_im[64];
            double d2ln_re[16 * 16], d2ln_im[16 * 16];
            for (int j = 0; j < Npr; ++j) dln_re[j] = dln_im[j] = 0.0;
            for (int j = 0; j < Npr * 16; ++j) d2ln_re[j] = d2ln_im[j] = 0.0;

            for (int nodeIdx = 0; nodeIdx < decayChain_size; ++nodeIdx) {
                const DecayNode& node = d_decayNodes[nodeIdx];
                const SL& sl = d_slComb[nodeIdx + sl_idx * decayChain_size];
                int L = sl.L;
                LorentzVector pM  = dm->getMomentum(evt_abs, node.mother_idx);
                LorentzVector pD1 = dm->getMomentum(evt_abs, node.daug1_idx);
                LorentzVector pD2 = dm->getMomentum(evt_abs, node.daug2_idx);
                double mm = pM.M();
                double qq = breakup_momentum(mm, pD1.M(), pD2.M());
                double md1 = pD1.M(), md2 = pD2.M();

                int match_r = -1;
                for (int r = 0; r < R; ++r) {
                    if (node.mother_idx == d_resonances[r].particle_idx && node.mass[0] <= 0) {
                        match_r = r; break;
                    }
                }

                // q0 链质量回退（与梯度 kernel R>1 一致）：
                // m0 = 共振态名义质量；子粒子质量 = 固定质量，否则自由共振态
                // 参数质量，否则事件质量
                double m0_q0;
                if (match_r >= 0)
                    m0_q0 = (d_resonances[match_r].param_count > 0)
                        ? d_all_params[d_resonances[match_r].param_offset + 0] : 1.0;
                else if (node.mass[0] > 0)
                    m0_q0 = node.mass[0];
                else
                    m0_q0 = 1.0;

                double md1_q0 = node.mass[1];
                double md2_q0 = node.mass[2];
                if (md1_q0 <= 0)
                    for (int r = 0; r < R; ++r)
                        if (node.daug1_idx == d_resonances[r].particle_idx && d_resonances[r].param_count > 0)
                            { md1_q0 = d_all_params[d_resonances[r].param_offset + 0]; break; }
                if (md1_q0 <= 0) md1_q0 = md1;
                if (md2_q0 <= 0)
                    for (int r = 0; r < R; ++r)
                        if (node.daug2_idx == d_resonances[r].particle_idx && d_resonances[r].param_count > 0)
                            { md2_q0 = d_all_params[d_resonances[r].param_offset + 0]; break; }
                if (md2_q0 <= 0) md2_q0 = md2;
                double q0 = breakup_momentum(m0_q0, md1_q0, md2_q0);

                if (match_r >= 0) {
                    const DeviceResonance& res = d_resonances[match_r];
                    const double* rp = d_all_params + res.param_offset;
                    int P_r = res.param_count;
                    if (P_r > 16) P_r = 16;
                    int off = d_res_off[match_r];
                    int cnt = d_res_cnt[match_r];

                    double Fr, Fi, dFr[16], dFi[16], d2Fr[16 * 16], d2Fi[16 * 16];
                    if (d_jit_out_full && nodeIdx == jit_target_node) {
                        // JIT 物化读取（pass-1 已算 F/dF/d2F；nvals = 2+2P+2P²）
                        const double* jb = d_jit_out_full
                            + ((size_t)s * nEvents + evt) * nSL * (2 + 2 * P_r + 2 * P_r * P_r)
                            + (size_t)sl_idx * (2 + 2 * P_r + 2 * P_r * P_r);
                        Fr = jb[0]; Fi = jb[1];
                        for (int j = 0; j < P_r; ++j) {
                            dFr[j] = jb[2 + 2 * j];
                            dFi[j] = jb[2 + 2 * j + 1];
                        }
                        for (int j = 0; j < P_r; ++j)
                            for (int k = 0; k < P_r; ++k) {
                                int b = 2 + 2 * P_r + 2 * (j * P_r + k);
                                d2Fr[j * P_r + k] = jb[b];
                                d2Fi[j * P_r + k] = jb[b + 1];
                            }
                    } else if (res.type == ResModelType::Interp) {
                        interpEval(aux + res.aux_offset, mm, Fr, Fi, dFr, dFi, P_r);
                        for (int j = 0; j < P_r; ++j)
                            for (int k = 0; k < P_r; ++k) { d2Fr[j * P_r + k] = 0; d2Fi[j * P_r + k] = 0; }
                    } else {
                        double p1_P = pD1.P(), p1_E = pD1.E;
                        double p1_ct = (p1_P > 0) ? pD1.Pz / p1_P : 0.0;
                        double p1_phi = atan2(pD1.Py, pD1.Px);
                        double p2_P = pD2.P(), p2_E = pD2.E;
                        double p2_ct = (p2_P > 0) ? pD2.Pz / p2_P : 0.0;
                        double p2_phi = atan2(pD2.Py, pD2.Px);
                        // P_r=0（固定/ONE 模型）: 只读值段（dF/d2F 段不存在）
                        evalCustomAll(aux, res.aux_offset, mm, qq, q0, L, node.bf_d,
                            md1_q0, md2_q0,
                            p1_P, p1_E, p1_ct, p1_phi,
                            p2_P, p2_E, p2_ct, p2_phi,
                            rp, P_r, Fr, Fi, dFr, dFi, d2Fr, d2Fi,
                            /*compute_2nd=*/(P_r > 0));
                    }

                    double den = Fr * Fr + Fi * Fi;
                    // 本节点 log-derivative（参数空间 → 位置空间）
                    double ndln_re[16], ndln_im[16];
                    for (int j_loc = 0; j_loc < cnt; ++j_loc) {
                        int p = pm ? pm[off + j_loc] : (off + j_loc);
                        if (p < 0 || p >= P_r) continue;
                        ndln_re[j_loc] = (dFr[p] * Fr + dFi[p] * Fi) / den;
                        ndln_im[j_loc] = (dFi[p] * Fr - dFr[p] * Fi) / den;
                        dln_re[off + j_loc] += ndln_re[j_loc];
                        dln_im[off + j_loc] += ndln_im[j_loc];
                    }
                    // d2ln = d2F/F − dln·dln（复数；d2F 对称 → 全矩阵写入）
                    for (int j_loc = 0; j_loc < cnt; ++j_loc) {
                        int p = pm ? pm[off + j_loc] : (off + j_loc);
                        if (p < 0 || p >= P_r) continue;
                        for (int k_loc = 0; k_loc < cnt; ++k_loc) {
                            int q = pm ? pm[off + k_loc] : (off + k_loc);
                            if (q < 0 || q >= P_r) continue;
                            int jj = off + j_loc, kk = off + k_loc;
                            double t = (d2Fr[p * P_r + q] * Fr + d2Fi[p * P_r + q] * Fi) / den;
                            double u = (d2Fi[p * P_r + q] * Fr - d2Fr[p * P_r + q] * Fi) / den;
                            d2ln_re[jj * 16 + kk] += t - (ndln_re[j_loc] * ndln_re[k_loc] - ndln_im[j_loc] * ndln_im[k_loc]);
                            d2ln_im[jj * 16 + kk] += u - (ndln_re[j_loc] * ndln_im[k_loc] + ndln_im[j_loc] * ndln_re[k_loc]);
                        }
                    }

                    // 跨共振 Bf 项: 匹配节点 aux 内含的 Bf 因子其 q0 链含共振态
                    // 子粒子质量（CVAR_MD1/2 为 VAR，aux 无该导数）→ 补 q0 链的
                    // 一阶/二阶对数导数（仅 mass 参数受影响；BW 的 aux 不含 Bf，
                    // 与梯度 kernel 一致——Bf 因子整体不进 F_total）
                    if (node.has_bf && res.type != ResModelType::BW)
                        addBfQ0HessianTerms(node, L, q0, m0_q0, md1_q0, md2_q0,
                            dln_re, dln_im, d2ln_re, d2ln_im,
                            d_resonances, R, d_res_off, d_res_cnt, pm);

                    double nr = Ftr * Fr - Fti * Fi;
                    double ni = Ftr * Fi + Fti * Fr;
                    Ftr = nr; Fti = ni;
                } else if (node.has_bf) {
                    double bf = Bf<double>(L, qq, q0, node.bf_d);
                    Ftr *= bf; Fti *= bf;
                    addBfQ0HessianTerms(node, L, q0, m0_q0, md1_q0, md2_q0,
                        dln_re, dln_im, d2ln_re, d2ln_im,
                        d_resonances, R, d_res_off, d_res_cnt, pm);
                }
            }

            // F_total = Π F_n；dF_t/d2F_t 位置空间（见头注释公式）
            double dF_t[16], dF_ti[16];
            for (int j = 0; j < Npr; ++j) {
                dF_t[j]  = Ftr * dln_re[j] - Fti * dln_im[j];
                dF_ti[j] = Ftr * dln_im[j] + Fti * dln_re[j];
            }
            double d2F_t[16 * 16], d2F_ti[16 * 16];
            for (int j = 0; j < Npr; ++j)
                for (int k = 0; k < Npr; ++k) {
                    double l2_r = d2ln_re[j * 16 + k] + dln_re[j] * dln_re[k] - dln_im[j] * dln_im[k];
                    double l2_i = d2ln_im[j * 16 + k] + dln_re[j] * dln_im[k] + dln_im[j] * dln_re[k];
                    d2F_t[j * 16 + k]  = Ftr * l2_r - Fti * l2_i;
                    d2F_ti[j * 16 + k] = Ftr * l2_i + Fti * l2_r;
                }

            // dF 输出（mixed Hessian 用）: [nSigma × nEv*nSL*Npr]；与 p 无关，仅首 chunk 写
            if (pc == 0 && d_dF_re_out) {
                size_t row = (size_t)s * ((size_t)nEvents * nSL * Npr);
                size_t base = row + (size_t)evt * nSL * Npr + sl_idx * Npr;
                for (int j = 0; j < Npr; ++j) {
                    d_dF_re_out[base + j] = dF_t[j];
                    d_dF_im_out[base + j] = dF_ti[j];
                }
            }

            // dS[j][p] = Σ_sl v_sl · dF_j · slamp_sl
            // d2S[j][k][p] = Σ_sl v_sl · d2F_jk · slamp_sl
            for (int p = plo; p < phi; ++p) {
                // slamp 全局布局：用全局事件索引 evt_abs 和全局行距 nTotal
                // （与模板版 618 行 sl_amp 索引一致）
                size_t idx = (size_t)sl_idx * nTotal + evt_abs * nPolar + p;
                auto sl_amp = slam[idx];
                // t = vv × slamp
                double t_re = (double)vv.x * sl_amp.real() - (double)vv.y * sl_amp.imag();
                double t_im = (double)vv.x * sl_amp.imag() + (double)vv.y * sl_amp.real();
                for (int j = 0; j < Npr; ++j) {
                    dS_re[j][p - plo] += sg * (dF_t[j] * t_re - dF_ti[j] * t_im);
                    dS_im[j][p - plo] += sg * (dF_t[j] * t_im + dF_ti[j] * t_re);
                }
                // termC 标量折叠：cwr/cwi 与 SL/σ 无关，p 内层直接缩放累加
                // （与原 Σ_p cwr·d2S_re − cwi·d2S_im 数学等价；double 累加顺序
                // 略变，1e-16 级差异）
                double cwr_p = d_S_re_full[evt * nPolar + p] * inv_I;
                double cwi_p = -d_S_im_full[evt * nPolar + p] * inv_I;
                for (int j = 0; j < Npr; ++j) {
                    for (int k = 0; k < Npr; ++k) {
                        double a = d2F_t[j * 16 + k] * t_re - d2F_ti[j * 16 + k] * t_im;
                        double b = d2F_t[j * 16 + k] * t_im + d2F_ti[j * 16 + k] * t_re;
                        d2S_acc_re[j * 16 + k] += sg * (cwr_p * a - cwi_p * b);
                        d2S_acc_im[j * 16 + k] += sg * (cwr_p * b + cwi_p * a);
                    }
                }
            }
        }
    }

        // chunk 消费：g / termB / dS 输出（跨 chunk 累加）
        for (int j = 0; j < Npr; ++j) {
            double acc = 0.0;
            for (int p = plo; p < phi; ++p) {
                double cwr = d_S_re_full[evt * nPolar + p] * inv_I;
                double cwi = -d_S_im_full[evt * nPolar + p] * inv_I;
                acc += cwr * dS_re[j][p - plo] - cwi * dS_im[j][p - plo];
            }
            g_acc[j] += acc;
        }
        for (int j = 0; j < Npr; ++j)
            for (int k = 0; k < Npr; ++k) {
                double b = 0.0;
                for (int p = plo; p < phi; ++p)
                    b += dS_re[k][p - plo] * dS_re[j][p - plo] +
                         dS_im[k][p - plo] * dS_im[j][p - plo];
                termB_acc[j * 16 + k] += b;
            }
        for (int j = 0; j < Npr; ++j)
            for (int p = plo; p < phi; ++p) {
                int idx = evt * Npr * nPolar + j * nPolar + p;
                d_dS_re_out[idx] = dS_re[j][p - plo];
                d_dS_im_out[idx] = dS_im[j][p - plo];
            }
    }

    // g[j] = -2·Σ_p (cwr·dS_re − cwi·dS_im)
    double g[16];
    for (int j = 0; j < Npr; ++j) g[j] = -2.0 * g_acc[j];

    // 输出 g（供 cross-block stage 2）
    for (int j = 0; j < Npr; ++j) {
        int gj = d_global_idx[j];
        if (gj >= 0) d_g_out[evt * Npr + j] = g[j];
    }

    // 同块 Hessian：Term A + B + C（j,k 同 Custom 模型）
    for (int j = 0; j < Npr; ++j) {
        int gj = d_global_idx[j];
        if (gj < 0) continue;
        for (int k = j; k < Npr; ++k) {
            int gk = d_global_idx[k];
            if (gk < 0) continue;
            double hjk = g[j] * g[k];  // Term A
            double termB = termB_acc[j * 16 + k];  // 已在 chunk 循环内累加
            double termC = d2S_acc_re[j * 16 + k];  // 已在 SL×σ×p 循环内折叠
            termB *= -2.0 * inv_I;
            termC *= -2.0;
            hjk += termB + termC;
            double contrib = weight * hjk;
            atomicAdd(&d_hess[gk * hess_ld + gj], contrib);
            if (j != k) atomicAdd(&d_hess[gj * hess_ld + gk], contrib);
            // phsp: I·(g g^T − H)
            if (default_weight == 0.0 && d_phsp_hessA != nullptr) {
                double pv = I_val * (g[j] * g[k] - hjk);
                atomicAdd(&d_phsp_hessA[gk * hess_ld + gj], pv);
                if (j != k) atomicAdd(&d_phsp_hessA[gj * hess_ld + gk], pv);
            }
        }
    }
    // phsp: I 和 I·g
    if (default_weight == 0.0 && d_phsp_I != nullptr) atomicAdd(d_phsp_I, I_val);
    if (default_weight == 0.0 && d_phsp_grad != nullptr)
        for (int j = 0; j < Npr; ++j) {
            int gj = d_global_idx[j];
            if (gj >= 0) atomicAdd(&d_phsp_grad[gj], I_val * g[j]);
        }
}


// ============================================================
// Stage 2: cross-block kernel — Term A + Term B between blocks
// Reconstructs I from d_I buffer; d_I[evt] holds I_e for each event.
// ============================================================
__global__ void hessianCrossBlockKernel(
    const double* d_g_A,        // [nEv * NTA]
    const double* d_dS_re_A,    // [nEv * NTA * nPolar]
    const double* d_dS_im_A,
    const double* d_g_B,
    const double* d_dS_re_B,
    const double* d_dS_im_B,
    const double* d_I,           // [nEv] intensities for inv_I reconstruction
    const int* d_global_idx_A,   // [NTA]
    const int* d_global_idx_B,   // [NTB]
    int NTA, int NTB, int nEvents, int nPolar,
    double* d_hess, int hess_ld,
    double default_weight,
    const double* d_event_weights,
    // Phsp cross-block: accumulates 2*Re(conj(dS_B)·dS_A)
    double* d_phsp_hessA = nullptr,
    int phsp_ld = 1)
{
    int evt = blockIdx.x * blockDim.x + threadIdx.x;
    if (evt >= nEvents) return;
    double weight = d_event_weights ? d_event_weights[evt] : default_weight;

    double I_val = d_I[evt];
    if (I_val < 1e-30) return;
    double inv_I = 1.0 / I_val;

    const double* gA = d_g_A + evt * NTA;
    const double* gB = d_g_B + evt * NTB;
    const double* dsReA = d_dS_re_A + evt * NTA * nPolar;
    const double* dsImA = d_dS_im_A + evt * NTA * nPolar;
    const double* dsReB = d_dS_re_B + evt * NTB * nPolar;
    const double* dsImB = d_dS_im_B + evt * NTB * nPolar;

    for (int ja = 0; ja < NTA; ++ja) {
        int gja = d_global_idx_A[ja];
        if (gja < 0) continue;
        for (int kb = 0; kb < NTB; ++kb) {
            int gkb = d_global_idx_B[kb];
            if (gkb < 0) continue;

            // Term A: g[j] * g[k]
            double hjk = gA[ja] * gB[kb];

            // Term B: -2/I * Σ_p Re(conj(dS_B)·dS_A)
            double termB = 0.0;
            for (int p = 0; p < nPolar; ++p) {
                int ia = ja * nPolar + p;
                int ib = kb * nPolar + p;
                termB += dsReB[ib] * dsReA[ia] + dsImB[ib] * dsImA[ia];
            }
            termB *= -2.0 * inv_I;
            hjk += termB;

            double contrib = weight * hjk;
            atomicAdd(&d_hess[gkb * hess_ld + gja], contrib);
            // Symmetric: also fill [gja * hess_ld + gkb] (valid even if A≠B since
            // the total Hessian must be symmetric; cross term pair A→B and B→A
            // both run, so this is safe.)
            atomicAdd(&d_hess[gja * hess_ld + gkb], contrib);

            // Phsp cross: I * (g·g^T - H) = 2 * Re(conj(dS_B)·dS_A)
            if (d_phsp_hessA) {
                double phsp_val = 2.0 * termB / (-2.0 * inv_I); // = -termB * I
                // Actually: I*(g*g^T - H) = I*(g*g - g*g + 2/I*Re) = 2*Re
                // = I * 2/I * Re = 2 * Re
                // The raw Re sum is termB_Re = Σ_p Re(conj(dS_B)·dS_A)
                // So phsp_val = 2 * termB_Re
                // And termB = -2/I * termB_Re → termB_Re = -termB * I / 2
                double termB_Re = -termB * I_val * 0.5;
                // Actually let me recompute: termB = -2/I * Σ Re, so Σ Re = -termB * I / 2
                // Phsp contribution: 2 * Σ Re = -termB * I
                // Simpler: compute Σ Re directly
                double sumRe = 0.0;
                for (int p = 0; p < nPolar; ++p) {
                    int ia = ja * nPolar + p;
                    int ib = kb * nPolar + p;
                    sumRe += dsReB[ib] * dsReA[ia] + dsImB[ib] * dsImA[ia];
                }
                phsp_val = 2.0 * sumRe;
                atomicAdd(&d_phsp_hessA[gkb * phsp_ld + gja], phsp_val);
                atomicAdd(&d_phsp_hessA[gja * phsp_ld + gkb], phsp_val);
            }
        }
    }
}

// ============================================================
// Stage 3: Per-block mixed Hessian ∂²L/∂v_a∂θ_j (vθ same-block)
// H[Re(v_a),θ_j] = -2w/I·[Re(conj(dS_j)·amp_a) + Re(conj(S)·slamp_a·dF_j) + g_j·Re(conj(S)·amp_a)]
// H[Im(v_a),θ_j] = +2w/I·[Im(conj(dS_j)·amp_a) + Im(conj(S)·slamp_a·dF_j) + g_j·Im(conj(S)·amp_a)]
// ============================================================
__global__ void hessianMixedBlockKernel(
    const double* d_S_re, const double* d_S_im,   // [nEv*nPolar]
    const double* d_I,                             // [nEv]
    const ctComplex* d_amp,                        // [nEv_total*nPolar*n_amp_total]
    const thrust::complex<double>* d_slamp_tab,    // [nSigma × nSLtotal * nTotal]
    const double* d_g,                             // [nEv*Npr]
    const double* d_dS_re, const double* d_dS_im,  // [nEv*Npr*nPolar]
    const double* d_dF_re, const double* d_dF_im,  // [nSigma × nEv*nSL*Npr]
    const int* d_global_idx,                       // [Npr]
    double* d_mixed, int mixed_ld,                 // [2*n_amp_total × P]
    int nEvents, int nSL, int Npr, int nPolar, int n_amp_total, int site,
    int nTotal_slamp,
    double default_weight, const double* d_event_weights,
    double* d_phsp_sum = nullptr,
    int evt_offset = 0,
    int nSigma = 1, const double* d_sign_tab = nullptr)
{
    int evt = blockIdx.x * blockDim.x + threadIdx.x;
    if (evt >= nEvents) return;
    int evt_abs = evt + evt_offset;
    double w = d_event_weights ? d_event_weights[evt] : default_weight;
    double I_val = d_I[evt];
    if (I_val < 1e-30) return;
    double inv_I = 1.0 / I_val;
    bool is_phsp = (default_weight == 0.0 && d_phsp_sum != nullptr);

    const double* g_ptr = d_g + evt * Npr;
    const double* dS_re_ptr = d_dS_re + evt * Npr * nPolar;
    const double* dS_im_ptr = d_dS_im + evt * Npr * nPolar;
    const double* Sr_ptr = d_S_re + evt * nPolar;
    const double* Si_ptr = d_S_im + evt * nPolar;

    for (int a = 0; a < nSL; ++a) {
        int global_a = site + a;
        const double* dF_re_ptr = d_dF_re + evt * nSL * Npr + a * Npr;
        const double* dF_im_ptr = d_dF_im + evt * nSL * Npr + a * Npr;

        for (int j = 0; j < Npr; ++j) {
            int gj = d_global_idx[j];
            if (gj < 0) continue;

            double term1_re = 0.0, term1_im = 0.0;
            double term2_re = 0.0, term2_im = 0.0;
            double term3_re = 0.0, term3_im = 0.0;

            for (int p = 0; p < nPolar; ++p) {
                ctComplex amp_ap = d_amp[evt * nPolar * n_amp_total + p * n_amp_total + global_a];
                double ar = (double)amp_ap.x, ai = (double)amp_ap.y;

                double sr = Sr_ptr[p], si = Si_ptr[p];

                int ds_idx = j * nPolar + p;
                double ds_re = dS_re_ptr[ds_idx], ds_im = dS_im_ptr[ds_idx];

                // Term 1: Re/Im(conj(dS_j) · amp_a)
                term1_re += ds_re * ar + ds_im * ai;
                term1_im += ds_re * ai - ds_im * ar;

                // Term 2: Σ_σ sgn(σ)·Re/Im(conj(S) · slamp_σ(a) · dF_σ(j))
                //（全同粒子：slamp 与 ∂F/∂θ 都随 σ 变化，必须在求和内）
                for (int s = 0; s < nSigma; ++s) {
                    double sg = (s == 0) ? 1.0 : d_sign_tab[s];
                    auto sl_amp = d_slamp_tab[(size_t)s * ((size_t)nSL * nTotal_slamp)
                                            + (size_t)a * nTotal_slamp + evt_abs * nPolar + p];
                    double sl_re = sl_amp.real(), sl_im = sl_amp.imag();
                    double dFr = dF_re_ptr[(size_t)s * ((size_t)nEvents * nSL * Npr) + j];
                    double dFi = dF_im_ptr[(size_t)s * ((size_t)nEvents * nSL * Npr) + j];
                    double sl_dF_re = sl_re * dFr - sl_im * dFi;
                    double sl_dF_im = sl_re * dFi + sl_im * dFr;
                    term2_re += sg * (sr * sl_dF_re + si * sl_dF_im);
                    term2_im += sg * (sr * sl_dF_im - si * sl_dF_re);
                }

                // Term 3 coef: Re/Im(conj(S) · amp_a)
                term3_re += sr * ar + si * ai;
                term3_im += sr * ai - si * ar;
            }

            if (is_phsp) {
                atomicAdd(&d_phsp_sum[global_a * mixed_ld + gj], term1_re + term2_re);
                atomicAdd(&d_phsp_sum[(n_amp_total + global_a) * mixed_ld + gj], term1_im + term2_im);
            } else {
                double gj_val = g_ptr[j];
                double coeff = w * 2.0 * inv_I;
                double val_re = -coeff * (term1_re + term2_re + gj_val * term3_re);
                double val_im =  coeff * (term1_im + term2_im + gj_val * term3_im);
                atomicAdd(&d_mixed[global_a * mixed_ld + gj], val_re);
                atomicAdd(&d_mixed[(n_amp_total + global_a) * mixed_ld + gj], val_im);
            }
        }
        // d_phsp_t3 is now computed in the pre-pass (computeSfromAmpsKernel),
        // which covers all SL channels regardless of free-param status.
    }
}

// ============================================================
// Stage 4: Cross-block mixed Hessian (vθ cross terms)
// Term 2 = 0 (∂θ_amp_a = 0 for a in A, θ in B). Terms 1+3 survive.
// ============================================================
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
    double* d_phsp_sum = nullptr,
    int evt_offset = 0)
{
    int evt = blockIdx.x * blockDim.x + threadIdx.x;
    if (evt >= nEvents) return;
    double w = d_event_weights ? d_event_weights[evt] : default_weight;
    double I_val = d_I[evt];
    if (I_val < 1e-30) return;
    double inv_I = 1.0 / I_val;
    bool is_phsp = (default_weight == 0.0 && d_phsp_sum != nullptr);

    const double* Sr_ptr = d_S_re + evt * nPolar;
    const double* Si_ptr = d_S_im + evt * nPolar;
    const double* gB_ptr = d_g_B + evt * NTb;
    const double* dS_reB_ptr = d_dS_re_B + evt * NTb * nPolar;
    const double* dS_imB_ptr = d_dS_im_B + evt * NTb * nPolar;

    for (int a = 0; a < nSL_A; ++a) {
        int ga = site_A + a;

        double term3_re = 0.0, term3_im = 0.0;
        for (int p = 0; p < nPolar; ++p) {
            ctComplex amp_ap = d_amp[evt * nPolar * n_amp_total + p * n_amp_total + ga];
            double ar = (double)amp_ap.x, ai = (double)amp_ap.y;
            double sr = Sr_ptr[p], si = Si_ptr[p];
            term3_re += sr * ar + si * ai;
            term3_im += sr * ai - si * ar;
        }

        for (int jb = 0; jb < NTb; ++jb) {
            int gjb = d_gidx_B[jb];
            if (gjb < 0) continue;
            double gj_val = gB_ptr[jb];

            double term1_re = 0.0, term1_im = 0.0;
            for (int p = 0; p < nPolar; ++p) {
                ctComplex amp_ap = d_amp[evt * nPolar * n_amp_total + p * n_amp_total + ga];
                double ar = (double)amp_ap.x, ai = (double)amp_ap.y;
                int ds_idx = jb * nPolar + p;
                double ds_re = dS_reB_ptr[ds_idx], ds_im = dS_imB_ptr[ds_idx];
                term1_re += ds_re * ar + ds_im * ai;
                term1_im += ds_re * ai - ds_im * ar;
            }

            if (is_phsp) {
                atomicAdd(&d_phsp_sum[ga * mixed_ld + gjb], term1_re);
                atomicAdd(&d_phsp_sum[(n_amp_total + ga) * mixed_ld + gjb], term1_im);
            } else {
                double coeff = w * 2.0 * inv_I;
                double val_re = -coeff * (term1_re + gj_val * term3_re);
                double val_im =  coeff * (term1_im + gj_val * term3_im);
                atomicAdd(&d_mixed[ga * mixed_ld + gjb], val_re);
                atomicAdd(&d_mixed[(n_amp_total + ga) * mixed_ld + gjb], val_im);
            }
        }
        // d_phsp_t3 is NOT written here: term3 is a per-amplitude sum,
        // already accumulated by hessianMixedBlockKernel (stage 3).
        // Writing again from cross-block would double-count.
    }
}

// Trivial kernel: negate per-event weights in-place (out[i] = -in[i])
__global__ void negateWeightsKernel(double* out, const double* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = -in[i];
}

// Scale phsp amplitudes by per-event weight: amp[e,p,a] *= sqrt(weight[e] / W_total)
// Applied to ALL nPol*nAmp entries for each event, so the subsequent CUBLAS_CGEMM
// computes A_weighted^H A_weighted = A^H diag(w) A / W_total.
__global__ void scalePhspAmpsKernel(
    ctComplex* d_amp, const double* d_weights,
    int nEvents, int nPolar, int nAmp, double inv_W_total, int evt_offset)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nEvents * nPolar * nAmp;
    if (idx >= total) return;

    int evt = (idx / nAmp) / nPolar;
    double scale = (d_weights != nullptr)
        ? sqrt(d_weights[evt] * inv_W_total)
        : sqrt(inv_W_total);  // uniform weight: 1/N

    ctFloat s = ctCastFloat(scale);
    d_amp[idx].x *= s;
    d_amp[idx].y *= s;
}

// Reorder vv block from interleaved [Re0,Im0,Re1,Im1,...] to grouped [Re0..Re_n, Im0..Im_n]
__global__ void reorderVVBlockKernel(
    const double* __restrict__ H_in, double* __restrict__ H_out,
    int nv, int in_stride, int out_stride)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int n2 = 2 * nv;
    if (i >= n2 || j >= n2) return;

    int in_row = (i < nv) ? 2 * i : 2 * (i - nv) + 1;
    int in_col = (j < nv) ? 2 * j : 2 * (j - nv) + 1;

    H_out[i * out_stride + j] = H_in[in_row * in_stride + in_col];
}

void reorderVVBlockInterleavedToGrouped(double* H, int nv, int stride)
{
    int n2 = 2 * nv;
    double* d_tmp;
    cudaMalloc(&d_tmp, n2 * stride * sizeof(double));
    cudaMemcpy(d_tmp, H, n2 * stride * sizeof(double), cudaMemcpyDeviceToDevice);

    dim3 block(16, 16);
    dim3 grid((n2 + 15) / 16, (n2 + 15) / 16);
    reorderVVBlockKernel<<<grid, block>>>(d_tmp, H, nv, stride, stride);
    cudaDeviceSynchronize();
    cudaFree(d_tmp);
}

// ===========================================================================
// Fast Hessian paths (benchmarked in ~/pwa/hessian/morefast/bench_hess.cu)
// ===========================================================================
//
// Key idea 1 (Gauss-Newton block-reduce): Bu = A^H·S needs no per-event CGEMM.
//   H_GN = Σ_e 4w_e·Bu_e·Bu_e^T/I_e²   (drops the -2w·tildeB/I term)
//   Each block processes BLK_EVENTS events, accumulates H in shared memory
//   (tiled), then one atomicAdd pass to global.
//
// Key idea 2 (FULL fast path): the dropped term aggregates into ONE weighted
//   Gram matrix: Σ_e (w_e/I_e)·tildeB_e = tilde(A^H diag(w_e/I_e) A),
//   computed by scaling A and a single plain CGEMM. So the FULL Hessian
//   (same math as computeDataHessianContrib) costs ~4× less.
//
// Conventions: Bu_std = [Re(A^H S), Im(A^H S)] (standard convention), which
// matches the conj'ed-B convention of the per-event baseline above.
// ===========================================================================

constexpr int kFastBlock = 256;

// I[e] = Σ_p |S[e,p]|² (S pre-computed by CUBLAS_CGEMV)
__global__ void computeIntensityKernel(
    const ctComplex* __restrict__ dS, ctFloat* __restrict__ dI,
    int nEvents, int nPolar)
{
    int evt = blockIdx.x * kFastBlock + threadIdx.x;
    if (evt >= nEvents) return;
    ctFloat sum = CTF(0.0);
    const ctComplex* Se = dS + evt * nPolar;
    for (int p = 0; p < nPolar; ++p)
        sum += Se[p].x * Se[p].x + Se[p].y * Se[p].y;
    dI[evt] = sum;
}

// Scale A for the weighted Gram matrix. Negative weights (bkg contribution)
// can't be absorbed by a real sqrt, so split into positive/negative parts:
//   Σ_e (w_e/I_e)·B_e = A+^H A+ − A−^H A−,  A± = A·sqrt(|w_e|/I_e) for ±w_e > 0
__global__ void scaleAmpsForGramKernel(
    ctComplex* __restrict__ dA_pos, ctComplex* __restrict__ dA_neg,
    const ctComplex* __restrict__ dA_src, const ctFloat* __restrict__ dI,
    const double* __restrict__ dW,
    int nEvents, int nPolar, int nAmp)
{
    // NOTE: dA_src/dI/dW all point into the current chunk (event 0..nEvents-1)
    int idx = blockIdx.x * kFastBlock + threadIdx.x;
    int total = nEvents * nPolar * nAmp;
    if (idx >= total) return;
    int evt = (idx / nAmp) / nPolar;
    ctFloat Ival = dI[evt];
    double w = (dW != nullptr) ? dW[evt] : 1.0;
    ctComplex z = ctMake(0.0f, 0.0f);
    dA_pos[idx] = z;
    dA_neg[idx] = z;
    if (Ival <= CTF(0.0) || w == 0.0) return;
    ctFloat s = ctCastFloat(sqrt(fabs(w) / (double)Ival));
    ctComplex a = dA_src[idx];
    a.x *= s; a.y *= s;
    if (w > 0) dA_pos[idx] = a;
    else       dA_neg[idx] = a;
}

// a[i] -= b[i]
__global__ void subComplexKernel(ctComplex* a, const ctComplex* b, int n) {
    int i = blockIdx.x * kFastBlock + threadIdx.x;
    if (i < n) { a[i].x -= b[i].x; a[i].y -= b[i].y; }
}

// Block-reduce Gauss-Newton outer product:
//   H += Σ_events 4w·Bu_std·Bu_std^T/I²,  Bu_std = [Re(A^H S), Im(A^H S)]
// Each block: BLK_EVENTS events → Bu in shared → tile-accumulate H in shared →
// one atomicAdd pass to global.
template<int NP, int BLK_EVENTS, int TILE>
__global__ void gnBlockReduceKernel(
    const ctComplex* __restrict__ dA, const ctComplex* __restrict__ dS,
    const ctFloat* __restrict__ dI, const double* __restrict__ dW,
    double* __restrict__ dH, int nEv, int nA)
{
    const int n2 = 2 * nA;
    const int hess_sz = n2 * n2;
    const int blk = blockIdx.x;
    int evt0 = blk * BLK_EVENTS;
    if (evt0 >= nEv) return;

    // Bu/s4 用 double 计算与存储（与 legacy per-event kernel 的 double 精度一致；
    // 避免 float 在 I 极小事件上放大误差）
    extern __shared__ double hs_fast[];
    double* sBu_all = hs_fast;                               // BLK_EVENTS × n2
    double* s4_all = sBu_all + BLK_EVENTS * n2;              // BLK_EVENTS
    double* sH = s4_all + BLK_EVENTS;                        // TILE

    // Compute Bu for all valid events in this block
    int n_ok = 0;
    #pragma unroll 1
    for (int le = 0; le < BLK_EVENTS; ++le) {
        int evt = evt0 + le;
        if (evt >= nEv) break;
        double* sBu = sBu_all + n_ok * n2;
        const ctComplex* A_evt = dA + evt * NP * nA;

        ctFloat Ival = dI[evt];
        double w = (dW != nullptr) ? dW[evt] : 1.0;
        if (Ival <= CTF(0.0)) continue;

        // S per polarization (registers, double 精度)
        double Sp_re[10], Sp_im[10];
        #pragma unroll
        for (int p = 0; p < NP; ++p) {
            Sp_re[p] = (double)dS[evt * NP + p].x;
            Sp_im[p] = (double)dS[evt * NP + p].y;
        }

        // Bu_std[i]   = Σ_p Re(conj(A[p,i])·S[p])
        // Bu_std[n+i] = Σ_p Im(conj(A[p,i])·S[p])
        for (int i = threadIdx.x; i < n2; i += kFastBlock) {
            double acc = 0.0;
            if (i < nA) {
                #pragma unroll
                for (int p = 0; p < NP; ++p) {
                    ctComplex a = A_evt[p * nA + i];
                    acc += (double)a.x * Sp_re[p] + (double)a.y * Sp_im[p];
                }
            } else {
                int ii = i - nA;
                #pragma unroll
                for (int p = 0; p < NP; ++p) {
                    ctComplex a = A_evt[p * nA + ii];
                    acc += (double)a.x * Sp_im[p] - (double)a.y * Sp_re[p];
                }
            }
            sBu[i] = acc;
        }
        if (threadIdx.x == 0) s4_all[n_ok] = 4.0 * w / ((double)Ival * (double)Ival);
        __syncthreads();
        ++n_ok;
    }
    if (n_ok == 0) return;

    // Tile-by-tile accumulation
    for (int t0 = 0; t0 < hess_sz; t0 += TILE) {
        int tile = min(TILE, hess_sz - t0);
        for (int idx = threadIdx.x; idx < tile; idx += kFastBlock) sH[idx] = 0.0;
        __syncthreads();

        #pragma unroll 1
        for (int le = 0; le < n_ok; ++le) {
            const double* sBu = sBu_all + le * n2;
            double s4 = s4_all[le];
            for (int idx = threadIdx.x; idx < tile; idx += kFastBlock) {
                int gi = t0 + idx;
                int i = gi / n2, j = gi % n2;
                sH[idx] += s4 * sBu[i] * sBu[j];
            }
        }
        __syncthreads();

        for (int idx = threadIdx.x; idx < tile; idx += kFastBlock) {
            if (sH[idx] == 0.0) continue;
            unsigned long long* p = (unsigned long long*)(dH + t0 + idx);
            unsigned long long old = *p, nv;
            do {
                nv = __double_as_longlong(__longlong_as_double(old) + sH[idx]);
                unsigned long long prev = atomicCAS(p, old, nv);
                if (prev == old) break;
                old = prev;
            } while (true);
        }
        __syncthreads();
    }
}

// Add -2·tildeB(B_total) to H. Standard convention [[R,-C],[C,R]] (matches
// the conj'ed-B baseline):
//   H_00,H_11 += -2·R ; H_01 += +2·C ; H_10 += -2·C
__global__ void addTildeBStdKernel(
    const ctComplex* __restrict__ dB, double* __restrict__ dH, int n)
{
    const int n2 = 2 * n;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n2 || j >= n2) return;
    double val = 0.0;
    if (i < n && j < n)       val = -2.0 * (double)dB[i * n + j].x;         // R
    else if (i < n && j >= n) val = +2.0 * (double)dB[i * n + (j - n)].y;   // +C
    else if (i >= n && j < n) val = -2.0 * (double)dB[(i - n) * n + j].y;   // -C
    else                      val = -2.0 * (double)dB[(i - n) * n + (j - n)].x;  // R
    unsigned long long* p = (unsigned long long*)(dH + i * n2 + j);
    unsigned long long old = *p, nv;
    do {
        nv = __double_as_longlong(__longlong_as_double(old) + val);
        unsigned long long prev = atomicCAS(p, old, nv);
        if (prev == old) break;
        old = prev;
    } while (true);
}

// Template dispatch for gnBlockReduceKernel by n_polar and BLK_EVENTS
template<int NP>
static void launchGNBlockReduce(int blkEvt, int shm, int nBlk,
    const ctComplex* dA, const ctComplex* dS, const ctFloat* dI,
    const double* dW, double* dH, int nEv, int nA)
{
    auto L = [&](auto* k) {
        cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, shm);
        k<<<nBlk, kFastBlock, shm>>>(dA, dS, dI, dW, dH, nEv, nA);
    };
    switch (blkEvt) {
        case 8:  L(&gnBlockReduceKernel<NP,8,4096>);  break;
        case 4:  L(&gnBlockReduceKernel<NP,4,4096>);  break;
        case 2:  L(&gnBlockReduceKernel<NP,2,4096>);  break;
        default: L(&gnBlockReduceKernel<NP,1,4096>);  break;
    }
}

static void launchGNBlockReduceDispatch(int nP, int blkEvt, int shm, int nBlk,
    const ctComplex* dA, const ctComplex* dS, const ctFloat* dI,
    const double* dW, double* dH, int nEv, int nA)
{
    switch (nP) {
        case 2:  launchGNBlockReduce<2>(blkEvt, shm, nBlk, dA, dS, dI, dW, dH, nEv, nA); break;
        case 3:  launchGNBlockReduce<3>(blkEvt, shm, nBlk, dA, dS, dI, dW, dH, nEv, nA); break;
        case 4:  launchGNBlockReduce<4>(blkEvt, shm, nBlk, dA, dS, dI, dW, dH, nEv, nA); break;
        case 5:  launchGNBlockReduce<5>(blkEvt, shm, nBlk, dA, dS, dI, dW, dH, nEv, nA); break;
        case 6:  launchGNBlockReduce<6>(blkEvt, shm, nBlk, dA, dS, dI, dW, dH, nEv, nA); break;
        case 7:  launchGNBlockReduce<7>(blkEvt, shm, nBlk, dA, dS, dI, dW, dH, nEv, nA); break;
        case 8:  launchGNBlockReduce<8>(blkEvt, shm, nBlk, dA, dS, dI, dW, dH, nEv, nA); break;
        case 9:  launchGNBlockReduce<9>(blkEvt, shm, nBlk, dA, dS, dI, dW, dH, nEv, nA); break;
        case 10: launchGNBlockReduce<10>(blkEvt, shm, nBlk, dA, dS, dI, dW, dH, nEv, nA); break;
        default: break;  // unsupported → caller falls back
    }
}

// FULL Hessian fast path — same math as computeDataHessianContrib:
//   H = Σ_e 4w·Bu·Bu^T/I²  −  2·tildeB(A^H diag(w/I) A)
// d_weights: nullptr = 1.0 (data), negative values = bkg contribution.
// d_hessian: 2n×2n accumulator (caller zeroes first), in-place additive.
void computeDataHessianContribFast(
    const ctComplex* d_amp, const ctComplex* d_vector,
    const double* d_weights,
    double* d_hessian,
    int nEvents, int n_polar, int n_amplitudes)
{
    // Fall back to the per-event CGEMM path for unsupported n_polar
    if (n_polar < 2 || n_polar > 10) {
        computeDataHessianContrib(d_amp, d_vector, d_weights,
            d_hessian, nEvents, n_polar, n_amplitudes);
        return;
    }
    if (nEvents <= 0) return;

    const int n = n_amplitudes;
    const int n2 = 2 * n;
    const int hess_sz = n2 * n2;

    // Per-GPU temporary buffers (allocated on the current device)
    const int max_chunk = 50000;  // events per chunk (bounds memory)
    int chunk_evt = min(max_chunk, nEvents);
    int chunk_total = chunk_evt * n_polar * n;

    ctComplex* dA_pos; cudaMalloc(&dA_pos, chunk_total * sizeof(ctComplex));
    ctComplex* dA_neg; cudaMalloc(&dA_neg, chunk_total * sizeof(ctComplex));
    ctComplex* dS;     cudaMalloc(&dS, chunk_evt * n_polar * sizeof(ctComplex));
    ctFloat*   dI;     cudaMalloc(&dI, chunk_evt * sizeof(ctFloat));
    ctComplex* dB_total; cudaMalloc(&dB_total, n * n * sizeof(ctComplex));
    ctComplex* dB_neg;   cudaMalloc(&dB_neg, n * n * sizeof(ctComplex));

    cublasHandle_t handle; cublasCreate(&handle);
    ctComplex alpha = ctMake(1.0, 0.0);
    ctComplex beta0 = ctMake(0.0, 0.0);

    // Block-reduce config (auto-fit shared memory)
    int blkEvt = 8;
    size_t bu_shm = (size_t)blkEvt * n2 * sizeof(double) + blkEvt * sizeof(double);
    size_t shm = bu_shm + 4096 * sizeof(double);
    const size_t kMaxShm = 48000;
    while (shm > kMaxShm && blkEvt > 1) {
        blkEvt /= 2;
        bu_shm = (size_t)blkEvt * n2 * sizeof(double) + blkEvt * sizeof(double);
        shm = bu_shm + 4096 * sizeof(double);
    }

    for (int off = 0; off < nEvents; off += max_chunk) {
        int chunk = min(max_chunk, nEvents - off);
        const ctComplex* Achunk = d_amp + off * n_polar * n;
        const double* Wchunk = (d_weights != nullptr) ? d_weights + off : nullptr;

        // 1. S = A^T·v (CUBLAS_CGEMV), I = |S|²
        CUBLAS_CGEMV(handle, CUBLAS_OP_T, n, chunk * n_polar,
            &alpha, Achunk, n, d_vector, 1, &beta0, dS, 1);
        computeIntensityKernel<<<(chunk + kFastBlock - 1) / kFastBlock, kFastBlock>>>(
            dS, dI, chunk, n_polar);

        // 2. scale A into +/− buffers
        scaleAmpsForGramKernel<<<(chunk * n_polar * n + kFastBlock - 1) / kFastBlock, kFastBlock>>>(
            dA_pos, dA_neg, Achunk, dI, Wchunk, chunk, n_polar, n);
        cudaDeviceSynchronize();

        // 3. B_total = A+^H A+ − A−^H A−  (two plain CGEMMs, K = chunk·n_polar)
        CUBLAS_CGEMM(handle, CUBLAS_OP_N, CUBLAS_OP_C, n, n, chunk * n_polar,
            &alpha, dA_pos, n, dA_pos, n, &beta0, dB_total, n);
        CUBLAS_CGEMM(handle, CUBLAS_OP_N, CUBLAS_OP_C, n, n, chunk * n_polar,
            &alpha, dA_neg, n, dA_neg, n, &beta0, dB_neg, n);
        cudaDeviceSynchronize();
        subComplexKernel<<<(n * n + kFastBlock - 1) / kFastBlock, kFastBlock>>>(
            dB_total, dB_neg, n * n);

        // 4. outer product 4w·Bu·Bu^T/I² (block-reduce)
        {
            int nBlk = (chunk + blkEvt - 1) / blkEvt;
            launchGNBlockReduceDispatch(n_polar, blkEvt, (int)shm, nBlk,
                Achunk, dS, dI, Wchunk, d_hessian, chunk, n);
        }

        // 5. −2·tildeB(B_total)
        {
            dim3 blk2(16, 16), grd2((n2 + 15) / 16, (n2 + 15) / 16);
            addTildeBStdKernel<<<grd2, blk2>>>(dB_total, d_hessian, n);
        }
    }

    cublasDestroy(handle);
    cudaFree(dA_pos); cudaFree(dA_neg); cudaFree(dS); cudaFree(dI);
    cudaFree(dB_total); cudaFree(dB_neg);
}

// Gauss-Newton approximation (drops the -2w·tildeB/I term):
//   H_GN = Σ_e 4w·Bu·Bu^T/I²
// ~4.5× faster than the full CGEMM path; useful for iterative second-order
// optimizers (Fisher-like curvature). Falls back to the full fast path when
// n_polar is unsupported by the templates.
void computeDataHessianContribGN(
    const ctComplex* d_amp, const ctComplex* d_vector,
    const double* d_weights,
    double* d_hessian,
    int nEvents, int n_polar, int n_amplitudes)
{
    // Fall back to the per-event CGEMM path for unsupported n_polar
    if (n_polar < 2 || n_polar > 10) {
        computeDataHessianContrib(d_amp, d_vector, d_weights,
            d_hessian, nEvents, n_polar, n_amplitudes);
        return;
    }
    if (nEvents <= 0) return;

    const int n = n_amplitudes;
    const int n2 = 2 * n;

    const int max_chunk = 50000;
    int chunk_evt = min(max_chunk, nEvents);

    ctComplex* dS; cudaMalloc(&dS, chunk_evt * n_polar * sizeof(ctComplex));
    ctFloat*   dI; cudaMalloc(&dI, chunk_evt * sizeof(ctFloat));

    cublasHandle_t handle; cublasCreate(&handle);
    ctComplex alpha = ctMake(1.0, 0.0);
    ctComplex beta0 = ctMake(0.0, 0.0);

    int blkEvt = 8;
    size_t bu_shm = (size_t)blkEvt * n2 * sizeof(double) + blkEvt * sizeof(double);
    size_t shm = bu_shm + 4096 * sizeof(double);
    const size_t kMaxShm = 48000;
    while (shm > kMaxShm && blkEvt > 1) {
        blkEvt /= 2;
        bu_shm = (size_t)blkEvt * n2 * sizeof(double) + blkEvt * sizeof(double);
        shm = bu_shm + 4096 * sizeof(double);
    }

    for (int off = 0; off < nEvents; off += max_chunk) {
        int chunk = min(max_chunk, nEvents - off);
        const ctComplex* Achunk = d_amp + off * n_polar * n;
        const double* Wchunk = (d_weights != nullptr) ? d_weights + off : nullptr;

        CUBLAS_CGEMV(handle, CUBLAS_OP_T, n, chunk * n_polar,
            &alpha, Achunk, n, d_vector, 1, &beta0, dS, 1);
        computeIntensityKernel<<<(chunk + kFastBlock - 1) / kFastBlock, kFastBlock>>>(
            dS, dI, chunk, n_polar);

        int nBlk = (chunk + blkEvt - 1) / blkEvt;
        launchGNBlockReduceDispatch(n_polar, blkEvt, (int)shm, nBlk,
            Achunk, dS, dI, Wchunk, d_hessian, chunk, n);
    }

    cublasDestroy(handle);
    cudaFree(dS); cudaFree(dI);
}

// float2 A 段 → double2 上转（float_amps_ 模式: A 存 float2, B̄/hessian 消费端按块上转）
__global__ void castF2ToDouble2Kernel(
    const float2* __restrict__ src, cuDoubleComplex* __restrict__ dst, int total)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    dst[i] = make_cuDoubleComplex((double)src[i].x, (double)src[i].y);
}
