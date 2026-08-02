#include "ComplexType.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>

#include <ComputeHessian.cuh>

// 就地共轭kernel
__global__ void conjKernel(ctComplex* data, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) data[i].y = -data[i].y;
}

// 合并Step A+B: 加载B和v到共享内存，计算S和Bu（old convention [[R,C],[-C,R]]），
// 然后计算per-event Hessian（无原子操作）
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

    // --- 计算 Bu = tildeB @ u，old convention tildeB = [[R, C], [-C, R]] ---
    // Bu[0:n]   = R*vr + C*vi
    // Bu[n:2n]  = -C*vr + R*vi
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

    // --- 计算 S = u^T @ Bu = Σ_i (vr[i]*Bu[i] + vi[i]*Bu[n+i]) ---
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
    // tildeB = [[R, C], [-C, R]] (old convention)
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

        // 2. Per-event Hessian (含S和Bu计算，old convention [[R,C],[-C,R]])
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
// 关键：phsp_matrix_未经conjKernel修复（与data部分不同），存储为共轭: sP.y = -C_true
// 因此 Pu = P_stored * v (复数乘法) 恰好得到 tildeP_true @ u:
//   Re(P*v) = R_true*vr + C_true*vi  = (tildeP_true @ u)_top
//   Im(P*v) = R_true*vi - C_true*vr  = (tildeP_true @ u)_bot
// tildeP贡献用 [[R,-C],[C,R]] 约定来补偿共轭存储: sP.y=-C_true → -sP.y=C_true
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

        // +2*w*tildeP/T, 用 [[R,-C],[C,R]] 约定补偿共轭存储
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
#include "AmpGen.cuh"

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

    double Sr[32] = { 0 }, Si[32] = { 0 };
    for (int p = 0; p < nPolar; ++p) {
        double sre = 0.0, sim = 0.0;
        for (int a = 0; a < n_amp; ++a) {
            ctComplex amp_ap = d_amp[evt * nPolar * n_amp + p * n_amp + a];
            ctComplex v_a = d_v[a];
            sre += (double)v_a.x * (double)amp_ap.x - (double)v_a.y * (double)amp_ap.y;
            sim += (double)v_a.x * (double)amp_ap.y + (double)v_a.y * (double)amp_ap.x;
        }
        Sr[p] = sre;
        Si[p] = sim;
        d_S_re[evt * nPolar + p] = sre;
        d_S_im[evt * nPolar + p] = sim;
    }
    double I_val = 0.0;
    for (int p = 0; p < nPolar; ++p) I_val += Sr[p] * Sr[p] + Si[p] * Si[p];
    d_I[evt] = I_val;

    // Accumulate term3 = conj(S) * amp_a for phsp mixed Hessian
    if (d_phsp_t3_re) {
        for (int a = 0; a < n_amp; ++a) {
            double t3_re = 0.0, t3_im = 0.0;
            for (int p = 0; p < nPolar; ++p) {
                ctComplex amp_ap = d_amp[evt * nPolar * n_amp + p * n_amp + a];
                double ar = (double)amp_ap.x, ai = (double)amp_ap.y;
                t3_re += Sr[p] * ar + Si[p] * ai;
                t3_im += Sr[p] * ai - Si[p] * ar;
            }
            atomicAdd(&d_phsp_t3_re[a], t3_re);
            atomicAdd(&d_phsp_t3_im[a], t3_im);
        }
    }
}

// ============================================================
// Stage 1: per-block kernel — same-resonance Hessian + output g,dS
// Template: Npr = params per resonance, Nres = resonances in this block
// ============================================================
template<int Npr, int Nres>
__global__ void hessianStage1Kernel(
    const thrust::complex<double>* d_slamps,
    const ctComplex* d_v,
    const DeviceMomenta* d_momenta,
    const DecayNode* d_decayNodes, int decayChain_size,
    const SL* d_slComb,
    const DeviceResonance* d_resonances,
    const double* d_all_params,
    const int* d_global_idx,
    double* d_hess, int hess_ld,
    int nEvents, int nSL, int nPolar, double bf_d, double default_weight,
    const double* d_event_weights,
    // Temp output for cross-block stage 2
    const double* d_S_re_full,   // [nEv * nPolar] pre-computed full S
    const double* d_S_im_full,   // [nEv * nPolar]
    // Temp output for cross-block stage 2
    double* d_g_out,       // [nEv * NT]
    double* d_dS_re_out,   // [nEv * NT * nPolar]
    double* d_dS_im_out,   // [nEv * NT * nPolar]
    double* d_dF_re_out = nullptr,  // [nEv * nSL * Npr] ∂F/∂θ for mixed Hessian
    double* d_dF_im_out = nullptr,
    // Phsp accumulators
    double* d_phsp_I = nullptr,
    double* d_phsp_grad = nullptr,
    double* d_phsp_hessA = nullptr,
    int evt_offset = 0)
{
    static_assert(Npr >= 1 && Npr <= 3, "Npr must be 1-3");
    static_assert(Nres >= 1 && Nres <= 4, "Nres must be 1-4");

    int evt = blockIdx.x * blockDim.x + threadIdx.x;
    if (evt >= nEvents) return;
    int evt_abs = evt + evt_offset;
    int nTotal = d_momenta->n_events * nPolar;
    double weight = d_event_weights ? d_event_weights[evt] : default_weight;

    constexpr int NT = Npr * Nres;
    int sl_per_res = nSL / Nres;

    // ===== AD variables per resonance =====
    using AD = Var<double, Npr, true>;
    AD m0_ad[Nres], g_ad[Nres];
    int ftg[Nres][Npr];
    for (int r = 0; r < Nres; ++r) {
        int po = d_resonances[r].param_offset;
        m0_ad[r] = AD(d_all_params[po]);
        m0_ad[r].grad[0] = 1.0;
        g_ad[r] = AD(d_all_params[po + 1]);
        g_ad[r].grad[1] = 1.0;
        for (int j = 0; j < Npr; ++j)
            ftg[r][j] = d_global_idx[r * Npr + j];
    }

    // ===== Read full S and I from pre-pass kernel =====
    double Sr[32] = { 0 }, Si[32] = { 0 };
    for (int p = 0; p < nPolar; ++p) {
        Sr[p] = d_S_re_full[evt * nPolar + p];
        Si[p] = d_S_im_full[evt * nPolar + p];
    }
    double I_val = 0.0;
    for (int p = 0; p < nPolar; ++p) I_val += Sr[p] * Sr[p] + Si[p] * Si[p];
    if (I_val < 1e-30) return;
    double inv_I = 1.0 / I_val;

    // ===== Pass 2: dS[j][p] and d²S[j][k][p] =====
    double dS_re[NT][32] = { {0} }, dS_im[NT][32] = { {0} };
    double d2S_re[NT][NT][32] = { {{0}} }, d2S_im[NT][NT][32] = { {{0}} };

    for (int sl_idx = 0; sl_idx < nSL; ++sl_idx) {
        int res = sl_idx / sl_per_res;
        if (res >= Nres) continue;
        ctComplex vv = d_v[sl_idx];

        using CV = ComplexVar<double, Npr, true>;
        CV R_ad(1.0, 0.0);
        {
            const DeviceResonance& target = d_resonances[res];
            AD* m0p = &m0_ad[res];
            AD* gp = &g_ad[res];

            for (int ni = 0; ni < decayChain_size; ++ni) {
                const DecayNode& node = d_decayNodes[ni];
                const SL& sl = d_slComb[sl_idx * decayChain_size + ni];
                int L = sl.L;
                LorentzVector pM = d_momenta->getMomentum(evt_abs, node.mother_idx);
                LorentzVector pD1 = d_momenta->getMomentum(evt_abs, node.daug1_idx);
                LorentzVector pD2 = d_momenta->getMomentum(evt_abs, node.daug2_idx);
                double mm = pM.M();
                double md1 = pD1.M();
                double md2 = pD2.M();
                double qq = breakup_momentum(mm, md1, md2);

                AD m0_q0, md1_q0, md2_q0;
                if (node.mother_idx == target.particle_idx && node.mass[0] <= 0)
                    m0_q0 = *m0p;
                else if (node.mass[0] > 0) m0_q0 = AD(node.mass[0]);
                else m0_q0 = AD(1.0);
                if (node.mass[1] <= 0 && node.daug1_idx == target.particle_idx)
                    md1_q0 = *m0p;
                else md1_q0 = AD(node.mass[1] > 0 ? node.mass[1] : md1);
                if (node.mass[2] <= 0 && node.daug2_idx == target.particle_idx)
                    md2_q0 = *m0p;
                else md2_q0 = AD(node.mass[2] > 0 ? node.mass[2] : md2);

                AD q0_ad = computeQ0AD(m0_q0, md1_q0, md2_q0);
                AD q_ad(qq);
                bool is_res = (node.mother_idx == target.particle_idx && node.mass[0] <= 0);

                CV nf;
                if (is_res) {
                    AD params_arr[2] = {*m0p, *gp};
                    nf = computeNodeFactor<AD>(L, AD(mm), q_ad, q0_ad,
                                              params_arr, 2, target.type, nullptr, 0, bf_d);
                } else {
                    auto bf = Bf<AD>(L, q_ad, q0_ad, bf_d);
                    nf.real = bf; nf.imag = AD(0.0);
                }
                CV new_R;
                new_R.real = R_ad.real * nf.real - R_ad.imag * nf.imag;
                new_R.imag = R_ad.real * nf.imag + R_ad.imag * nf.real;
                R_ad = new_R;
            }
        }

        // Output ∂F/∂θ for mixed Hessian
        if (d_dF_re_out) {
            for (int j = 0; j < Npr; ++j) {
                int fidx = evt * nSL * Npr + sl_idx * Npr + j;
                d_dF_re_out[fidx] = R_ad.real.grad[j];
                d_dF_im_out[fidx] = R_ad.imag.grad[j];
            }
        }

        int j0 = res * Npr;
        for (int p = 0; p < nPolar; ++p) {
            auto sl_amp = d_slamps[sl_idx * nTotal + evt_abs * nPolar + p];
            double sl_re = sl_amp.real(), sl_im = sl_amp.imag();
            double t_re = (double)vv.x * sl_re - (double)vv.y * sl_im;
            double t_im = (double)vv.x * sl_im + (double)vv.y * sl_re;

            for (int j = 0; j < Npr; ++j) {
                double dFr = R_ad.real.grad[j], dFi = R_ad.imag.grad[j];
                dS_re[j0 + j][p] += dFr * t_re - dFi * t_im;
                dS_im[j0 + j][p] += dFr * t_im + dFi * t_re;
            }
            for (int j = 0; j < Npr; ++j) {
                for (int k = j; k < Npr; ++k) {
                    double d2Fr = R_ad.real.hess[j][k], d2Fi = R_ad.imag.hess[j][k];
                    d2S_re[j0 + j][j0 + k][p] += d2Fr * t_re - d2Fi * t_im;
                    d2S_im[j0 + j][j0 + k][p] += d2Fr * t_im + d2Fi * t_re;
                }
            }
        }
    }

    // ===== Gradient g[j] =====
    double g[NT] = { 0 };
    for (int p = 0; p < nPolar; ++p) {
        double cwr = Sr[p] * inv_I, cwi = -Si[p] * inv_I;
        for (int j = 0; j < NT; ++j)
            g[j] += cwr * dS_re[j][p] - cwi * dS_im[j][p];
    }
    for (int j = 0; j < NT; ++j) g[j] *= -2.0;

    // ===== Output g, dS to temp buffers =====
    for (int j = 0; j < NT; ++j)
        d_g_out[evt * NT + j] = g[j];
    for (int j = 0; j < NT; ++j)
        for (int p = 0; p < nPolar; ++p) {
            int idx = evt * NT * nPolar + j * nPolar + p;
            d_dS_re_out[idx] = dS_re[j][p];
            d_dS_im_out[idx] = dS_im[j][p];
        }

    // ===== Same-resonance Hessian → d_hess =====
    double H_loc[NT][NT] = { {0} };
    for (int j = 0; j < NT; ++j) {
        for (int k = j; k < NT; ++k) {
            double hjk = g[j] * g[k];  // Term A

            // Term B
            double termB = 0.0;
            for (int p = 0; p < nPolar; ++p)
                termB += dS_re[k][p] * dS_re[j][p] + dS_im[k][p] * dS_im[j][p];
            termB *= -2.0 * inv_I;
            hjk += termB;

            // Term C (same-resonance only)
            int rj = j / Npr, rk = k / Npr;
            if (rj == rk) {
                double termC = 0.0;
                for (int p = 0; p < nPolar; ++p) {
                    double cwr = Sr[p] * inv_I, cwi = -Si[p] * inv_I;
                    termC += cwr * d2S_re[j][k][p] - cwi * d2S_im[j][k][p];
                }
                termC *= -2.0;
                hjk += termC;
            }

            H_loc[j][k] = hjk;
            if (j != k) H_loc[k][j] = hjk;

            int gj = ftg[rj][j % Npr];
            int gk = ftg[rk][k % Npr];
            if (gj >= 0 && gk >= 0) {
                double contrib = weight * hjk;
                atomicAdd(&d_hess[gk * hess_ld + gj], contrib);
                if (j != k) atomicAdd(&d_hess[gj * hess_ld + gk], contrib);
            }
            // Phsp: I * (g·g^T - H) for same-resonance (use hess_ld, global stride)
            if (default_weight == 0.0 && d_phsp_hessA != nullptr && gj >= 0 && gk >= 0) {
                atomicAdd(&d_phsp_hessA[gk * hess_ld + gj], I_val * (g[j] * g[k] - hjk));
                if (j != k) atomicAdd(&d_phsp_hessA[gj * hess_ld + gk], I_val * (g[j] * g[k] - hjk));
            }
        }
    }

    // Phsp: accumulate I and I*g
    if (default_weight == 0.0 && d_phsp_I != nullptr)
        atomicAdd(d_phsp_I, I_val);
    if (default_weight == 0.0 && d_phsp_grad != nullptr) {
        for (int j = 0; j < NT; ++j) {
            int gj = ftg[j / Npr][j % Npr];
            if (gj >= 0) atomicAdd(&d_phsp_grad[gj], I_val * g[j]);
        }
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
    const thrust::complex<double>* d_slamps,        // [nSLtotal * nTotal]
    const double* d_g,                             // [nEv*Npr]
    const double* d_dS_re, const double* d_dS_im,  // [nEv*Npr*nPolar]
    const double* d_dF_re, const double* d_dF_im,  // [nEv*nSL*Npr]
    const int* d_global_idx,                       // [Npr]
    double* d_mixed, int mixed_ld,                 // [2*n_amp_total × P]
    int nEvents, int nSL, int Npr, int nPolar, int n_amp_total, int site,
    int nTotal_slamp,
    double default_weight, const double* d_event_weights,
    double* d_phsp_sum = nullptr,
    int evt_offset = 0)
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

            double dF_re = dF_re_ptr[j], dF_im = dF_im_ptr[j];

            for (int p = 0; p < nPolar; ++p) {
                ctComplex amp_ap = d_amp[evt * nPolar * n_amp_total + p * n_amp_total + global_a];
                double ar = (double)amp_ap.x, ai = (double)amp_ap.y;

                double sr = Sr_ptr[p], si = Si_ptr[p];

                int ds_idx = j * nPolar + p;
                double ds_re = dS_re_ptr[ds_idx], ds_im = dS_im_ptr[ds_idx];

                auto sl_amp = d_slamps[a * nTotal_slamp + evt_abs * nPolar + p];
                double sl_re = sl_amp.real(), sl_im = sl_amp.imag();

                // Term 1: Re/Im(conj(dS_j) · amp_a)
                term1_re += ds_re * ar + ds_im * ai;
                term1_im += ds_re * ai - ds_im * ar;

                // Term 2: Re/Im(conj(S) · slamp_a · dF_j)
                double sl_dF_re = sl_re * dF_re - sl_im * dF_im;
                double sl_dF_im = sl_re * dF_im + sl_im * dF_re;
                term2_re += sr * sl_dF_re + si * sl_dF_im;
                term2_im += sr * sl_dF_im - si * sl_dF_re;

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
