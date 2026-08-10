#include <ComputeNLL.cuh>
#include <complex>
#include "ComplexType.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <iostream>
#include <stdio.h>
#include <vector>

constexpr int kWarpSize = 32;
constexpr int kBlockSize = 256;
constexpr int kWPBlock = kBlockSize / kWarpSize;

//////////////////////////
/////// 快速似然值+梯度：CUBLAS_CGEMV + 简单kernel + CUB规约 ///////
//////////////////////////

// Kernel: 每event计算|S|²→factor, 写w=S/factor, 累加-log(factor)*weight到NLL
__global__ void computeFactorsAndWeightsKernel(
    ctComplex* __restrict__ d_S,          // 输入S=A^T·v；输出w=S/factor（含权重）
    double* __restrict__ d_nll,
    const double* __restrict__ d_weights, // 每event权重（null则=1）
    int nEvents, int n_polar)
{
    int evt = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ double s_logf_partial[256];
    double my_logf = 0.0;

    if (evt < nEvents) {
        ctComplex* S_evt = d_S + evt * n_polar;
        const ctFloat2* S_f2 = reinterpret_cast<const ctFloat2*>(S_evt);
        ctFloat factor = CTF(0.0);
#pragma unroll
        for (int p = 0; p < n_polar; ++p) {
            ctFloat2 s = S_f2[p];
            factor += s.x * s.x + s.y * s.y;
        }

        ctFloat weight = (d_weights != nullptr) ? ctCastFloat(d_weights[evt]) : CTF(1.0);

        if (factor == CTF(0.0)) {
            my_logf = (double)(-ctLog(CTF(1e-10)) * weight);
#pragma unroll
            for (int p = 0; p < n_polar; ++p) {
                S_evt[p].x = CTF(0.0);
                S_evt[p].y = CTF(0.0);
            }
        }
        else {
            my_logf = (double)(-ctLog(factor) * weight);
            ctFloat inv_f = weight / factor;
            const ctFloat kMaxW = CTF(1e10);
#pragma unroll
            for (int p = 0; p < n_polar; ++p) {
                ctFloat2 s = S_f2[p];
                ctFloat wx = s.x * inv_f;
                ctFloat wy = s.y * inv_f;
                wx = ctFmin(ctFmax(wx, -kMaxW), kMaxW);
                wy = ctFmin(ctFmax(wy, -kMaxW), kMaxW);
                S_evt[p].x = wx;
                S_evt[p].y = wy;
            }
        }
    }

    // 块内归约 my_logf
    s_logf_partial[threadIdx.x] = my_logf;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_logf_partial[threadIdx.x] += s_logf_partial[threadIdx.x + s];
        }
        __syncthreads();
    }

    // 第一个线程将块和原子加到全局 NLL
    if (threadIdx.x == 0) {
        atomicAdd(d_nll, s_logf_partial[0]);
    }
}

// 就地共轭kernel: y[i] = conj(x[i])
__global__ void conjugateKernel(ctComplex* __restrict__ data, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) data[i].y = -data[i].y;
}

// 小向量axpy: y[i] += alpha * x[i], 单block
__global__ void axpyComplexKernel(ctComplex* y, const ctComplex* __restrict__ x, ctComplex alpha, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        y[i] = ctMake(
            y[i].x + alpha.x * x[i].x - alpha.y * x[i].y,
            y[i].y + alpha.x * x[i].y + alpha.y * x[i].x);
}

void axpyComplex(ctComplex* y, const ctComplex* x, ctComplex alpha, int n) {
    axpyComplexKernel << <1, 64 >> > (y, x, alpha, n);
}


// ===========================================================================
// 融合前向 kernel: matvec + |S|² + logf + w + NLL (n_polar 模板化)
// ===========================================================================

template<int NP>
__global__ void kern_fused_fwd(
    const ctComplex* __restrict__ A, int nA, int nE,
    const ctComplex* __restrict__ v, const double* __restrict__ W,
    ctComplex* __restrict__ w, double* __restrict__ N)
{
    int warp = threadIdx.x / kWarpSize, lane = threadIdx.x % kWarpSize;
    int evt = blockIdx.x * kWPBlock + warp, base = evt * NP;
    extern __shared__ ctFloat sbuf[];
    ctComplex* sv = reinterpret_cast<ctComplex*>(sbuf);
    double* sn = reinterpret_cast<double*>(sbuf + nA * 2);
    for (int i = threadIdx.x; i < nA; i += kBlockSize) sv[i] = v[i];
    if (lane == 0) sn[warp] = 0.0;
    __syncthreads();

    if (evt < nE) {
        ctFloat Sr[NP] = {CTF(0.0)}, Si[NP] = {CTF(0.0)};
        for (int i = lane; i < nA; i += kWarpSize) {
            ctComplex vi = sv[i]; ctFloat vx = vi.x, vy = vi.y;
            #pragma unroll
            for (int p = 0; p < NP; ++p) {
                ctComplex a = A[i + (base + p) * nA];
                Sr[p] += a.x * vx - a.y * vy;
                Si[p] += a.x * vy + a.y * vx;
            }
        }
        #pragma unroll
        for (int p = 0; p < NP; ++p) {
            ctFloat re = Sr[p], im = Si[p];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                re += ctShflDown(re, off);
                im += ctShflDown(im, off);
            }
            Sr[p] = re; Si[p] = im;
        }
        if (lane == 0) {
            ctFloat f = CTF(0.0);
            #pragma unroll
            for (int p = 0; p < NP; ++p) f += Sr[p] * Sr[p] + Si[p] * Si[p];
            ctFloat wt = (W != nullptr) ? ctCastFloat(W[evt]) : CTF(1.0);
            sn[warp] = (f == CTF(0.0)) ? (double)(-ctLog(CTF(1e-10)) * wt) : (double)(-ctLog(f) * wt);
            ctFloat inv = (f == CTF(0.0)) ? CTF(0.0) : wt / f;
            const ctFloat kM = CTF(1e10);
            #pragma unroll
            for (int p = 0; p < NP; ++p) {
                ctFloat wx = Sr[p] * inv, wy = Si[p] * inv;
                wx = ctFmin(ctFmax(wx, -kM), kM); wy = ctFmin(ctFmax(wy, -kM), kM);
                w[base + p] = ctMake(wx, wy);
            }
        }
    }
    __syncthreads();
    if (warp == 0) {
        double val = (lane < kWPBlock) ? sn[lane] : 0.0;
        val += ctShflDown(val, 4);
        val += ctShflDown(val, 2);
        val += ctShflDown(val, 1);
        if (lane == 0) atomicAdd(N, val);
    }
}

static void launch_fused_fwd(int nP, int nA, int nE,
    const ctComplex* dA, const ctComplex* dv, const double* dW,
    ctComplex* dw, double* dN)
{
    int g = (nE + kWPBlock - 1) / kWPBlock;
    int shm = nA * (int)sizeof(ctComplex) + kWPBlock * (int)sizeof(double);
    auto L = [&](auto* k) { k<<<g, kBlockSize, shm>>>(dA, nA, nE, dv, dW, dw, dN); };
    switch (nP) {
        case 2: L(&kern_fused_fwd<2>); break;   case 3: L(&kern_fused_fwd<3>); break;
        case 4: L(&kern_fused_fwd<4>); break;   case 5: L(&kern_fused_fwd<5>); break;
        case 6: L(&kern_fused_fwd<6>); break;   case 7: L(&kern_fused_fwd<7>); break;
        case 8: L(&kern_fused_fwd<8>); break;   case 9: L(&kern_fused_fwd<9>); break;
        case 10:L(&kern_fused_fwd<10>); break;  case 11:L(&kern_fused_fwd<11>); break;
        case 12:L(&kern_fused_fwd<12>); break;  case 14:L(&kern_fused_fwd<14>); break;
        case 16:L(&kern_fused_fwd<16>); break;  case 18:L(&kern_fused_fwd<18>); break;
        default: { // fallback: cublas + old kernel
            ctComplex* d_S; cudaMalloc(&d_S, nE * nP * sizeof(ctComplex));
            cublasHandle_t h; cublasCreate(&h);
            ctComplex alpha=ctMake(1,0), beta=ctMake(0,0);
            CUBLAS_CGEMV(h, CUBLAS_OP_T, nA, nE*nP, &alpha, dA, nA, dv, 1, &beta, d_S, 1);
            int gb=(nE+kBlockSize-1)/kBlockSize;
            computeFactorsAndWeightsKernel<<<gb,kBlockSize>>>(d_S,dN,dW,nE,nP);
            cudaMemcpy(dw, d_S, nE*nP*sizeof(ctComplex), cudaMemcpyDeviceToDevice);
            cudaFree(d_S); cublasDestroy(h);
        } break;
    }
}

// -----------------------------------------------------------------------------
// Auto-tune: 在初始化时测一次最优分块数 (chunk size 10-20MB 最稳定)
// -----------------------------------------------------------------------------
static int tuneChunkCount(cublasHandle_t h, int nA, int nT,
                           const ctComplex* dA, const ctComplex* dv, ctComplex* dS) {
    static int s_best_nch = 0;
    if (s_best_nch != 0) return s_best_nch;

    ctComplex a1 = ctMake(1,0), b0 = ctMake(0,0);
    float best = 1e9f; int best_nch = 1;
    for (int nch : {1, 2, 4, 8, 16, 32, 64}) {
        int cp = nT / nch;
        if (cp < 1000) break;
        long long colOff = (long long)cp * nA;  // elements per chunk
        // Warmup
        for (int w = 0; w < 3; ++w) {
            for (int k = 0; k < nch; ++k)
                CUBLAS_CGEMV(h, CUBLAS_OP_T, nA, cp, &a1, dA + k * colOff, nA,
                            dv, 1, &b0, dS + k * cp, 1);
            cudaDeviceSynchronize();
        }
        cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
        cudaEventRecord(s);
        for (int i = 0; i < 10; ++i) {
            for (int k = 0; k < nch; ++k)
                CUBLAS_CGEMV(h, CUBLAS_OP_T, nA, cp, &a1, dA + k * colOff, nA,
                            dv, 1, &b0, dS + k * cp, 1);
            cudaDeviceSynchronize();
        }
        cudaEventRecord(e); cudaEventSynchronize(e);
        float ms; cudaEventElapsedTime(&ms, s, e);
        cudaEventDestroy(s); cudaEventDestroy(e);
        float t = ms / 10.0f;
        if (t < best) { best = t; best_nch = nch; }
    }
    s_best_nch = best_nch;
    return best_nch;
}

// -----------------------------------------------------------------------------
// 主函数: 计算 NLL + 梯度 (cublas 分块 + float logf/div)
// -----------------------------------------------------------------------------
double computeFactorNLL(const ctComplex* d_amp, const ctComplex* d_vector,
    ctComplex* d_grad_out,
    int nEvents, int n_polar, int n_amplitudes,
    const double* d_weights,
    ctComplex* d_w_out)
{
    const int nTotal = nEvents * n_polar;
    const long long strideA = (long long)n_amplitudes;  // column stride for A

    // ----- 静态缓冲区和 cublas handle (只分配一次) -----
    static ctComplex* s_d_S = nullptr;
    static cublasHandle_t s_handle = nullptr;
    static double* s_d_nll = nullptr;
    static int s_alloc_n = 0, s_nchunks = 0;
    if (s_alloc_n < nTotal || s_handle == nullptr) {
        if (s_d_S) cudaFree(s_d_S);
        if (s_d_nll) cudaFree(s_d_nll);
        if (s_handle) cublasDestroy(s_handle);
        cudaMalloc(&s_d_S, nTotal * sizeof(ctComplex));
        cudaMalloc(&s_d_nll, sizeof(double));
        cublasCreate(&s_handle);
        s_alloc_n = nTotal;
        s_nchunks = 0;  // trigger re-tune
    }

    // ----- Auto-tune chunk count (first call only) -----
    if (s_nchunks == 0) {
        s_nchunks = tuneChunkCount(s_handle, n_amplitudes, nTotal,
                                    d_amp, d_vector, s_d_S);
    }

    // ----- 第一步：S = A^T·v (cublas 分块) -----
    const int nch = s_nchunks;
    const int cp = nTotal / nch;
    const long long colStride = (long long)cp * n_amplitudes;  // elements to skip per chunk
    ctComplex a1 = ctMake(1, 0), b0 = ctMake(0, 0);
    for (int k = 0; k < nch; ++k) {
        CUBLAS_CGEMV(s_handle, CUBLAS_OP_T, n_amplitudes, cp, &a1,
                    d_amp + k * colStride, strideA, d_vector, 1, &b0,
                    s_d_S + k * cp, 1);
    }

    // ----- 第二步: factor + w + NLL (float logf/div, 一次性处理全部 S) -----
    cudaMemset(s_d_nll, 0, sizeof(double));
    int gridBlocks = (nEvents + kBlockSize - 1) / kBlockSize;
    computeFactorsAndWeightsKernel<<<gridBlocks, kBlockSize>>>(
        s_d_S, s_d_nll, d_weights, nEvents, n_polar);

    double raw_nll;
    cudaMemcpy(&raw_nll, s_d_nll, sizeof(double), cudaMemcpyDeviceToHost);

    if (d_w_out != nullptr) {
        cudaMemcpy(d_w_out, s_d_S, nTotal * sizeof(ctComplex), cudaMemcpyDeviceToDevice);
    }

    // ----- 第三步: 梯度 grad = -A·conj(w) (cublas 分块, 第一块覆盖, 后续累加) -----
    {
        int gradConj = (nTotal + kBlockSize - 1) / kBlockSize;
        conjugateKernel<<<gradConj, kBlockSize>>>(s_d_S, nTotal);

        ctComplex alpha = ctMake(-1.0f, 0.0f);
        ctComplex beta0 = ctMake(0.0f, 0.0f);
        ctComplex beta1 = ctMake(1.0f, 0.0f);
        for (int k = 0; k < nch; ++k) {
            ctComplex beta = (k == 0) ? beta0 : beta1;
            CUBLAS_CGEMV(s_handle, CUBLAS_OP_N, n_amplitudes, cp, &alpha,
                        d_amp + k * colStride, strideA,
                        s_d_S + k * cp, 1, &beta, d_grad_out, 1);
        }

        int gradZero = (n_amplitudes + kBlockSize - 1) / kBlockSize;
        conjugateKernel<<<gradZero, kBlockSize>>>(d_grad_out, n_amplitudes);
    }

    return raw_nll;
}

//////////////////////////
/////// 二次型 v^H·M·v 自定义核（小矩阵优化，替代CUBLAS_CGEMV+cublasCdotc）//////
//////////////////////////

// n<=64: 单block，M和v全放共享内存，warp shuffle规约
__global__ void quadratic_form_small(
    const ctComplex* __restrict__ M,
    const ctComplex* __restrict__ v,
    ctComplex* __restrict__ d_P_vec,
    ctPhspReal* __restrict__ result_r,
    ctPhspReal* __restrict__ result_i,
    int n)
{
    // 动态共享内存（float 版 64×64=16KB，double 版 64×64=64KB 超 48KB 上限，
    // 由调用方按 n 分配并设置 cudaFuncAttributeMaxDynamicSharedMemorySize）
    extern __shared__ ctComplex smem[];
    ctComplex* sM = smem;          // n×n
    ctComplex* sv = smem + n * n;  // n

    int nn = n * n;
    for (int idx = threadIdx.x; idx < nn; idx += blockDim.x) sM[idx] = M[idx];
    for (int idx = threadIdx.x; idx < n; idx += blockDim.x) sv[idx] = v[idx];
    __syncthreads();

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int num_warps = blockDim.x / 32;

    ctFloat warp_r = CTF(0.0), warp_i = CTF(0.0);

    // 每warp负责若干行: y[i] = Σ_j M[i,j] * v[j]（行主序=连续访问）
    for (int i = warp_id; i < n; i += num_warps) {
        ctFloat yr0 = CTF(0.0), yi0 = CTF(0.0), yr1 = CTF(0.0), yi1 = CTF(0.0),
                yr2 = CTF(0.0), yi2 = CTF(0.0), yr3 = CTF(0.0), yi3 = CTF(0.0);
        int row_off = i * n;
        int j;
        // 4路ILP: 步长128
        for (j = lane_id; j + 96 < n; j += 128) {
            ctComplex m0 = sM[row_off + j], m1 = sM[row_off + j + 32];
            ctComplex m2 = sM[row_off + j + 64], m3 = sM[row_off + j + 96];
            ctComplex v0 = sv[j], v1 = sv[j + 32], v2 = sv[j + 64], v3 = sv[j + 96];
            yr0 += m0.x * v0.x - m0.y * v0.y; yi0 += m0.x * v0.y + m0.y * v0.x;
            yr1 += m1.x * v1.x - m1.y * v1.y; yi1 += m1.x * v1.y + m1.y * v1.x;
            yr2 += m2.x * v2.x - m2.y * v2.y; yi2 += m2.x * v2.y + m2.y * v2.x;
            yr3 += m3.x * v3.x - m3.y * v3.y; yi3 += m3.x * v3.y + m3.y * v3.x;
        }
        for (; j + 32 < n; j += 64) {
            ctComplex m0 = sM[row_off + j], m1 = sM[row_off + j + 32];
            ctComplex v0 = sv[j], v1 = sv[j + 32];
            yr0 += m0.x * v0.x - m0.y * v0.y; yi0 += m0.x * v0.y + m0.y * v0.x;
            yr1 += m1.x * v1.x - m1.y * v1.y; yi1 += m1.x * v1.y + m1.y * v1.x;
        }
        for (; j < n; j += 32) {
            ctComplex m = sM[row_off + j]; ctComplex vj = sv[j];
            yr0 += m.x * vj.x - m.y * vj.y; yi0 += m.x * vj.y + m.y * vj.x;
        }
        ctFloat yr = yr0 + yr1 + yr2 + yr3, yi = yi0 + yi1 + yi2 + yi3;

        // warp内规约（先规约，再写d_P_vec，否则只存了lane0的部分和）
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            yr += ctShflDown(yr, off);
            yi += ctShflDown(yi, off);
        }
        // 规约后lane0持有完整y[i]，写入d_P_vec并累积phsp
        if (lane_id == 0) {
            d_P_vec[i] = ctMake(yr, yi);
            ctComplex vi = sv[i];
            warp_r += yr * vi.x + yi * vi.y;
            warp_i += yr * vi.y - yi * vi.x;
        }
    }

    // 跨warp规约
    __shared__ ctFloat sr[32], si[32];
    if (lane_id == 0) { sr[warp_id] = warp_r; si[warp_id] = warp_i; }
    __syncthreads();
    if (warp_id == 0) {
        ctFloat fr = CTF(0.0), fi = CTF(0.0);
        for (int w = lane_id; w < num_warps; w += 32) { fr += sr[w]; fi += si[w]; }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            fr += ctShflDown(fr, off);
            fi += ctShflDown(fi, off);
        }
        if (lane_id == 0) { *result_r = fr; *result_i = fi; }
    }
}

// n>64: 多block，v放共享内存，M从全局用__ldg读取，atomicAdd归并结果
__global__ void quadratic_form_large(
    const ctComplex* __restrict__ M,
    const ctComplex* __restrict__ v,
    ctComplex* __restrict__ d_P_vec,
    ctPhspReal* __restrict__ result_r,
    ctPhspReal* __restrict__ result_i,
    int n)
{
    extern __shared__ ctComplex sv[];
    int tid = threadIdx.x;
    for (int i = tid; i < n; i += blockDim.x) sv[i] = v[i];
    __syncthreads();

    int warp_id = tid / 32, lane_id = tid % 32, num_warps = blockDim.x / 32;
    int rows_per_block = (n + gridDim.x - 1) / gridDim.x;
    int r0 = blockIdx.x * rows_per_block, r1 = min(r0 + rows_per_block, n);

    ctFloat warp_r = CTF(0.0), warp_i = CTF(0.0);
    for (int i = r0 + warp_id; i < r1; i += num_warps) {
        ctFloat yr0 = CTF(0.0), yi0 = CTF(0.0), yr1 = CTF(0.0), yi1 = CTF(0.0),
                yr2 = CTF(0.0), yi2 = CTF(0.0), yr3 = CTF(0.0), yi3 = CTF(0.0);
        int row_off = i * n;
        int j;
        for (j = lane_id; j + 96 < n; j += 128) {
            ctComplex m0 = __ldg(M + row_off + j), m1 = __ldg(M + row_off + j + 32);
            ctComplex m2 = __ldg(M + row_off + j + 64), m3 = __ldg(M + row_off + j + 96);
            ctComplex v0 = sv[j], v1 = sv[j + 32], v2 = sv[j + 64], v3 = sv[j + 96];
            yr0 += m0.x * v0.x - m0.y * v0.y; yi0 += m0.x * v0.y + m0.y * v0.x;
            yr1 += m1.x * v1.x - m1.y * v1.y; yi1 += m1.x * v1.y + m1.y * v1.x;
            yr2 += m2.x * v2.x - m2.y * v2.y; yi2 += m2.x * v2.y + m2.y * v2.x;
            yr3 += m3.x * v3.x - m3.y * v3.y; yi3 += m3.x * v3.y + m3.y * v3.x;
        }
        for (; j + 32 < n; j += 64) {
            ctComplex m0 = __ldg(M + row_off + j), m1 = __ldg(M + row_off + j + 32);
            ctComplex v0 = sv[j], v1 = sv[j + 32];
            yr0 += m0.x * v0.x - m0.y * v0.y; yi0 += m0.x * v0.y + m0.y * v0.x;
            yr1 += m1.x * v1.x - m1.y * v1.y; yi1 += m1.x * v1.y + m1.y * v1.x;
        }
        for (; j < n; j += 32) {
            ctComplex m = __ldg(M + row_off + j); ctComplex vj = sv[j];
            yr0 += m.x * vj.x - m.y * vj.y; yi0 += m.x * vj.y + m.y * vj.x;
        }
        ctFloat yr = yr0 + yr1 + yr2 + yr3, yi = yi0 + yi1 + yi2 + yi3;

        // warp内规约（先规约，再写d_P_vec）
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            yr += ctShflDown(yr, off);
            yi += ctShflDown(yi, off);
        }
        if (lane_id == 0) {
            d_P_vec[i] = ctMake(yr, yi);
            ctComplex vi = sv[i];
            warp_r += yr * vi.x + yi * vi.y;
            warp_i += yr * vi.y - yi * vi.x;
        }
    }

    __shared__ ctFloat sr[32], si[32];
    if (lane_id == 0) { sr[warp_id] = warp_r; si[warp_id] = warp_i; }
    __syncthreads();
    if (warp_id == 0) {
        ctFloat fr = CTF(0.0), fi = CTF(0.0);
        for (int w = lane_id; w < num_warps; w += 32) { fr += sr[w]; fi += si[w]; }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            fr += ctShflDown(fr, off);
            fi += ctShflDown(fi, off);
        }
        if (lane_id == 0) {
            atomicAdd(result_r, fr);
            atomicAdd(result_i, fi);
        }
    }
}

void computeQuadraticForm(const ctComplex* d_M, const ctComplex* d_v,
    ctComplex* d_P_vec, ctPhspReal* d_phsp_r, ctPhspReal* d_phsp_i, int n)
{
    if (n <= 64) {
        int blk = (n <= 32) ? 32 : 128;
        size_t shm = (size_t)(n * n + n) * sizeof(ctComplex);
        // double 版 64×64×16B=64KB 超过默认 48KB，需显式允许
        cudaFuncSetAttribute(quadratic_form_small,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm);
        quadratic_form_small << <1, blk, shm >> > (d_M, d_v, d_P_vec, d_phsp_r, d_phsp_i, n);
    }
    else {
        int blk = 256;
        int warps_per_block = blk / 32;
        int grid = (n + warps_per_block - 1) / warps_per_block;
        if (grid < 16) grid = 16;
        if (grid > 256) grid = 256;
        cudaMemsetAsync(d_phsp_r, 0, sizeof(ctPhspReal));
        cudaMemsetAsync(d_phsp_i, 0, sizeof(ctPhspReal));
        quadratic_form_large << <grid, blk, n * sizeof(ctComplex) >> > (
            d_M, d_v, d_P_vec, d_phsp_r, d_phsp_i, n);
    }
}
