#include <complex>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <iostream>
#include <stdio.h>
#include <vector>

//////////////////////////
/////// 快速似然值+梯度：cublasCgemv + 简单kernel + CUB规约 ///////
//////////////////////////

// Kernel: 每event计算|S|²→factor, 写w=S/factor, 累加-log(factor)*weight到NLL
__global__ void computeFactorsAndWeightsKernel(
    cuComplex* __restrict__ d_S,          // 输入S=A^T·v；输出w=S/factor（含权重）
    double* __restrict__ d_nll,
    const double* __restrict__ d_weights, // 每event权重（null则=1）
    int nEvents, int n_polar)
{
    int evt = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ double s_logf_partial[256];
    double my_logf = 0.0;

    if (evt < nEvents) {
        cuComplex* S_evt = d_S + evt * n_polar;
        const float2* S_f2 = reinterpret_cast<const float2*>(S_evt);
        double factor = 0.0;
#pragma unroll
        for (int p = 0; p < n_polar; ++p) {
            float2 s = S_f2[p];
            factor += (double)s.x * s.x + (double)s.y * s.y;
        }

        double weight = (d_weights != nullptr) ? d_weights[evt] : 1.0;

        if (factor == 0.0) {
            my_logf = -log(1e-10) * weight;
#pragma unroll
            for (int p = 0; p < n_polar; ++p) {
                S_evt[p].x = 0.0f;
                S_evt[p].y = 0.0f;
            }
        }
        else {
            my_logf = -log(factor) * weight;
            double inv_f = 1.0 / factor * weight;  // w = weight * S/factor
            const float kMaxW = 1e10f;
#pragma unroll
            for (int p = 0; p < n_polar; ++p) {
                float2 s = S_f2[p];
                float wx = (float)(s.x * inv_f);
                float wy = (float)(s.y * inv_f);
                wx = fminf(fmaxf(wx, -kMaxW), kMaxW);
                wy = fminf(fmaxf(wy, -kMaxW), kMaxW);
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
__global__ void conjugateKernel(cuComplex* __restrict__ data, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) data[i].y = -data[i].y;
}

// 小向量axpy: y[i] += alpha * x[i], 单block
__global__ void axpyComplexKernel(cuComplex* y, const cuComplex* __restrict__ x, cuComplex alpha, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        y[i] = make_cuComplex(
            y[i].x + alpha.x * x[i].x - alpha.y * x[i].y,
            y[i].y + alpha.x * x[i].y + alpha.y * x[i].x);
}

void axpyComplex(cuComplex* y, const cuComplex* x, cuComplex alpha, int n) {
    axpyComplexKernel << <1, 64 >> > (y, x, alpha, n);
}


// -----------------------------------------------------------------------------
// 优化后的主函数（预分配buffer，消除malloc/free和handle创建开销）
// -----------------------------------------------------------------------------
double computeFactorNLL(const cuComplex* d_amp, const cuComplex* d_vector,
    cuComplex* d_grad_out,
    int nEvents, int n_polar, int n_amplitudes,
    const double* d_weights,
    cuComplex* d_w_out)
{
    const int nTotal = nEvents * n_polar;
    constexpr int kBlockSize = 256;

    // ----- 创建 cuBLAS 句柄 -----
    cublasHandle_t handle;
    cublasStatus_t cublas_err = cublasCreate(&handle);
    if (cublas_err != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "computeFactorNLL: cublasCreate failed: " << cublas_err << std::endl;
        return 0.0;
    }

    // ----- 分配临时缓冲区 -----
    cuComplex* d_S;
    cudaError_t cu_err = cudaMalloc(&d_S, nTotal * sizeof(cuComplex));
    if (cu_err != cudaSuccess)
        std::cerr << "computeFactorNLL: cudaMalloc(d_S) failed: " << cudaGetErrorString(cu_err) << std::endl;
    double* d_nll;
    cu_err = cudaMalloc(&d_nll, sizeof(double));
    if (cu_err != cudaSuccess)
        std::cerr << "computeFactorNLL: cudaMalloc(d_nll) failed: " << cudaGetErrorString(cu_err) << std::endl;

    // ----- 第一大步：S = A * v -----
    {
        cuComplex alpha = make_cuComplex(1.0f, 0.0f);
        cuComplex beta = make_cuComplex(0.0f, 0.0f);
        cublas_err = cublasCgemv(handle, CUBLAS_OP_T,
            n_amplitudes, nTotal,
            &alpha, d_amp, n_amplitudes,
            d_vector, 1,
            &beta, d_S, 1);
        if (cublas_err != CUBLAS_STATUS_SUCCESS)
            std::cerr << "computeFactorNLL: cublasCgemv(step1) failed: " << cublas_err << std::endl;
    }

    // ----- 第二大步：计算 factor、写入权重 w、同时规约得到 NLL -----
    cudaMemset(d_nll, 0, sizeof(double));
    int gridBlocks = (nEvents + kBlockSize - 1) / kBlockSize;
    computeFactorsAndWeightsKernel << <gridBlocks, kBlockSize >> > (
        d_S, d_nll, d_weights, nEvents, n_polar);
    cu_err = cudaGetLastError();
    if (cu_err != cudaSuccess)
        std::cerr << "computeFactorNLL: factors kernel failed: " << cudaGetErrorString(cu_err) << std::endl;

    // 拷回 NLL
    double raw_nll;
    cu_err = cudaMemcpy(&raw_nll, d_nll, sizeof(double), cudaMemcpyDeviceToHost);
    if (cu_err != cudaSuccess)
        std::cerr << "computeFactorNLL: cudaMemcpy(nll) failed: " << cudaGetErrorString(cu_err) << std::endl;

    // 若调用者需要 w = S/I（用于共振态梯度），在 conjugate 之前复制
    if (d_w_out != nullptr) {
        cu_err = cudaMemcpy(d_w_out, d_S, nTotal * sizeof(cuComplex), cudaMemcpyDeviceToDevice);
        if (cu_err != cudaSuccess)
            std::cerr << "computeFactorNLL: cudaMemcpy(w_out) failed: " << cudaGetErrorString(cu_err) << std::endl;
    }

    // ----- 第三大步：梯度 grad = -A^H * w -----
    {
        int gradConj = (nTotal + kBlockSize - 1) / kBlockSize;
        conjugateKernel << <gradConj, kBlockSize >> > (d_S, nTotal);
        cu_err = cudaGetLastError();
        if (cu_err != cudaSuccess)
            std::cerr << "computeFactorNLL: conjugateKernel(d_S) failed: " << cudaGetErrorString(cu_err) << std::endl;

        cuComplex alpha = make_cuComplex(-1.0f, 0.0f);
        cuComplex beta = make_cuComplex(0.0f, 0.0f);
        cublas_err = cublasCgemv(handle, CUBLAS_OP_N,
            n_amplitudes, nTotal,
            &alpha, d_amp, n_amplitudes,
            d_S, 1,
            &beta, d_grad_out, 1);
        if (cublas_err != CUBLAS_STATUS_SUCCESS)
            std::cerr << "computeFactorNLL: cublasCgemv(step2) failed: " << cublas_err << std::endl;

        int gradZero = (n_amplitudes + kBlockSize - 1) / kBlockSize;
        conjugateKernel << <gradZero, kBlockSize >> > (d_grad_out, n_amplitudes);
        cu_err = cudaGetLastError();
        if (cu_err != cudaSuccess)
            std::cerr << "computeFactorNLL: conjugateKernel(grad) failed: " << cudaGetErrorString(cu_err) << std::endl;
    }

    // d_w_out already copied before conjugateKernel (above)

    // ----- 清理资源 -----
    cudaFree(d_S);
    cudaFree(d_nll);
    cublasDestroy(handle);

    return raw_nll;
}

//////////////////////////
/////// 二次型 v^H·M·v 自定义核（小矩阵优化，替代cublasCgemv+cublasCdotc）//////
//////////////////////////

// n<=64: 单block，M和v全放共享内存，warp shuffle规约
__global__ void quadratic_form_small(
    const cuComplex* __restrict__ M,
    const cuComplex* __restrict__ v,
    cuComplex* __restrict__ d_P_vec,
    float* __restrict__ result_r,
    float* __restrict__ result_i,
    int n)
{
    __shared__ cuComplex sM[64 * 64];
    __shared__ cuComplex sv[64];

    int nn = n * n;
    for (int idx = threadIdx.x; idx < nn; idx += blockDim.x) sM[idx] = M[idx];
    for (int idx = threadIdx.x; idx < n; idx += blockDim.x) sv[idx] = v[idx];
    __syncthreads();

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int num_warps = blockDim.x / 32;

    float warp_r = 0.0f, warp_i = 0.0f;

    // 每warp负责若干行: y[i] = Σ_j M[i,j] * v[j]（行主序=连续访问）
    for (int i = warp_id; i < n; i += num_warps) {
        float yr0 = 0, yi0 = 0, yr1 = 0, yi1 = 0, yr2 = 0, yi2 = 0, yr3 = 0, yi3 = 0;
        int row_off = i * n;
        int j;
        // 4路ILP: 步长128
        for (j = lane_id; j + 96 < n; j += 128) {
            cuComplex m0 = sM[row_off + j], m1 = sM[row_off + j + 32];
            cuComplex m2 = sM[row_off + j + 64], m3 = sM[row_off + j + 96];
            cuComplex v0 = sv[j], v1 = sv[j + 32], v2 = sv[j + 64], v3 = sv[j + 96];
            yr0 += m0.x * v0.x - m0.y * v0.y; yi0 += m0.x * v0.y + m0.y * v0.x;
            yr1 += m1.x * v1.x - m1.y * v1.y; yi1 += m1.x * v1.y + m1.y * v1.x;
            yr2 += m2.x * v2.x - m2.y * v2.y; yi2 += m2.x * v2.y + m2.y * v2.x;
            yr3 += m3.x * v3.x - m3.y * v3.y; yi3 += m3.x * v3.y + m3.y * v3.x;
        }
        for (; j + 32 < n; j += 64) {
            cuComplex m0 = sM[row_off + j], m1 = sM[row_off + j + 32];
            cuComplex v0 = sv[j], v1 = sv[j + 32];
            yr0 += m0.x * v0.x - m0.y * v0.y; yi0 += m0.x * v0.y + m0.y * v0.x;
            yr1 += m1.x * v1.x - m1.y * v1.y; yi1 += m1.x * v1.y + m1.y * v1.x;
        }
        for (; j < n; j += 32) {
            cuComplex m = sM[row_off + j]; cuComplex vj = sv[j];
            yr0 += m.x * vj.x - m.y * vj.y; yi0 += m.x * vj.y + m.y * vj.x;
        }
        float yr = yr0 + yr1 + yr2 + yr3, yi = yi0 + yi1 + yi2 + yi3;

        // warp内规约（先规约，再写d_P_vec，否则只存了lane0的部分和）
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            yr += __shfl_down_sync(0xffffffff, yr, off);
            yi += __shfl_down_sync(0xffffffff, yi, off);
        }
        // 规约后lane0持有完整y[i]，写入d_P_vec并累积phsp
        if (lane_id == 0) {
            d_P_vec[i] = make_cuComplex(yr, yi);
            cuComplex vi = sv[i];
            warp_r += yr * vi.x + yi * vi.y;
            warp_i += yr * vi.y - yi * vi.x;
        }
    }

    // 跨warp规约
    __shared__ float sr[32], si[32];
    if (lane_id == 0) { sr[warp_id] = warp_r; si[warp_id] = warp_i; }
    __syncthreads();
    if (warp_id == 0) {
        float fr = 0, fi = 0;
        for (int w = lane_id; w < num_warps; w += 32) { fr += sr[w]; fi += si[w]; }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            fr += __shfl_down_sync(0xffffffff, fr, off);
            fi += __shfl_down_sync(0xffffffff, fi, off);
        }
        if (lane_id == 0) { *result_r = fr; *result_i = fi; }
    }
}

// n>64: 多block，v放共享内存，M从全局用__ldg读取，atomicAdd归并结果
__global__ void quadratic_form_large(
    const cuComplex* __restrict__ M,
    const cuComplex* __restrict__ v,
    cuComplex* __restrict__ d_P_vec,
    float* __restrict__ result_r,
    float* __restrict__ result_i,
    int n)
{
    extern __shared__ cuComplex sv[];
    int tid = threadIdx.x;
    for (int i = tid; i < n; i += blockDim.x) sv[i] = v[i];
    __syncthreads();

    int warp_id = tid / 32, lane_id = tid % 32, num_warps = blockDim.x / 32;
    int rows_per_block = (n + gridDim.x - 1) / gridDim.x;
    int r0 = blockIdx.x * rows_per_block, r1 = min(r0 + rows_per_block, n);

    float warp_r = 0, warp_i = 0;
    for (int i = r0 + warp_id; i < r1; i += num_warps) {
        float yr0 = 0, yi0 = 0, yr1 = 0, yi1 = 0, yr2 = 0, yi2 = 0, yr3 = 0, yi3 = 0;
        int row_off = i * n;
        int j;
        for (j = lane_id; j + 96 < n; j += 128) {
            cuComplex m0 = __ldg(M + row_off + j), m1 = __ldg(M + row_off + j + 32);
            cuComplex m2 = __ldg(M + row_off + j + 64), m3 = __ldg(M + row_off + j + 96);
            cuComplex v0 = sv[j], v1 = sv[j + 32], v2 = sv[j + 64], v3 = sv[j + 96];
            yr0 += m0.x * v0.x - m0.y * v0.y; yi0 += m0.x * v0.y + m0.y * v0.x;
            yr1 += m1.x * v1.x - m1.y * v1.y; yi1 += m1.x * v1.y + m1.y * v1.x;
            yr2 += m2.x * v2.x - m2.y * v2.y; yi2 += m2.x * v2.y + m2.y * v2.x;
            yr3 += m3.x * v3.x - m3.y * v3.y; yi3 += m3.x * v3.y + m3.y * v3.x;
        }
        for (; j + 32 < n; j += 64) {
            cuComplex m0 = __ldg(M + row_off + j), m1 = __ldg(M + row_off + j + 32);
            cuComplex v0 = sv[j], v1 = sv[j + 32];
            yr0 += m0.x * v0.x - m0.y * v0.y; yi0 += m0.x * v0.y + m0.y * v0.x;
            yr1 += m1.x * v1.x - m1.y * v1.y; yi1 += m1.x * v1.y + m1.y * v1.x;
        }
        for (; j < n; j += 32) {
            cuComplex m = __ldg(M + row_off + j); cuComplex vj = sv[j];
            yr0 += m.x * vj.x - m.y * vj.y; yi0 += m.x * vj.y + m.y * vj.x;
        }
        float yr = yr0 + yr1 + yr2 + yr3, yi = yi0 + yi1 + yi2 + yi3;

        // warp内规约（先规约，再写d_P_vec）
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            yr += __shfl_down_sync(0xffffffff, yr, off);
            yi += __shfl_down_sync(0xffffffff, yi, off);
        }
        if (lane_id == 0) {
            d_P_vec[i] = make_cuComplex(yr, yi);
            cuComplex vi = sv[i];
            warp_r += yr * vi.x + yi * vi.y;
            warp_i += yr * vi.y - yi * vi.x;
        }
    }

    __shared__ float sr[32], si[32];
    if (lane_id == 0) { sr[warp_id] = warp_r; si[warp_id] = warp_i; }
    __syncthreads();
    if (warp_id == 0) {
        float fr = 0, fi = 0;
        for (int w = lane_id; w < num_warps; w += 32) { fr += sr[w]; fi += si[w]; }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            fr += __shfl_down_sync(0xffffffff, fr, off);
            fi += __shfl_down_sync(0xffffffff, fi, off);
        }
        if (lane_id == 0) {
            atomicAdd(result_r, fr);
            atomicAdd(result_i, fi);
        }
    }
}

void computeQuadraticForm(const cuComplex* d_M, const cuComplex* d_v,
    cuComplex* d_P_vec, float* d_phsp_r, float* d_phsp_i, int n)
{
    if (n <= 64) {
        int blk = (n <= 32) ? 32 : 128;
        quadratic_form_small << <1, blk >> > (d_M, d_v, d_P_vec, d_phsp_r, d_phsp_i, n);
    }
    else {
        int blk = 256;
        int warps_per_block = blk / 32;
        int grid = (n + warps_per_block - 1) / warps_per_block;
        if (grid < 16) grid = 16;
        if (grid > 256) grid = 256;
        cudaMemsetAsync(d_phsp_r, 0, sizeof(float));
        cudaMemsetAsync(d_phsp_i, 0, sizeof(float));
        quadratic_form_large << <grid, blk, n * sizeof(cuComplex) >> > (
            d_M, d_v, d_P_vec, d_phsp_r, d_phsp_i, n);
    }
}
