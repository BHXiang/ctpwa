#include <complex>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <iostream>
#include <stdio.h>
#include <vector>

// 最简单的合并版本
__global__ void simpleMagnitudeSum(const cuComplex* __restrict__ vector,
    double* __restrict__ final_result, int M) {
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;

    double sum = 0.0;

    // 处理当前线程的元素
    if (i < M) {
        cuComplex val = vector[i];
        sum = val.x * val.x + val.y * val.y;
    }

    // printf("Thread %d processed index %d with partial sum %f\n", tid, i,
    // sum);

    // 使用共享内存进行归约
    extern __shared__ double sdata[];
    sdata[tid] = sum;
    __syncthreads();

    // 标准归约
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 原子加最终结果
    if (tid == 0) {
        atomicAdd(final_result, sdata[0]);
    }
}

// 使用合并核函数的优化版本
void computePHSPfactor(const cuComplex* d_matrix, const cuComplex* d_vector,
    cuComplex* d_B, double* d_final_result, int M, int N) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    // 步骤1: 使用cuBLAS计算矩阵-向量乘法
    const cuComplex alpha = make_cuComplex(1.0, 0.0);
    const cuComplex beta = make_cuComplex(0.0, 0.0);

    // 根据矩阵存储顺序选择合适的操作
    // 如果矩阵是行主序的M×N，使用转置操作
    cublasCgemv(handle, CUBLAS_OP_N, M, N, // 矩阵维度
        &alpha, d_matrix, M,       // lda = N
        d_vector, 1, &beta, d_B, 1);

    cublasDestroy(handle);

    // 步骤2: 使用合并的核函数计算模方并求和
    int blockSize = 256;
    int gridSize = min(65535, (M + blockSize - 1) / blockSize);
    size_t shared_mem_size = blockSize * sizeof(double);

    // 初始化最终结果
    double h_zero = 0.0;
    cudaMemcpy(d_final_result, &h_zero, sizeof(double), cudaMemcpyHostToDevice);

    // 选择不同的合并核函数：

    // 版本1: 简单的合并版本（推荐）
    simpleMagnitudeSum << <gridSize, blockSize, shared_mem_size >> > (
        d_B, d_final_result, M);
}

//////////////////////////
/////// 似然值计算 ///////
//////////////////////////

// 合并的核函数：计算模平方、分组求和、对数计算和最终求和（带权重）
__global__ void computeNLLKernelWeighted(const cuComplex* __restrict__ vector,
    const double* __restrict__ weights,
    cuComplex* __restrict__ group_sums,
    double* __restrict__ total_sum,
    int nlength, int npolar,
    double phsp_factor = 1.0) {
    extern __shared__ double sdata[];

    int tid = threadIdx.x;
    int group_idx = blockIdx.x * blockDim.x + tid;
    int total_groups = nlength / npolar;

    // 每个线程处理一个组
    double group_contribution = 0.0;
    if (group_idx < total_groups) {
        // 计算当前组的模平方和
        int start_idx = group_idx * npolar;
        double group_sum = 0.0;
        for (int i = 0; i < npolar; i++) {
            int idx = start_idx + i;
            if (idx < nlength) {
                cuComplex val = vector[idx];
                group_sum += (val.x * val.x + val.y * val.y);
            }
        }

        // 计算权重因子
        double weight = 1.0;
        if (weights != nullptr) {
            weight = weights[group_idx];
        }

        // 存储组和的倒数（带权重）到全局内存
        if (group_sums != nullptr) {
            // 存储 w_g / U_g，用于梯度计算
            group_sums[group_idx] = make_cuComplex(weight / group_sum, 0.0);
        }

        // 计算带权重的负对数似然
        const double epsilon = 1e-10;
        group_contribution =
            weight * (-log(fmax(group_sum / phsp_factor, epsilon)));
    }

    sdata[tid] = group_contribution;
    __syncthreads();

    // 块内归约
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 将块内总和累加到全局总和
    if (tid == 0) {
        // 使用原子操作将块结果加到最终结果
#if __CUDA_ARCH__ >= 600
        atomicAdd(total_sum, sdata[0]);
#else
        unsigned long long* total_sum_ull = (unsigned long long*)total_sum;
        unsigned long long old, new_val;
        old = *total_sum_ull;
        do {
            new_val =
                __double_as_longlong(__longlong_as_double(old) + sdata[0]);
        } while (atomicCAS(total_sum_ull, old, new_val) != old);
#endif
    }
}

// 合并的核函数：计算模平方、分组求和、对数计算和最终求和
__global__ void computeNLLKernel(const cuComplex* __restrict__ vector,
    cuComplex* __restrict__ group_sums,
    double* __restrict__ total_sum, int nlength,
    int npolar, double phsp_factor = 1.0) {
    extern __shared__ double sdata[];

    int tid = threadIdx.x;
    int group_idx = blockIdx.x * blockDim.x + tid;
    int total_groups = nlength / npolar;

    // 每个线程处理一个组
    double group_sum = 0.0;
    if (group_idx < total_groups) {
        // 计算当前组的模平方和
        int start_idx = group_idx * npolar;
        for (int i = 0; i < npolar; i++) {
            int idx = start_idx + i;
            if (idx < nlength) {
                cuComplex val = vector[idx];
                group_sum += (val.x * val.x + val.y * val.y);
            }
        }

        // 存储组和到全局内存（如果提供了group_sums指针）
        if (group_sums != nullptr) {
            group_sums[group_idx] = make_cuComplex(1.0 / group_sum, 0.0);
        }

        // printf("Group %d: sum = %f\n", group_idx, group_sum);

        // 计算负对数似然
        const double epsilon = 1e-10;
        sdata[tid] = -log(fmax(group_sum / phsp_factor, epsilon));
    }
    else {
        sdata[tid] = 0.0;
    }

    __syncthreads();

    // 块内归约：计算当前块的log总和
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 将块内总和累加到全局总和
    if (tid == 0) {
        // 使用原子操作将块结果加到最终结果
#if __CUDA_ARCH__ >= 600
        atomicAdd(total_sum, sdata[0]);
#else
        // 兼容旧架构的double原子操作实现
        unsigned long long* total_sum_ull = (unsigned long long*)total_sum;
        unsigned long long old, new_val;
        old = *total_sum_ull;
        do {
            new_val =
                __double_as_longlong(__longlong_as_double(old) + sdata[0]);
        } while (atomicCAS(total_sum_ull, old, new_val) != old);
#endif
    }
}

// 统一的NLL计算函数，支持权重
void computeNll(const cuComplex* d_matrix, const cuComplex* d_vector,
    const double* d_weights, cuComplex* d_S, cuComplex* d_Q,
    double* d_final_result, int nlength, int ngls, int npolar,
    double phsp_factor = 1.0) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    // 步骤1: 使用cuBLAS计算矩阵-向量乘法 d_S = matrix * vector
    const cuComplex alpha = make_cuComplex(1.0, 0.0);
    const cuComplex beta = make_cuComplex(0.0, 0.0);

    cublasCgemv(handle, CUBLAS_OP_N, nlength, ngls, &alpha, d_matrix, nlength,
        d_vector, 1, &beta, d_S, 1);

    cublasDestroy(handle);

    // 检查cuBLAS错误
    cudaError_t cuda_error = cudaGetLastError();
    if (cuda_error != cudaSuccess) {
        printf("cuBLAS error: %s\n", cudaGetErrorString(cuda_error));
        return;
    }

    // 步骤2: 使用合并的核函数计算NLL
    int total_groups = nlength / npolar;

    // 设置线程块和网格大小
    int blockSize = 256;
    int gridSize = (total_groups + blockSize - 1) / blockSize;
    size_t shared_mem_size = blockSize * sizeof(double);

    // 初始化最终结果
    double h_zero = 0.0;
    cudaMemcpy(d_final_result, &h_zero, sizeof(double), cudaMemcpyHostToDevice);

    // 调用对应的核函数
    if (d_weights != nullptr) {
        // 使用带权重的核函数
        computeNLLKernelWeighted << <gridSize, blockSize, shared_mem_size >> > (
            d_S, d_weights, d_Q, d_final_result, nlength, npolar, phsp_factor);
    }
    else {
        // 使用原始核函数（向后兼容）
        computeNLLKernel << <gridSize, blockSize, shared_mem_size >> > (
            d_S, d_Q, d_final_result, nlength, npolar, phsp_factor);
    }

    // 检查核函数执行错误
    cuda_error = cudaGetLastError();
    if (cuda_error != cudaSuccess) {
        printf("Kernel error: %s\n", cudaGetErrorString(cuda_error));
    }
}

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


// -----------------------------------------------------------------------------
// 优化后的主函数
// -----------------------------------------------------------------------------
double computeFactorNLL(const cuComplex* d_amp, const cuComplex* d_vector,
    cuComplex* d_grad_out,
    int nEvents, int n_polar, int n_amplitudes,
    const double* d_weights)
{
    const int nTotal = nEvents * n_polar;
    constexpr int kBlockSize = 256;

    // ----- 创建 cuBLAS 句柄 -----
    cublasHandle_t handle;
    cublasCreate(&handle);

    // ----- 分配临时缓冲区 -----
    cuComplex* d_S;
    cudaMalloc(&d_S, nTotal * sizeof(cuComplex));     // 复用为 S 和 w
    double* d_nll;
    cudaMalloc(&d_nll, sizeof(double));               // 全局 NLL 累加器

    // ----- 第一大步：S = A * v -----
    // A: nTotal×n_amplitudes行主序 → cuBLAS列主序视作 n_amplitudes×nTotal
    // CUBLAS_OP_T: y = A_col^T * v_col = (nTotal×n_amplitudes) * (n_amplitudes×1)
    {
        cuComplex alpha = make_cuComplex(1.0f, 0.0f);
        cuComplex beta = make_cuComplex(0.0f, 0.0f);
        cublasCgemv(handle, CUBLAS_OP_T,
            n_amplitudes, nTotal,           // m=n_amplitudes, n=nTotal
            &alpha, d_amp, n_amplitudes,
            d_vector, 1,
            &beta, d_S, 1);
    }

    // ----- 第二大步：计算 factor、写入权重 w、同时规约得到 NLL -----
    cudaMemset(d_nll, 0, sizeof(double));
    int gridBlocks = (nEvents + kBlockSize - 1) / kBlockSize;
    computeFactorsAndWeightsKernel << <gridBlocks, kBlockSize >> > (
        d_S, d_nll, d_weights, nEvents, n_polar);

    // 拷回 NLL
    double raw_nll;
    cudaMemcpy(&raw_nll, d_nll, sizeof(double), cudaMemcpyDeviceToHost);

    // NaN/Inf检测：若NLL异常，清零梯度并返回大值，防止优化器发散
    // if (isnan(raw_nll) || isinf(raw_nll)) {
    //     fprintf(stderr, "WARNING: computeFactorNLL produced %s, resetting\n",
    //             isnan(raw_nll) ? "NaN" : "Inf");
    //     cudaMemset(d_grad_out, 0, n_amplitudes * sizeof(cuComplex));
    //     cudaFree(d_S);
    //     cudaFree(d_nll);
    //     cublasDestroy(handle);
    //     return 1e30;  // 返回大值让优化器远离此区域
    // }

    // ----- 第三大步：梯度 grad = - (A_row)^H * w -----
    // A_col (n_amplitudes×nTotal): CUBLAS_OP_N → A_col * conj(w) → conj → grad
    {
        int gridConj = (nTotal + kBlockSize - 1) / kBlockSize;
        conjugateKernel << <gridConj, kBlockSize >> > (d_S, nTotal);

        cuComplex alpha = make_cuComplex(-1.0f, 0.0f);
        cuComplex beta = make_cuComplex(0.0f, 0.0f);
        cublasCgemv(handle, CUBLAS_OP_N,
            n_amplitudes, nTotal,
            &alpha, d_amp, n_amplitudes,
            d_S, 1,
            &beta, d_grad_out, 1);
    }

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
// __global__ void computeFactorsKernel(
//     const cuComplex* __restrict__ d_S,
//     cuComplex* __restrict__ d_Q,
//     double* __restrict__ d_logf,
//     int nEvents, int n_polar) {
//     int evt = blockIdx.x * blockDim.x + threadIdx.x;
//     if (evt >= nEvents) return;

//     const cuComplex* S_evt = d_S + evt * n_polar;
//     double factor = 0.0;
// #pragma unroll
//     for (int pol = 0; pol < n_polar; ++pol) {
//         cuComplex s = S_evt[pol];
//         factor += s.x * s.x + s.y * s.y;
//     }
//     d_Q[evt] = make_cuComplex((float)(1.0 / factor), 0.0f);
//     d_logf[evt] = -log(fmax(factor, 1e-10));
// }

// // 梯度累加kernel: 2D grid=(n_amplitudes, nBlocksPerAmp), 块内共享内存规约
// // 缺点: 每block处理少量event时atomicAdd开销可忽略，无需CUB分段规约
// //        (CUB分段规约需分离实/虚部+额外中间数组，对此场景没有优势)
// __global__ void accumulateGradientKernel(
//     const cuComplex* __restrict__ A,
//     const cuComplex* __restrict__ S,
//     const cuComplex* __restrict__ d_Q,
//     cuComplex* __restrict__ d_grad,
//     int nEvents, int n_polar, int n_amplitudes) {
//     int amp = blockIdx.x;
//     if (amp >= n_amplitudes) return;

//     int evt = blockIdx.y * blockDim.x + threadIdx.x;
//     const int stride_evt = n_polar * n_amplitudes;
//     cuComplex partial = make_cuComplex(0, 0);

//     if (evt < nEvents) {
//         const cuComplex* A_evt = A + evt * stride_evt;
//         const cuComplex* S_evt = S + evt * n_polar;
//         double inv_factor = d_Q[evt].x;

//         cuComplex sum = make_cuComplex(0, 0);
// #pragma unroll
//         for (int pol = 0; pol < n_polar; ++pol) {
//             cuComplex a_val = A_evt[pol * n_amplitudes + amp];
//             cuComplex s_val = S_evt[pol];
//             sum.x += a_val.x * s_val.x + a_val.y * s_val.y;
//             sum.y += a_val.x * s_val.y - a_val.y * s_val.x;
//         }
//         partial.x = sum.x * inv_factor;
//         partial.y = sum.y * inv_factor;
//     }

//     extern __shared__ cuComplex s_partial[];
//     s_partial[threadIdx.x] = partial;
//     __syncthreads();

//     for (int s = blockDim.x / 2; s > 0; s >>= 1) {
//         if (threadIdx.x < s) {
//             s_partial[threadIdx.x].x += s_partial[threadIdx.x + s].x;
//             s_partial[threadIdx.x].y += s_partial[threadIdx.x + s].y;
//         }
//         __syncthreads();
//     }

//     if (threadIdx.x == 0) {
//         atomicAdd(&(d_grad[amp].x), -s_partial[0].x);
//         atomicAdd(&(d_grad[amp].y), -s_partial[0].y);
//     }
// }

// double computeFactorNLL(const cuComplex* d_amp, const cuComplex* d_vector,
//     cuComplex* d_grad_out,
//     int nEvents, int n_polar, int n_amplitudes) {
//     const int nTotal = nEvents * n_polar;
//     constexpr int kBlockSize = 256;

//     // Step 1: S = A * v  单次cublasCgemv
//     cuComplex* d_S;
//     cudaMalloc(&d_S, nTotal * sizeof(cuComplex));
//     {
//         cublasHandle_t handle;
//         cublasCreate(&handle);
//         const cuComplex alpha = make_cuComplex(1.0f, 0.0f);
//         const cuComplex beta = make_cuComplex(0.0f, 0.0f);
//         cublasCgemv(handle, CUBLAS_OP_T,
//             n_amplitudes, nTotal,
//             &alpha, d_amp, n_amplitudes,
//             d_vector, 1,
//             &beta, d_S, 1);
//         cublasDestroy(handle);
//     }

//     // Step 2: 简单kernel算|S|²→factor, 输出1/factor和-log(factor)
//     cuComplex* d_Q;
//     cudaMalloc(&d_Q, nEvents * sizeof(cuComplex));
//     double* d_logf;
//     cudaMalloc(&d_logf, nEvents * sizeof(double));
//     {
//         int gridSize = (nEvents + kBlockSize - 1) / kBlockSize;
//         computeFactorsKernel<<<gridSize, kBlockSize>>>(
//             d_S, d_Q, d_logf, nEvents, n_polar);
//     }

//     // Step 3: CUB DeviceReduce::Sum 多级规约求和 → NLL
//     double* d_nll;
//     cudaMalloc(&d_nll, sizeof(double));
//     {
//         void* d_temp = nullptr;
//         size_t temp_bytes = 0;
//         cub::DeviceReduce::Sum(d_temp, temp_bytes, d_logf, d_nll, nEvents);
//         cudaMalloc(&d_temp, temp_bytes);
//         cub::DeviceReduce::Sum(d_temp, temp_bytes, d_logf, d_nll, nEvents);
//         cudaFree(d_temp);
//     }
//     double raw_nll;
//     cudaMemcpy(&raw_nll, d_nll, sizeof(double), cudaMemcpyDeviceToHost);
//     cudaFree(d_nll);
//     cudaFree(d_logf);

//     // Step 4: 梯度 kernel — 2D grid: (n_amplitudes, nBlocksPerAmp)
//     cudaMemset(d_grad_out, 0, n_amplitudes * sizeof(cuComplex));
//     {
//         int nBlocksPerAmp = (nEvents + kBlockSize - 1) / kBlockSize;
//         dim3 grid(n_amplitudes, nBlocksPerAmp);
//         size_t shm = kBlockSize * sizeof(cuComplex);
//         accumulateGradientKernel<<<grid, kBlockSize, shm>>>(
//             d_amp, d_S, d_Q, d_grad_out, nEvents, n_polar, n_amplitudes);
//     }

//     cudaFree(d_S);
//     cudaFree(d_Q);

//     return raw_nll;
// }

