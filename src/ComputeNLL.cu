#include <complex>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
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
    simpleMagnitudeSum<<<gridSize, blockSize, shared_mem_size>>>(
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

        // 计算负对数似然
        const double epsilon = 1e-10;
        sdata[tid] = -log(fmax(group_sum / phsp_factor, epsilon));
    } else {
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
        computeNLLKernelWeighted<<<gridSize, blockSize, shared_mem_size>>>(
            d_S, d_weights, d_Q, d_final_result, nlength, npolar, phsp_factor);
    } else {
        // 使用原始核函数（向后兼容）
        computeNLLKernel<<<gridSize, blockSize, shared_mem_size>>>(
            d_S, d_Q, d_final_result, nlength, npolar, phsp_factor);
    }

    // 检查核函数执行错误
    cuda_error = cudaGetLastError();
    if (cuda_error != cudaSuccess) {
        printf("Kernel error: %s\n", cudaGetErrorString(cuda_error));
    }
}

// // 合并的核函数：计算模平方、分组求和、对数计算和最终求和
// __global__ void computeNLLKernel(
//     const cuComplex *__restrict__ vector,
//     cuComplex *__restrict__ group_sums,
//     double *__restrict__ total_sum,
//     int nlength,
//     int npolar,
//     double phsp_factor = 1.0)
// {
//     extern __shared__ double sdata[];

//     int tid = threadIdx.x;
//     int group_idx = blockIdx.x * blockDim.x + tid;
//     int total_groups = nlength / npolar;

//     // 每个线程处理一个组
//     double group_sum = 0.0;
//     if (group_idx < total_groups)
//     {
//         // 计算当前组的模平方和
//         int start_idx = group_idx * npolar;
//         for (int i = 0; i < npolar; i++)
//         {
//             int idx = start_idx + i;
//             if (idx < nlength)
//             {
//                 cuComplex val = vector[idx];
//                 group_sum += (val.x * val.x + val.y * val.y);
//             }
//         }

//         // 存储组和到全局内存（如果提供了group_sums指针）
//         if (group_sums != nullptr)
//         {
//             group_sums[group_idx] = make_cuComplex(1.0 / group_sum, 0.0);
//         }

//         // 计算负对数似然
//         const double epsilon = 1e-10;
//         sdata[tid] = -log(fmax(group_sum / phsp_factor, epsilon));
//     }
//     else
//     {
//         sdata[tid] = 0.0;
//     }

//     __syncthreads();

//     // 块内归约：计算当前块的log总和
//     for (int s = blockDim.x / 2; s > 0; s >>= 1)
//     {
//         if (tid < s)
//         {
//             sdata[tid] += sdata[tid + s];
//         }
//         __syncthreads();
//     }

//     // 将块内总和累加到全局总和
//     if (tid == 0)
//     {
//         // 使用原子操作将块结果加到最终结果
// #if __CUDA_ARCH__ >= 600
//         atomicAdd(total_sum, sdata[0]);
// #else
//         // 兼容旧架构的double原子操作实现
//         unsigned long long *total_sum_ull = (unsigned long long *)total_sum;
//         unsigned long long old, new_val;
//         old = *total_sum_ull;
//         do
//         {
//             new_val = __double_as_longlong(__longlong_as_double(old) +
//             sdata[0]);
//         } while (atomicCAS(total_sum_ull, old, new_val) != old);
// #endif
//     }
// }

// // 优化的NLL计算函数，使用cuBLAS和合并核函数
// void computeNll(
//     const cuComplex *d_matrix,
//     const cuComplex *d_vector,
//     cuComplex *d_S,
//     cuComplex *d_Q,
//     double *d_final_result,
//     int nlength, int ngls,
//     int npolar,
//     double phsp_factor = 1.0)
// {
//     cublasHandle_t handle;
//     cublasCreate(&handle);

//     // 步骤1: 使用cuBLAS计算矩阵-向量乘法 d_S = matrix * vector
//     const cuComplex alpha = make_cuComplex(1.0, 0.0);
//     const cuComplex beta = make_cuComplex(0.0, 0.0);

//     // 使用非转置操作（根据您的测试结果）
//     cublasCgemv(handle, CUBLAS_OP_N,
//                 nlength, ngls, // 矩阵维度: nlength × ngls
//                 &alpha,
//                 d_matrix, nlength, // lda = nlength
//                 d_vector, 1,
//                 &beta,
//                 d_S, 1);

//     cublasDestroy(handle);

//     // 检查cuBLAS错误
//     cudaError_t cuda_error = cudaGetLastError();
//     if (cuda_error != cudaSuccess)
//     {
//         printf("cuBLAS error: %s\n", cudaGetErrorString(cuda_error));
//         return;
//     }

//     // 步骤2: 使用合并的核函数计算NLL
//     int total_groups = nlength / npolar;

//     // 设置线程块和网格大小
//     int blockSize = 256;
//     int gridSize = (total_groups + blockSize - 1) / blockSize;
//     size_t shared_mem_size = blockSize * sizeof(double);

//     // 初始化最终结果
//     double h_zero = 0.0;
//     cudaMemcpy(d_final_result, &h_zero, sizeof(double),
//     cudaMemcpyHostToDevice);

//     // 调用合并核函数
//     computeNLLKernel<<<gridSize, blockSize, shared_mem_size>>>(d_S, d_Q,
//     d_final_result, nlength, npolar, phsp_factor);

//     // 检查核函数执行错误
//     cuda_error = cudaGetLastError();
//     if (cuda_error != cudaSuccess)
//     {
//         printf("Kernel error: %s\n", cudaGetErrorString(cuda_error));
//     }
// }

// 核函数：计算复数模平方并按 npolar 分组求和
// __global__ void computeModSquareAndReduce(
//     const cuComplex *__restrict__ complex_result,
//     double *__restrict__ final_result,
//     int nEvents, int npolar)
// {
//     int event_idx = blockIdx.x * blockDim.x + threadIdx.x;

//     if (event_idx >= nEvents)
//         return;

//     double sum = 0.0;

//     // 每个线程处理一个 event，累加对应的 npolar 个元素的模平方
//     for (int polar_idx = 0; polar_idx < npolar; polar_idx++)
//     {
//         int global_idx = event_idx * npolar + polar_idx;
//         cuComplex val = complex_result[global_idx];
//         double mod_square = val.x * val.x + val.y * val.y;
//         sum += mod_square;
//     }

//     final_result[event_idx] = sum;
// }

// void computeWeightResult(
//     const cuComplex *d_matrix,
//     const cuComplex *d_vector,
//     double *d_final_result,
//     int nEvents, int ngls, int npolar)
// {
//     cublasHandle_t handle;
//     cublasCreate(&handle);

//     // 分配设备内存
//     cuComplex *d_complex_result = nullptr;
//     cudaMalloc(&d_complex_result, nEvents * npolar * sizeof(cuComplex));

//     // cuBLAS 矩阵向量乘法参数
//     const cuComplex alpha = make_cuComplex(1.0, 0.0);
//     const cuComplex beta = make_cuComplex(0.0, 0.0);

//     // 执行矩阵向量乘法: complex_result = matrix * vector
//     // 注意：cuBLAS 使用列主序，所以我们需要适当调整参数
//     // 如果矩阵是行主序的 M x N 矩阵，在 cuBLAS 中相当于列主序的 N x M
//     矩阵的转置 cublasCgemv(handle,
//                 CUBLAS_OP_N,      // 如果矩阵是行主序，可能需要使用
//                 CUBLAS_OP_T nEvents * npolar, // 矩阵行数 ngls, // 矩阵列数
//                 &alpha,
//                 d_matrix, nEvents * npolar,
//                 d_vector, 1,
//                 &beta,
//                 d_complex_result, 1);

//     // 检查 cuBLAS 调用是否成功
//     cudaError_t cuda_error = cudaGetLastError();
//     if (cuda_error != cudaSuccess)
//     {
//         printf("cuBLAS error: %s\n", cudaGetErrorString(cuda_error));
//     }

//     // 对于较小的 npolar，使用简单版本
//     int blockSize = 256;
//     int gridSize = (nEvents + blockSize - 1) / blockSize;

//     computeModSquareAndReduce<<<gridSize, blockSize>>>(
//         d_complex_result, d_final_result, nEvents, npolar);

//     // 检查核函数执行是否成功
//     cuda_error = cudaGetLastError();
//     if (cuda_error != cudaSuccess)
//     {
//         printf("Kernel error: %s\n", cudaGetErrorString(cuda_error));
//     }

//     // 同步确保所有操作完成
//     cudaDeviceSynchronize();

//     // 清理资源
//     cudaFree(d_complex_result);
//     cublasDestroy(handle);
// }
