#include <ComputeResults.cuh>
#include "ComplexType.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>
// #include <iostream>
// #include <complex>
#include <cublas_v2.h>

__device__ __host__ int getInterferenceIndex(int i, int j, int npartials)
{
    if (i > j) {
        int temp = i;
        i = j;
        j = temp;
    }
    // 上三角矩阵的线性索引（包括对角线）
    return i * npartials - i * (i - 1) / 2 + (j - i);
}

// 核函数：计算复数模平方并按 npolar 分组求和，同时计算总和
__global__ void
computeModTotalWeight(const ctComplex* __restrict__ complex_result,
    double* __restrict__ final_result,
    double* __restrict__ total_sum, int nEvents, int npolar)
{
    int event_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (event_idx >= nEvents)
        return;

    double sum = 0.0;

    // 每个线程处理一个 event，累加对应的 npolar 个元素的模平方
    for (int polar_idx = 0; polar_idx < npolar; polar_idx++) {
        int global_idx = event_idx * npolar + polar_idx;
        ctComplex val = complex_result[global_idx];
        double mod_square = val.x * val.x + val.y * val.y;
        sum += mod_square;
    }

    final_result[event_idx] = sum;

    // 使用原子操作累加总和
    atomicAdd(total_sum, sum);
}

// 修改后的部分权重核函数，同时计算干涉矩阵
template <int N_PARTIALS>
__global__ void
computeModWithInterference(const ctComplex* __restrict__ result_matrix,
    double* __restrict__ final_result,
    double* __restrict__ interference_matrix,
    // double *__restrict__ event_interference,
    int* d_nSLvectors, double* total_result,
    int npartials, int nEvents, int ngls, int npolar)
{
    int event_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (event_idx >= nEvents)
        return;

    int ninterference = npartials * (npartials + 1) / 2;
    double total_result_value = *total_result;

    // 如果总权重为零，跳过计算
    if (total_result_value <= 0.0) {
        return;
    }

    // 局部数组：假设npartials <= 200，否则需要调整MAX_PARTIALS
    const int MAX_PARTIALS = N_PARTIALS;
    const int MAX_INTERFERENCE = MAX_PARTIALS * (MAX_PARTIALS + 1) / 2; // 5050

    if (npartials > MAX_PARTIALS) {
        printf("Error: npartials=%d exceeds maximum supported value (%d).\n",
            npartials, MAX_PARTIALS);
        return;
    }

    // 使用局部数组，避免设备端malloc
    double partial_real[MAX_PARTIALS];
    double partial_imag[MAX_PARTIALS];
    double interference_accumulator[MAX_INTERFERENCE];

    // 初始化干涉累加器为0
    for (int k = 0; k < ninterference; k++) {
        interference_accumulator[k] = 0.0;
    }

    for (int polar_idx = 0; polar_idx < npolar; polar_idx++) {
        // 初始化部分振幅为0
        for (int p = 0; p < npartials; p++) {
            partial_real[p] = 0.0;
            partial_imag[p] = 0.0;
        }

        int sltotal = 0;

        // 计算每个部分的振幅
        for (int p = 0; p < npartials; p++) {
            for (int s = 0; s < d_nSLvectors[p]; s++) {
                ctComplex val = result_matrix[event_idx * npolar * ngls +
                    polar_idx * ngls + (sltotal + s)];
                partial_real[p] += val.x;
                partial_imag[p] += val.y;
            }
            sltotal += d_nSLvectors[p];
        }

        // 计算每个部分的模方并累加到final_result
        for (int p = 0; p < npartials; p++) {
            double partial_intensity = partial_real[p] * partial_real[p] +
                partial_imag[p] * partial_imag[p];
            final_result[p * nEvents + event_idx] += partial_intensity;
        }

        // 计算干涉矩阵元素（仅上三角部分）
        for (int i = 0; i < npartials; i++) {
            for (int j = i; j < npartials; j++) {
                double interference_value = 0.0;

                if (i == j) {
                    double partial_intensity =
                        partial_real[i] * partial_real[i] +
                        partial_imag[i] * partial_imag[i];
                    interference_value = partial_intensity / total_result_value;
                }
                else {
                    // 非对角线元素：2 * Re(A_i * A_j^*) / total_result
                    float a = partial_real[i];
                    float b = partial_imag[i];
                    float c = partial_real[j];
                    float d = partial_imag[j];
                    interference_value =
                        2.0 * (a * c + b * d) / total_result_value;
                }

                // 计算干涉矩阵中的索引（上三角存储）
                int idx = getInterferenceIndex(i, j, npartials);

                // 原子累加到全局干涉矩阵（对所有事件和极化求和）
                atomicAdd(&interference_matrix[idx], interference_value);

                // 累加到线程本地干涉矩阵累加器（仅对当前事件的极化求和）
                interference_accumulator[idx] += interference_value;
            }
        }
    }

    // // 将干涉矩阵累加器写入event_interference
    // for (int k = 0; k < ninterference; k++)
    // {
    //     event_interference[event_idx + nEvents * k] = total_result_value *
    //     interference_accumulator[k];
    // }
}

// 主计算函数
void computeResults(const ctComplex* d_matrix, const ctComplex* d_vector,
    double* d_total_result, double* d_total_integral,
    double* d_partial_result,
    // double *d_partial_sums,
    double* d_interference_matrix,
    // double *d_event_interference,
    int* d_nSLvectors, int npartials, int nEvents, int ngls,
    int npolar)
{
    cublasHandle_t handle;
    cublasCreate(&handle);

    // 分配设备内存
    ctComplex* d_complex_result = nullptr;
    cudaMalloc(&d_complex_result, nEvents * npolar * sizeof(ctComplex));

    // cuBLAS 矩阵向量乘法
    const ctComplex alpha = ctMake(1.0, 0.0);
    const ctComplex beta = ctMake(0.0, 0.0);

    CUBLAS_CGEMV(handle, CUBLAS_OP_T, ngls, nEvents * npolar, &alpha, d_matrix,
        ngls, d_vector, 1, &beta, d_complex_result, 1);

    // 检查 cuBLAS 调用
    cudaError_t cuda_error = cudaGetLastError();
    if (cuda_error != cudaSuccess) {
        printf("cuBLAS error: %s\n", cudaGetErrorString(cuda_error));
    }

    // 计算总权重
    int blockSize = 128; // 减小blockSize以减少共享内存使用
    int gridSize = (nEvents + blockSize - 1) / blockSize;

    computeModTotalWeight << <gridSize, blockSize >> > (
        d_complex_result, d_total_result, d_total_integral, nEvents, npolar);

    // 清理中间结果
    cudaFree(d_complex_result);

    ctComplex* d_result_matrix = nullptr;
    cudaMalloc(&d_result_matrix, ngls * nEvents * npolar * sizeof(ctComplex));

    // 使用 cuBLAS 矩阵乘对角矩阵
    // d_matrix: (nEvents * npolar) × ngls (列主序)
    // d_vector: ngls 向量
    // 计算 d_result_matrix = d_matrix * diag(d_vector)，形状相同
    CUBLAS_CDGMM(handle, CUBLAS_SIDE_LEFT, ngls, nEvents * npolar, d_matrix,
        ngls, d_vector, 1, d_result_matrix,
        ngls);

    // 计算部分权重和干涉矩阵
    // 共享内存用于存储部分振幅的实部和虚部（每个部分2个double）以及干涉矩阵累加器
    // 每个线程需要 (2*npartials + ninterference) 个double，其中ninterference =
    // npartials*(npartials+1)/2 整个block需要乘以blockSize int ninterference =
    // npartials * (npartials + 1) / 2; size_t shared_mem_size = blockSize * (2
    // * npartials + ninterference) * sizeof(double);

    if (npartials <= 50) {
        // computeModWithInterference<50><<<gridSize,
        // blockSize>>>(d_result_matrix, d_partial_result,
        // d_interference_matrix, d_event_interference, d_nSLvectors,
        // d_total_integral, npartials, nEvents, npolar);
        computeModWithInterference<50> << <gridSize, blockSize >> > (
            d_result_matrix, d_partial_result, d_interference_matrix,
            d_nSLvectors, d_total_integral, npartials, nEvents, ngls, npolar);
    }
    else if (npartials <= 200) {
        // computeModWithInterference<200><<<gridSize,
        // blockSize>>>(d_result_matrix, d_partial_result,
        // d_interference_matrix, d_event_interference, d_nSLvectors,
        // d_total_integral, npartials, nEvents, npolar);
        computeModWithInterference<200> << <gridSize, blockSize >> > (
            d_result_matrix, d_partial_result, d_interference_matrix,
            d_nSLvectors, d_total_integral, npartials, nEvents, ngls, npolar);
    }
    else {
        // computeModWithInterference<1000><<<gridSize,
        // blockSize>>>(d_result_matrix, d_partial_result,
        // d_interference_matrix, d_event_interference, d_nSLvectors,
        // d_total_integral, npartials, nEvents, npolar);
        computeModWithInterference<1000> << <gridSize, blockSize >> > (
            d_result_matrix, d_partial_result, d_interference_matrix,
            d_nSLvectors, d_total_integral, npartials, nEvents, ngls, npolar);
    }
    // computeModWithInterference<<<gridSize, blockSize,
    // shared_mem_size>>>(d_result_matrix, d_partial_result, d_partial_sums,
    // d_interference_matrix, d_event_interference, d_nSLvectors,
    // d_total_integral, npartials, nEvents, npolar);

    // 检查核函数执行
    cuda_error = cudaGetLastError();
    if (cuda_error != cudaSuccess) {
        printf("Kernel error: %s\n", cudaGetErrorString(cuda_error));
    }

    // 同步确保所有操作完成
    cudaDeviceSynchronize();

    // 清理资源
    cudaFree(d_result_matrix);
    cublasDestroy(handle);
}

