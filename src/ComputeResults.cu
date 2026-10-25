#include <ComputeResults.cuh>
#include "Backend.h"     // CTPWA_KERNEL / CTPWA_TID / CTPWA_LAUNCH / 双后端
#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>
#include <stdexcept>
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
CTPWA_KERNEL void
computeModTotalWeight(const ctComplex* __restrict__ complex_result,
    double* __restrict__ final_result,
    double* __restrict__ total_sum, int nEvents, int npolar)
{
    const int event_idx = CTPWA_TID();

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

// 修改后的部分权重核函数，同时计算干涉矩阵 + 逐事件干涉项
// d_wave_mask: 可空; [npartials] 0/1 选择子集 —— 非空时把每事件总强度
// 覆盖为 Σ_pol |Σ_{p∈S} A_p|² 并原子累加 d_selected_integral（其余输出不变）
template <int N_PARTIALS>
CTPWA_KERNEL void
computeModWithInterference(const ctComplex* __restrict__ result_matrix,
    double* __restrict__ final_result,
    double* __restrict__ interference_matrix,
    double* __restrict__ event_interference,   // [nEvents × ninterference] 原始(未归一) Σ_λ2Re(A_i A_j*)
    int* d_nSLvectors, double* total_result,
    const int* __restrict__ d_wave_mask,
    double* __restrict__ d_selected_integral,
    double* __restrict__ d_event_total,
    const int* __restrict__ d_pair_list,       // 可空; [npairs] 上三角线性索引, 只导出这些对
    int npairs_exported,
    int npartials, int nEvents, int ngls, int npolar)
{
    const int event_idx = CTPWA_TID();

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

    double selected_acc = 0.0;   // 子集总强度 Σ_pol |Σ_{p∈S} A_p|²

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

        // 子集总强度: Σ_pol |Σ_{p∈S} A_p|²（按极化累加到 selected_acc,
        // 循环结束后覆盖 d_event_total[ev] 并累加 d_selected_integral）
        if (d_wave_mask != nullptr) {
            double sr = 0.0, si = 0.0;
            for (int p = 0; p < npartials; ++p)
                if (d_wave_mask[p]) { sr += partial_real[p]; si += partial_imag[p]; }
            selected_acc += sr * sr + si * si;
        }

        // 计算干涉矩阵元素（仅上三角部分）。
        // 输出【未归一】: 不再除以 total_result_value。原因: writeResult 按事件批
        // 调用本 kernel 时, *total_result 是"实时累计积分"——逐批除会得到不同分母
        // → 干涉矩阵错误; 统一"未归一输出 + 调用方拿到总积分后一次性归一"在整表
        // 与分批两种调用下数值一致（整表单次调用由 host 补 ÷ 总积分等价复现）。
        for (int i = 0; i < npartials; i++) {
            for (int j = i; j < npartials; j++) {
                double interference_value = 0.0;

                if (i == j) {
                    double partial_intensity =
                        partial_real[i] * partial_real[i] +
                        partial_imag[i] * partial_imag[i];
                    interference_value = partial_intensity;
                }
                else {
                    // 非对角线元素：2 * Re(A_i * A_j^*)
                    float a = partial_real[i];
                    float b = partial_imag[i];
                    float c = partial_real[j];
                    float d = partial_imag[j];
                    interference_value = 2.0 * (a * c + b * d);
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

    // 将干涉矩阵累加器写入 event_interference（原始值, 未除以总积分:
    // 与 weight_<i> 同单位, 满足 weight 守恒: Σ_i|A_i|² + Σ_{i<j}2Re(A_iA_j*) = total）
    // d_pair_list 给出要导出的对(上三角线性索引); 写为紧凑布局 [npairs × nEvents]
    if (event_interference != nullptr) {
        if (d_pair_list != nullptr) {
            for (int q = 0; q < npairs_exported; ++q) {
                int k = d_pair_list[q];
                event_interference[event_idx + nEvents * q] =
                    interference_accumulator[k];
            }
        } else {
            for (int k = 0; k < ninterference; k++) {
                event_interference[event_idx + nEvents * k] =
                    interference_accumulator[k];
            }
        }
    }

    // 子集总强度覆盖每事件总值 + 子集积分累加（下游归一化自动切换）
    if (d_wave_mask != nullptr) {
        d_event_total[event_idx] = selected_acc;
        atomicAdd(d_selected_integral, selected_acc);
    }
}

// 主计算函数
// 分配失败/内核错误一律立即抛出带上下文的异常——绝不带 nullptr 继续调用 cuBLAS
// （历史上 cudaMalloc OOM 未检查 → nullptr 进 cublasZdgmm → 误导性的
// "illegal memory access"，compute-sanitizer 才定位到真实原因）。
void computeResults(const ctComplex* d_matrix, const ctComplex* d_vector,
    double* d_total_result, double* d_total_integral,
    double* d_partial_result,
    double* d_interference_matrix,
    double* d_event_interference,               // 可空; [npairs或ninterf × nEvents]
    const int* d_wave_mask,                     // 可空; 选中子集(0/1)
    double* d_selected_integral,                // 可空; 子集积分
    double* d_total_out,                        // 可空; = d_total_result, 子集时覆盖
    const int* d_pair_list, int npairs_exported,// 可空; 只导出这些对
    int* d_nSLvectors, int npartials, int nEvents, int ngls,
    int npolar)
{
    auto alloc_or_throw = [](void** p, size_t bytes, const char* what) {
        cudaError_t e = cudaMalloc(p, bytes);
        if (e == cudaSuccess) return;
        size_t free_mem = 0, total_mem = 0;
        cudaMemGetInfo(&free_mem, &total_mem);
        char buf[512];
        snprintf(buf, sizeof(buf),
                 "computeResults: cudaMalloc(%s, %.2f GB) 失败: %s"
                 "（剩余 %.1f / 共 %.1f GB）——请减小批次/检查驻留表",
                 what, (double)bytes / 1073741824.0, cudaGetErrorString(e),
                 (double)free_mem / 1073741824.0, (double)total_mem / 1073741824.0);
        throw std::runtime_error(buf);
    };
    auto cublas_or_throw = [](cublasStatus_t st, const char* what) {
        if (st == CUBLAS_STATUS_SUCCESS) return;
        char buf[256];
        snprintf(buf, sizeof(buf), "computeResults: %s 失败 (cublas err %d)",
                 what, (int)st);
        throw std::runtime_error(buf);
    };
    auto kernel_or_throw = [](const char* what) {
        cudaError_t e = cudaGetLastError();
        if (e == cudaSuccess) return;
        char buf[256];
        snprintf(buf, sizeof(buf), "computeResults: %s 失败: %s", what,
                 cudaGetErrorString(e));
        throw std::runtime_error(buf);
    };

    cublasHandle_t handle;
    cublas_or_throw(cublasCreate(&handle), "cublasCreate");

    // 分配设备内存
    ctComplex* d_complex_result = nullptr;
    alloc_or_throw(reinterpret_cast<void**>(&d_complex_result),
                   (size_t)nEvents * npolar * sizeof(ctComplex),
                   "complex_result 临时");

    // cuBLAS 矩阵向量乘法
    const ctComplex alpha = ctMake(1.0, 0.0);
    const ctComplex beta = ctMake(0.0, 0.0);

    cublas_or_throw(CUBLAS_CGEMV(handle, CUBLAS_OP_T, ngls, nEvents * npolar,
                                 &alpha, d_matrix, ngls, d_vector, 1, &beta,
                                 d_complex_result, 1),
                    "cublasCgemv(A·v)");

    // 计算总权重
    int blockSize = 128; // 减小blockSize以减少共享内存使用
    int gridSize = (nEvents + blockSize - 1) / blockSize;

    CTPWA_LAUNCH(computeModTotalWeight, gridSize, blockSize, 0,
        d_complex_result, d_total_result, d_total_integral, nEvents, npolar);
    kernel_or_throw("computeModTotalWeight");

    // 清理中间结果
    cudaFree(d_complex_result);

    ctComplex* d_result_matrix = nullptr;
    alloc_or_throw(reinterpret_cast<void**>(&d_result_matrix),
                   (size_t)ngls * nEvents * npolar * sizeof(ctComplex),
                   "result_matrix 临时（v·A 物化）");

    // 使用 cuBLAS 矩阵乘对角矩阵
    // d_matrix: (nEvents * npolar) × ngls (列主序)
    // d_vector: ngls 向量
    // 计算 d_result_matrix = d_matrix * diag(d_vector)，形状相同
    cublas_or_throw(CUBLAS_CDGMM(handle, CUBLAS_SIDE_LEFT, ngls,
                                 nEvents * npolar, d_matrix, ngls, d_vector, 1,
                                 d_result_matrix, ngls),
                    "cublasCdgmm");

    // 计算部分权重和干涉矩阵
    // 共享内存用于存储部分振幅的实部和虚部（每个部分2个double）以及干涉矩阵累加器
    // 每个线程需要 (2*npartials + ninterference) 个double，其中ninterference =
    // npartials*(npartials+1)/2 整个block需要乘以blockSize int ninterference =
    // npartials * (npartials + 1) / 2; size_t shared_mem_size = blockSize * (2
    // * npartials + ninterference) * sizeof(double);

    if (npartials <= 50) {
        CTPWA_LAUNCH(computeModWithInterference<50>, gridSize, blockSize, 0,
            d_result_matrix, d_partial_result, d_interference_matrix,
            d_event_interference, d_nSLvectors, d_total_integral,
            d_wave_mask, d_selected_integral, d_total_out,
            d_pair_list, npairs_exported,
            npartials, nEvents, ngls, npolar);
    }
    else if (npartials <= 200) {
        CTPWA_LAUNCH(computeModWithInterference<200>, gridSize, blockSize, 0,
            d_result_matrix, d_partial_result, d_interference_matrix,
            d_event_interference, d_nSLvectors, d_total_integral,
            d_wave_mask, d_selected_integral, d_total_out,
            d_pair_list, npairs_exported,
            npartials, nEvents, ngls, npolar);
    }
    else {
        CTPWA_LAUNCH(computeModWithInterference<1000>, gridSize, blockSize, 0,
            d_result_matrix, d_partial_result, d_interference_matrix,
            d_event_interference, d_nSLvectors, d_total_integral,
            d_wave_mask, d_selected_integral, d_total_out,
            d_pair_list, npairs_exported,
            npartials, nEvents, ngls, npolar);
    }
    kernel_or_throw("computeModWithInterference");

    // 同步确保所有操作完成
    cudaDeviceSynchronize();
    {
        cudaError_t se = cudaGetLastError();
        if (se != cudaSuccess) {
            char buf[256];
            snprintf(buf, sizeof(buf), "computeResults: 同步失败: %s",
                     cudaGetErrorString(se));
            throw std::runtime_error(buf);
        }
    }

    // 清理资源
    cudaFree(d_result_matrix);
    cublasDestroy(handle);
}

// ---------------------------------------------------------------------------
// DIYMC: 逐事件总强度，不分配 result_matrix / partial / interference。
// 与 computeResults 的 computeModTotalWeight 语义一致：
//   S[e,p] = Σ_i A[e,p,i] v_i,  out[e] = Σ_p |S[e,p]|²
// 矩阵布局与 A 表一致：[nEvents*npolar × ngls]，列主序。
// ---------------------------------------------------------------------------
void computeTotalIntensity(const ctComplex* d_matrix, const ctComplex* d_vector,
    double* d_total_result, int nEvents, int ngls, int npolar)
{
    if (nEvents <= 0) return;
    if ((long long)nEvents * (long long)npolar > 2147483647LL)
        throw std::runtime_error(
            "computeTotalIntensity: nEvents*npolar exceeds INT_MAX; "
            "reduce the batch size");

    auto fail = [](const char* what, cudaError_t e) {
        char buf[512];
        snprintf(buf, sizeof(buf), "computeTotalIntensity: %s 失败: %s", what,
                 cudaGetErrorString(e));
        throw std::runtime_error(buf);
    };

    cublasHandle_t handle = nullptr;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("computeTotalIntensity: cublasCreate 失败");

    ctComplex* d_complex = nullptr;
    cudaError_t e = cudaMalloc(&d_complex,
        (size_t)nEvents * (size_t)npolar * sizeof(ctComplex));
    if (e != cudaSuccess) {
        cublasDestroy(handle);
        fail("complex_result cudaMalloc", e);
    }

    double* d_sum = nullptr;
    e = cudaMalloc(&d_sum, sizeof(double));
    if (e != cudaSuccess) {
        cudaFree(d_complex);
        cublasDestroy(handle);
        fail("sum cudaMalloc", e);
    }
    cudaMemset(d_sum, 0, sizeof(double));

    const ctComplex alpha = ctMake(1.0, 0.0);
    const ctComplex beta = ctMake(0.0, 0.0);
    cublasStatus_t st = CUBLAS_CGEMV(handle, CUBLAS_OP_T, ngls,
        nEvents * npolar, &alpha, d_matrix, ngls, d_vector, 1, &beta,
        d_complex, 1);
    if (st != CUBLAS_STATUS_SUCCESS) {
        cudaFree(d_sum);
        cudaFree(d_complex);
        cublasDestroy(handle);
        char buf[256];
        snprintf(buf, sizeof(buf),
                 "computeTotalIntensity: CUBLAS_CGEMV 失败 (cublas err %d)",
                 (int)st);
        throw std::runtime_error(buf);
    }

    const int blockSize = 128;
    const int gridSize = (nEvents + blockSize - 1) / blockSize;
    CTPWA_LAUNCH(computeModTotalWeight, gridSize, blockSize, 0,
        d_complex, d_total_result, d_sum, nEvents, npolar);
    cudaDeviceSynchronize();
    e = cudaGetLastError();

    cudaFree(d_sum);
    cudaFree(d_complex);
    cublasDestroy(handle);
    if (e != cudaSuccess) fail("computeModTotalWeight", e);
}

