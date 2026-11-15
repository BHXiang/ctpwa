#include <ComputeResults.cuh>
#include "Backend.h"     // CTPWA_KERNEL / CTPWA_TID / CTPWA_LAUNCH / 双后端
#include <cuda_runtime.h>
#include <stdio.h>
#include <string>
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
//
// 两个模板参数都直接决定**每线程局部内存**（ptxas 的 STACK 帧）。CUDA 的局部
// 内存按"最大常驻线程数 × 每线程字节"整块预留，与 grid 大小无关：
//   V100: 2048 线程/SM × 80 SM = 163 840 线程
//   ⇒ <50,*>: ~11 KB/线程 → ~1.7 GiB；<200,*>: ~164 KB/线程 → ~25 GiB；
//     <1000,*>: ~4 MB/线程 → 直接超出单线程 512 KB 上限。
// 所以桶跨度不能太大（51 波落进 <200> 桶就足以把 32 GB 卡吃光，见 v0.3.21）。
//
// N_PAIRS == 0 表示调用方不需要逐事件干涉项（writeResult 的路径）：累加器整块
// 被 `if constexpr` 挡掉、不进栈帧，局部内存退化为 2·N_PARTIALS 个 double。
// N_PAIRS > 0 时累加器只保留"被导出的对"的槽位（d_pair_lookup 给出紧凑槽位），
// 而不是原来的 ninterference 个槽。
template <int N_PARTIALS, int N_PAIRS>
CTPWA_KERNEL void
computeModWithInterference(const ctComplex* __restrict__ result_matrix,
    double* __restrict__ final_result,
    double* __restrict__ interference_matrix,
    double* __restrict__ event_interference,   // [nEvents × npairs_exported] 原始(未归一) Σ_λ2Re(A_i A_j*)
    int* d_nSLvectors, double* total_result,
    const int* __restrict__ d_wave_mask,
    double* __restrict__ d_selected_integral,
    double* __restrict__ d_event_total,
    const int* __restrict__ d_pair_lookup,     // 可空; [ninterference] 导出对→紧凑槽位, 未导出 = -1
    int npairs_exported,
    int npartials, int nEvents, int ngls, int npolar)
{
    const int event_idx = CTPWA_TID();

    if (event_idx >= nEvents)
        return;

    double total_result_value = *total_result;

    // 如果总权重为零，跳过计算
    if (total_result_value <= 0.0) {
        return;
    }

    // 局部数组：假设npartials <= N_PARTIALS，否则需要调整模板桶
    const int MAX_PARTIALS = N_PARTIALS;

    if (npartials > MAX_PARTIALS) {
        printf("Error: npartials=%d exceeds maximum supported value (%d).\n",
            npartials, MAX_PARTIALS);
        return;
    }

    // 使用局部数组，避免设备端malloc
    double partial_real[MAX_PARTIALS];
    double partial_imag[MAX_PARTIALS];
    // 逐事件干涉累加器: 只为被导出的对保留槽位。N_PAIRS==0 时只有 1 个占位
    // 元素且所有访问都被 `if constexpr` 删掉 → 不进栈帧。
    double interference_accumulator[N_PAIRS > 0 ? N_PAIRS : 1];

    // 初始化干涉累加器为0
    if constexpr (N_PAIRS > 0) {
        for (int q = 0; q < N_PAIRS; q++) {
            interference_accumulator[q] = 0.0;
        }
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
                    double a = partial_real[i];
                    double b = partial_imag[i];
                    double c = partial_real[j];
                    double d = partial_imag[j];
                    interference_value = 2.0 * (a * c + b * d);
                }

                // 计算干涉矩阵中的索引（上三角存储）
                int idx = getInterferenceIndex(i, j, npartials);

                // 原子累加到全局干涉矩阵（对所有事件和极化求和）
                atomicAdd(&interference_matrix[idx], interference_value);

                // 逐事件干涉项: 只累加被导出的对（槽位由 d_pair_lookup 给出,
                // 每线程只保留 npairs_exported 个槽位而非 ninterference 个）
                if constexpr (N_PAIRS > 0) {
                    const int q = d_pair_lookup[idx];
                    if (q >= 0) {
                        interference_accumulator[q] += interference_value;
                    }
                }
            }
        }
    }

    // 将干涉矩阵累加器写入 event_interference（原始值, 未除以总积分:
    // 与 weight_<i> 同单位, 满足 weight 守恒: Σ_i|A_i|² + Σ_{i<j}2Re(A_iA_j*) = total）
    // 布局为紧凑的 [npairs_exported × nEvents]
    if constexpr (N_PAIRS > 0) {
        if (event_interference != nullptr) {
            for (int q = 0; q < npairs_exported; ++q) {
                event_interference[(size_t)event_idx + (size_t)nEvents * q] =
                    interference_accumulator[q];
            }
        }
    }

    // 子集总强度覆盖每事件总值 + 子集积分累加（下游归一化自动切换）
    if (d_wave_mask != nullptr) {
        d_event_total[event_idx] = selected_acc;
        atomicAdd(d_selected_integral, selected_acc);
    }
}

// 局部内存池诊断: CUDA 按"每线程局部字节 × 最大常驻线程数(与 grid 无关)"整块预留,
// 所以大模板桶会让"NLL 拟合一切正常、一写结果就 OOM"。超过空闲显存 25% 时提示一次
// （指向分波数/导出对数，而不是让用户去怀疑数据量）。
static void warnIfLocalMemoryHeavy(size_t local_per_thread, const char* tag)
{
    static std::string warned;
    if (warned.find(tag) != std::string::npos) return;
    const size_t resident = ctpwa::max_resident_threads();
    if (resident <= 1) return;   // CPU 后端: 没有局部内存池概念
    const size_t pool = local_per_thread * resident;
    const size_t free_mem = ctpwa::available_memory();
    if (free_mem == 0 || pool * 4 < free_mem) return;
    warned += tag;
    fprintf(stderr,
        "[ctpwa] warn: %s 每线程局部内存 %.1f KB × 常驻 %zu 线程 ≈ %.2f GiB"
        "（当前空闲 %.2f GiB）——局部内存按常驻线程数整块预留、与 grid 无关；"
        "若此处 OOM，先减分波数/导出对数，而不是减 phsp\n",
        tag, local_per_thread / 1024.0, resident,
        pool / 1073741824.0, free_mem / 1073741824.0);
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
    // 局部内存(不是共享内存)是这里的成本大头: 每线程 2·N_PARTIALS 个 double +
    // (导出时) N_PAIRS 个 double。CUDA 按"最大常驻线程数 × 每线程字节"整块预留，
    // 与 grid 无关，所以桶必须贴着实际需要选，且不需要逐事件干涉项时 N_PAIRS=0
    // 让累加器整块消失（164 KB/线程 → 3 KB/线程，V100 上 25 GiB → 0.5 GiB）。
    int pair_bucket = 0;
    int* d_pair_lookup = nullptr;
    if (d_event_interference != nullptr) {
        if (d_pair_list == nullptr || npairs_exported <= 0) {
            throw std::runtime_error(
                "computeResults: 导出逐事件干涉项需要非空 pair_list 且 pairs>0"
                "（不再支持一次性导出全部 ninterference 对——每线程局部内存会爆）");
        }
        if (npairs_exported <= 32) pair_bucket = 32;
        else if (npairs_exported <= 128) pair_bucket = 128;
        else if (npairs_exported <= 512) pair_bucket = 512;
        else if (npairs_exported <= 2048) pair_bucket = 2048;
        else {
            char buf[256];
            snprintf(buf, sizeof(buf),
                     "computeResults: 一次最多导出 2048 对逐事件干涉项(收到 %d)，"
                     "请拆成多次 writeInterfResult 调用",
                     npairs_exported);
            throw std::runtime_error(buf);
        }
        // [ninterference] 查找表: 导出对 → 紧凑槽位, 其余 -1
        const int ninterference = npartials * (npartials + 1) / 2;
        std::vector<int> h_pairs(npairs_exported);
        std::vector<int> h_lookup(ninterference, -1);
        cudaMemcpy(h_pairs.data(), d_pair_list,
                   (size_t)npairs_exported * sizeof(int),
                   cudaMemcpyDeviceToHost);
        for (int q = 0; q < npairs_exported; ++q) {
            int k = h_pairs[q];
            if (k < 0 || k >= ninterference) {
                char buf[256];
                snprintf(buf, sizeof(buf),
                         "computeResults: pair_list[%d]=%d 越界 (0..%d)", q, k,
                         ninterference - 1);
                throw std::runtime_error(buf);
            }
            h_lookup[k] = q;
        }
        alloc_or_throw(reinterpret_cast<void**>(&d_pair_lookup),
                       (size_t)ninterference * sizeof(int), "pair_lookup");
        cudaMemcpy(d_pair_lookup, h_lookup.data(),
                   (size_t)ninterference * sizeof(int), cudaMemcpyHostToDevice);
    }

#define CTPWA_MOD_LAUNCH(NP_, NPAIR_)                                         \
    CTPWA_LAUNCH_P((&computeModWithInterference<NP_, NPAIR_>), gridSize,       \
        blockSize, 0, d_result_matrix, d_partial_result,                       \
        d_interference_matrix, d_event_interference, d_nSLvectors,             \
        d_total_integral, d_wave_mask, d_selected_integral, d_total_out,       \
        d_pair_lookup, npairs_exported, npartials, nEvents, ngls, npolar)

#define CTPWA_MOD_LAUNCH_BY_PAIRS(NP_)                                        \
    do {                                                                      \
        switch (pair_bucket) {                                                \
        case 0:   CTPWA_MOD_LAUNCH(NP_, 0);    break;                         \
        case 32:  CTPWA_MOD_LAUNCH(NP_, 32);   break;                         \
        case 128: CTPWA_MOD_LAUNCH(NP_, 128);  break;                         \
        case 512: CTPWA_MOD_LAUNCH(NP_, 512);  break;                         \
        default:  CTPWA_MOD_LAUNCH(NP_, 2048); break;                         \
        }                                                                     \
    } while (0)

    if (npartials <= 50) {
        CTPWA_MOD_LAUNCH_BY_PAIRS(50);
        warnIfLocalMemoryHeavy((size_t)(2 * 50 + (pair_bucket ? pair_bucket : 1))
                                   * sizeof(double),
            "computeModWithInterference<50>");
    }
    else if (npartials <= 200) {
        CTPWA_MOD_LAUNCH_BY_PAIRS(200);
        warnIfLocalMemoryHeavy((size_t)(2 * 200 + (pair_bucket ? pair_bucket : 1))
                                   * sizeof(double),
            "computeModWithInterference<200>");
    }
    else {
        CTPWA_MOD_LAUNCH_BY_PAIRS(1000);
        warnIfLocalMemoryHeavy((size_t)(2 * 1000 + (pair_bucket ? pair_bucket : 1))
                                   * sizeof(double),
            "computeModWithInterference<1000>");
    }

#undef CTPWA_MOD_LAUNCH_BY_PAIRS
#undef CTPWA_MOD_LAUNCH
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
    if (d_pair_lookup != nullptr) cudaFree(d_pair_lookup);
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

