#include "Backend.h"     // CTPWA_KERNEL / CTPWA_TID / CTPWA_LAUNCH / 双后端
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>
#include <string>

#include <ComputeBF.cuh>

// 分配/启动失败一律抛异常（旧实现只 printf，调用方会拿着 nullptr 继续跑）
static inline void bfCheckCuda(cudaError_t e, const char* what)
{
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("computeBranchingFractions: ") + what +
                                 " 失败: " + cudaGetErrorString(e));
}

static inline void bfCheckCublas(cublasStatus_t s, const char* what)
{
    if (s != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string("computeBranchingFractions: cuBLAS ") + what +
                                 " 失败 (status " + std::to_string((int)s) + ")");
}

// 每事件/极化: 按nSLvectors分组SL分量到各粒子，累加模方和散射矩阵
// 关键: 不除以total_result（与computeModWithInterference不同），产出原始散射矩阵
template <int N_PARTIALS>
CTPWA_KERNEL void computeBFKernel(
    const ctComplex* __restrict__ result_matrix,  // [nEvents × npolar × ngls]
    double* __restrict__ d_partial_integral,       // [npartials]
    double* __restrict__ d_scattering_matrix,      // [npartials × npartials]
    double* __restrict__ d_total_integral,         // [1]
    double* __restrict__ d_square_integral,        // [npartials] Σ|A_i|⁴ (可空)
    const int* __restrict__ d_nSLvectors,          // [npartials]
    int npartials, int nEvents, int ngls, int npolar)
{
    const int event_idx = CTPWA_TID();
    if (event_idx >= nEvents) return;

    double partial_real[N_PARTIALS];
    double partial_imag[N_PARTIALS];

    for (int polar_idx = 0; polar_idx < npolar; polar_idx++) {
        for (int p = 0; p < npartials; p++) {
            partial_real[p] = 0.0;
            partial_imag[p] = 0.0;
        }

        int sltotal = 0;
        for (int p = 0; p < npartials; p++) {
            for (int s = 0; s < d_nSLvectors[p]; s++) {
                ctComplex val = result_matrix[
                    event_idx * npolar * ngls +
                    polar_idx * ngls + (sltotal + s)];
                partial_real[p] += val.x;
                partial_imag[p] += val.y;
            }
            sltotal += d_nSLvectors[p];
        }

        double event_total = 0.0;
        for (int p = 0; p < npartials; p++) {
            double intensity = partial_real[p] * partial_real[p] +
                               partial_imag[p] * partial_imag[p];
            if (d_partial_integral != nullptr)
                atomicAdd(&d_partial_integral[p], intensity);
            if (d_square_integral != nullptr)
                atomicAdd(&d_square_integral[p], intensity * intensity);
            event_total += intensity;
        }

        if (d_scattering_matrix != nullptr) {
            for (int i = 0; i < npartials; i++) {
                // 对角线: 2 * |A_i|^2
                double diag_val = 2.0 * (partial_real[i] * partial_real[i] +
                                         partial_imag[i] * partial_imag[i]);
                atomicAdd(&d_scattering_matrix[i * npartials + i], diag_val);

                for (int j = i + 1; j < npartials; j++) {
                    // 非对角线: 2 * Re(A_i * conj(A_j))
                    double cross_val = 2.0 * (partial_real[i] * partial_real[j] +
                                              partial_imag[i] * partial_imag[j]);
                    atomicAdd(&d_scattering_matrix[i * npartials + j], cross_val);
                    // 对称元素
                    atomicAdd(&d_scattering_matrix[j * npartials + i], cross_val);
                }
            }
        }

        if (d_total_integral != nullptr)
            atomicAdd(d_total_integral, event_total);
    }
}

void computeBranchingFractions(
    const ctComplex* d_matrix,
    const ctComplex* d_vector,
    double* d_partial_integral,
    double* d_scattering_matrix,
    double* d_total_integral,
    double* d_square_integral,
    int* d_nSLvectors,
    int npartials, int nEvents, int ngls, int npolar,
    ctComplex* d_complex_result,
    ctComplex* d_result_matrix)
{
    if (nEvents <= 0 || ngls <= 0 || npolar <= 0) return;   // 空批：不碰 workspace
    if (d_complex_result == nullptr || d_result_matrix == nullptr)
        throw std::runtime_error("computeBranchingFractions: workspace 为 nullptr"
                                 "（应由调用方预分配）");

    cublasHandle_t handle;
    bfCheckCublas(cublasCreate(&handle), "cublasCreate");

    const ctComplex alpha = ctMake(1.0f, 0.0f);
    const ctComplex beta  = ctMake(0.0f, 0.0f);

    // 1. cuBLAS gemv: 计算 S = A * v（写入调用方 workspace）
    bfCheckCublas(CUBLAS_CGEMV(handle, CUBLAS_OP_T,
        ngls, nEvents * npolar,
        &alpha, d_matrix, ngls,
        d_vector, 1,
        &beta, d_complex_result, 1), "gemv");

    // 2. cuBLAS dgmm: A * diag(v) → result_matrix（写入调用方 workspace）
    bfCheckCublas(CUBLAS_CDGMM(handle, CUBLAS_SIDE_LEFT,
        ngls, nEvents * npolar,
        d_matrix, ngls,
        d_vector, 1,
        d_result_matrix, ngls), "dgmm");

    bfCheckCublas(cublasDestroy(handle), "cublasDestroy");

    // 3. 启动kernel
    constexpr int kBlockSize = 128;
    int gridSize = (nEvents + kBlockSize - 1) / kBlockSize;

    if (npartials <= 50)
        CTPWA_LAUNCH(computeBFKernel<50>, gridSize, kBlockSize, 0,
            d_result_matrix, d_partial_integral, d_scattering_matrix,
            d_total_integral, d_square_integral, d_nSLvectors,
            npartials, nEvents, ngls, npolar);
    else if (npartials <= 200)
        CTPWA_LAUNCH(computeBFKernel<200>, gridSize, kBlockSize, 0,
            d_result_matrix, d_partial_integral, d_scattering_matrix,
            d_total_integral, d_square_integral, d_nSLvectors,
            npartials, nEvents, ngls, npolar);
    else
        CTPWA_LAUNCH(computeBFKernel<1000>, gridSize, kBlockSize, 0,
            d_result_matrix, d_partial_integral, d_scattering_matrix,
            d_total_integral, d_square_integral, d_nSLvectors,
            npartials, nEvents, ngls, npolar);

    // 不再只 printf：启动/执行错误直接抛出，避免调用方拿着坏结果继续跑
    bfCheckCuda(cudaGetLastError(), "computeBFKernel 启动");
    bfCheckCuda(cudaDeviceSynchronize(), "computeBFKernel 执行");
    // workspace 由调用方统一释放（跨 batch / 跨扰动方向复用）
}

#include <cmath>

void computeBFErrors(
    const double* J,
    const double* cov,
    double* h_bf_errors,
    int npartials, int n2)
{
    for (int i = 0; i < npartials; ++i) {
        double err2 = 0.0;
        for (int p = 0; p < n2; ++p) {
            double J_ip = J[i * n2 + p];
            if (std::abs(J_ip) < 1e-20) continue;
            for (int q = 0; q < n2; ++q) {
                double J_iq = J[i * n2 + q];
                if (std::abs(J_iq) < 1e-20) continue;
                err2 += J_ip * cov[p * n2 + q] * J_iq;
            }
        }
        h_bf_errors[i] = std::sqrt(std::max(0.0, err2));
    }
}
