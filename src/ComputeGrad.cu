#include <cassert>
#include <cmath>
#include <cstring>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>

// 优化后的M计算核函数
__global__ void compute_M_kernel(const cuComplex* d_D, // [K, A, B]
    const cuComplex* d_S, // [A, B]
    int K, int A, int B,
    cuComplex* d_M) // [K, A]
{
    int a = blockIdx.x * blockDim.x + threadIdx.x;

    if (a < A) {
        // 每个线程处理所有K值和当前a的B维度
        for (int m = 0; m < K; m++) {
            cuComplex sum = make_cuComplex(0.0, 0.0);

            // 预计算基础索引
            int base_d_index = m * A * B + a * B;
            int base_s_index = a * B;

            // 循环B维度（B很小，可以展开）
            for (int b = 0; b < B; b++) {
                // 计算 S[a,b] * conj(D[m,a,b])
                sum = cuCaddf(sum, cuCmulf(d_S[base_s_index + b], cuConjf(d_D[base_d_index + b])));
            }
            // printf("M kernel: a=%d, m=%d, M=%f+%fi\n", a, m, cuCrealf(sum), cuCimagf(sum));

            d_M[m * A + a] = sum;
        }
    }
}

// 计算权重总和的核函数
__global__ void sumWeightsKernel(const double* __restrict__ weights,
    double* __restrict__ sum, int A)
{
    extern __shared__ double sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // 初始化共享内存
    sdata[tid] = 0.0;

    // 加载数据到共享内存
    if (idx < A) {
        sdata[tid] = weights[idx];
    }
    __syncthreads();

    // 归约求和（支持任意block大小）
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 原子加
    if (tid == 0) {
        atomicAdd(sum, sdata[0]);
    }
}

// 计算梯度的主函数（带权重支持）
void compute_gradient(const cuComplex* d_D,    // [K, A, B]
    const cuComplex* d_P,    // [K, N]
    const cuComplex* d_S,    // [A, B]
    const cuComplex* d_Q,    // [A]，现在存储的是 w_g/U_g
    const cuComplex* d_B,    // [N]
    const double* d_weights, // [A]，新增：权重数组
    const double weight_integral, // 权重总和（如果提供了权重数组）
    double phsp_factor,
    int K, int A, int B, int N,
    cuComplex* d_grad, // [K]
    cublasHandle_t cublas_handle)
{
    // 1. 计算中间矩阵 M[A, K] = ∑b (S[a,b] * conj(D[m,a,b]))
    cuComplex* d_M;
    cudaMalloc(&d_M, A * K * sizeof(cuComplex));

    // 优化后的线程分配：每个线程处理一个a的所有K和B
    int threadsPerBlock = 256;
    int blocksPerGrid = (A + threadsPerBlock - 1) / threadsPerBlock;

    compute_M_kernel << <blocksPerGrid, threadsPerBlock >> > (d_D, d_S, K, A, B,
        d_M);

    // 2. 计算权重总和（如果提供了权重数组）
    double sum_weight = static_cast<double>(A); // 默认所有权重为1
    if (d_weights != nullptr) {
        sum_weight = weight_integral;
    }

    // 3. 计算数据项: grad_data = -2 * M^T * (w/U)
    // 注意：d_Q现在存储的是 w_g/U_g
    cuComplex alpha_c = make_cuComplex(-2.0, 0.0); // 直接乘以-2
    cuComplex beta_c = make_cuComplex(0.0, 0.0);

    cublasCgemv(cublas_handle,
        CUBLAS_OP_T,      // 转置操作
        A, K,             // 转置后的维度是 A × K
        &alpha_c, d_M, A, // lda = A
        d_Q, 1,           // d_Q现在是 w/U
        &beta_c, d_grad, 1);

    // 4. 计算相位空间项（如果需要）
    if (phsp_factor != 0.0) {
        cuComplex* d_term2;
        cudaMalloc(&d_term2, K * sizeof(cuComplex));

        // 相位空间项系数: 2 * sum_weight / phsp_factor
        cuComplex alpha2 = make_cuComplex(2.0 * sum_weight / phsp_factor, 0.0);
        cublasCgemv(cublas_handle, CUBLAS_OP_C, N, K, &alpha2, d_P, N, d_B, 1, &beta_c, d_term2, 1);

        // printf("A=%d, sum_weight = %f, phsp_factor = %f\n", A, sum_weight, phsp_factor);

        // cuComplex h_term2[K];
        // cudaMemcpy(h_term2, d_term2, K * sizeof(cuComplex), cudaMemcpyDeviceToHost);
        // for (int i = 0; i < K; i++)
        // {
        //     std::cout << "term2[" << i << "] = (" << cuCrealf(h_term2[i]) << "," << cuCimagf(h_term2[i]) << ")\n";
        // }

        // 将相位空间项加到梯度上
        cuComplex one = make_cuComplex(1.0, 0.0);
        cublasCaxpy(cublas_handle, K, &one, d_term2, 1, d_grad, 1);

        cudaFree(d_term2);
    }

    // 5. 释放临时内存
    cudaFree(d_M);
}

