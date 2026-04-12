#ifndef COMPUTEGRAD_CUH
#define COMPUTEGRAD_CUH

#include <cuComplex.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// 包装函数：计算梯度
// void computeGradient(
//     cuComplex *d_grad, // 设备上的梯度向量
//     // const cuComplex *d_v, // 设备上的向量
//     const cuComplex *d_A, // 设备上的矩阵
//     const cuComplex *d_B, // 设备上的中间结果B
//     const double C,             // 中间结果C
//     const cuComplex *d_T, // 设备上的张量
//     const cuComplex *d_S, // 设备上的中间结果S
//     const double *d_Q,          // 设备上的中间结果Q
//     int nlength, int ngls, int npolar, int phsp_length);
// void compute_gradient(
//     const cuComplex *d_D,
//     const cuComplex *d_P,
//     const cuComplex *d_S,
//     const double *d_U,
//     const cuComplex *d_Q,
//     double phsp_factor,
//     int K, int A, int B, int N,
//     cuComplex *d_grad,
//     cudaStream_t stream = 0);
// void compute_gradient(
//     const cuComplex *d_D,
//     const cuComplex *d_P,
//     const cuComplex *d_S,
//     const double *d_U,
//     const cuComplex *d_Q,
//     double phsp_factor,
//     int K, int A, int B, int N,
//     cuComplex *d_grad); //, cublasHandle_t cublas_handle);

void compute_gradient(const cuComplex *d_D,    // [K, A, B]
                      const cuComplex *d_P,    // [K, N]
                      const cuComplex *d_S,    // [A, B]
                      const double *d_Q,       // [A]
                      const cuComplex *d_B,    // [N]
                      const double *d_weights, // [A]，新增：权重数组
                      double phsp_factor, int K, int A, int B, int N,
                      cuComplex *d_grad, // [K]
                      cublasHandle_t cublas_handle);

#endif