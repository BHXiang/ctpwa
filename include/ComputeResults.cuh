#ifndef COMPUTERESULTS_CUH
#define COMPUTERESULTS_CUH

#include "ComplexType.h"
#include <cuda_runtime.h>

void computeWeightResult(const ctComplex *d_matrix, const ctComplex *d_vector,
                         double *d_total_result, double *d_total_integral,
                         double *d_partial_result, int *d_nSLvectors,
                         int npartials, int nEvents, int ngls, int npolar);
// void computeWeightResult(const ctComplex *d_matrix, const ctComplex
// *d_vector, double *d_total_result, double *d_total_integral, double
// *d_partial_result, double *d_partial_sums, int *d_nSLvectors, int npartials,
// int nEvents, int ngls, int npolar); void computeResults(
//     const ctComplex *d_matrix,
//     const ctComplex *d_vector,
//     double *d_total_result,
//     double *d_total_integral,
//     double *d_partial_result,
//     double *d_interference_matrix,
//     double *d_event_interference,
//     int *d_nSLvectors,
//     int npartials, int nEvents, int ngls, int npolar);
void computeResults(const ctComplex *d_matrix, const ctComplex *d_vector,
                    double *d_total_result, double *d_total_integral,
                    double *d_partial_result, double *d_interference_matrix,
                    // double *d_event_interference,
                    int *d_nSLvectors, int npartials, int nEvents, int ngls,
                    int npolar);

// void computeInterference(
//     const ctComplex *d_M,          // 矩阵 M [ngls][nEvents*npolar]
//     const ctComplex *d_v,          // 参数向量 v [ngls]
//     const ctComplex *d_Cov_v,      // v的协方差矩阵 [ngls][ngls]
//     double *d_interference_matrix, // 输出干涉矩阵 [ninterference]
//     double *d_interference_errors, // 输出干涉矩阵标准差 [ninterference]
//     int *d_nSLvectors,
//     int npartials, int nEvents, int ngls, int npolar);

#endif
