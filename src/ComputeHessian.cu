#include <cuComplex.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>

#include <ComputeHessian.cuh>

// 就地共轭kernel
__global__ void conjKernel(cuComplex* data, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) data[i].y = -data[i].y;
}

// 合并Step A+B: 加载B和v到共享内存，计算S和Bu（old convention [[R,C],[-C,R]]），
// 然后计算per-event Hessian（无原子操作）
__global__ void perEventHessianKernel(
    const cuComplex* __restrict__ d_B,       // chunk × n²
    const cuComplex* __restrict__ d_v,       // n
    const double* __restrict__ d_weights,    // chunk
    double* __restrict__ d_hessian_chunk,    // chunk × 4n² (output, per-event slot)
    int nEvents, int n)
{
    const int n2 = 2 * n;
    const int hess_sz = n2 * n2;
    const int evt = blockIdx.x;
    if (evt >= nEvents) return;

    __shared__ cuComplex sB[64 * 64];       // B matrix
    __shared__ double svr[64];              // real(v)
    __shared__ double svi[64];              // imag(v)
    __shared__ double sBu[128];             // Bu = tildeB @ u (2n, NOT divided by S)
    __shared__ double sS;                   // S = u^T @ tildeB @ u

    // 协作加载 B_k → shared memory
    const cuComplex* Bk = d_B + evt * n * n;
    int nn = n * n;
    for (int idx = threadIdx.x; idx < nn; idx += blockDim.x)
        sB[idx] = Bk[idx];
    // 加载 v → shared memory (real/imag split)
    for (int idx = threadIdx.x; idx < n; idx += blockDim.x) {
        svr[idx] = d_v[idx].x;
        svi[idx] = d_v[idx].y;
    }
    __syncthreads();

    double weight = (d_weights != nullptr) ? d_weights[evt] : 1.0;

    // --- 计算 Bu = tildeB @ u，old convention tildeB = [[R, C], [-C, R]] ---
    // Bu[0:n]   = R*vr + C*vi
    // Bu[n:2n]  = -C*vr + R*vi
    for (int i = threadIdx.x; i < n2; i += blockDim.x) {
        double sum = 0.0;
        if (i < n) {
            for (int j = 0; j < n; ++j)
                sum += sB[i * n + j].x * svr[j] + sB[i * n + j].y * svi[j];
        } else {
            int ii = i - n;
            for (int j = 0; j < n; ++j)
                sum += -sB[ii * n + j].y * svr[j] + sB[ii * n + j].x * svi[j];
        }
        sBu[i] = sum;
    }
    __syncthreads();

    // --- 计算 S = u^T @ Bu = Σ_i (vr[i]*Bu[i] + vi[i]*Bu[n+i]) ---
    if (threadIdx.x == 0) {
        double S = 0.0;
        for (int i = 0; i < n; ++i)
            S += svr[i] * sBu[i] + svi[i] * sBu[n + i];
        if (S <= 0.0) S = 1e-30;
        sS = S;
    }
    __syncthreads();

    double invS = 1.0 / sS;
    double invS2 = invS * invS;
    double* hess_out = d_hessian_chunk + evt * hess_sz;

    // --- per-event Hessian: H = 4*w*Bu*Bu^T/S² - 2*w*tildeB/S ---
    // tildeB = [[R, C], [-C, R]] (old convention)
    for (int idx = threadIdx.x; idx < hess_sz; idx += blockDim.x) {
        int i = idx / n2, j = idx % n2;
        double val = 4.0 * weight * sBu[i] * sBu[j] * invS2;

        int a, b;
        if (i < n && j < n) {
            a = i; b = j;
            val += -2.0 * weight * sB[a * n + b].x * invS;
        } else if (i < n && j >= n) {
            a = i; b = j - n;
            val += -2.0 * weight * sB[a * n + b].y * invS;
        } else if (i >= n && j < n) {
            a = i - n; b = j;
            val += 2.0 * weight * sB[a * n + b].y * invS;
        } else {
            a = i - n; b = j - n;
            val += -2.0 * weight * sB[a * n + b].x * invS;
        }
        hess_out[idx] = val;
    }
}

// Step C: 沿events维度归约 per-event Hessian → 单次atomicAdd到全局
__global__ void reduceHessianKernel(
    const double* __restrict__ d_hessian_chunk,  // nEvents × 4n²
    double* __restrict__ d_hessian_global,        // 4n²
    int nEvents, int n)
{
    const int n2 = 2 * n;
    const int hess_sz = n2 * n2;
    const int elem = blockIdx.x;   // 0..hess_sz-1, 每个block归约一个(i,j)
    if (elem >= hess_sz) return;

    __shared__ double s_partial[256];
    double sum = 0.0;
    for (int evt = threadIdx.x; evt < nEvents; evt += blockDim.x)
        sum += d_hessian_chunk[evt * hess_sz + elem];

    s_partial[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) s_partial[threadIdx.x] += s_partial[threadIdx.x + s];
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        unsigned long long* ptr = (unsigned long long*)(d_hessian_global + elem);
        unsigned long long old = *ptr, new_val;
        do {
            new_val = __double_as_longlong(__longlong_as_double(old) + s_partial[0]);
        } while (atomicCAS(ptr, old, new_val) != old);
    }
}

void computeDataHessianContrib(
    const cuComplex* d_amp, const cuComplex* d_vector,
    const double* d_weights,
    double* d_hessian,
    int nEvents, int n_polar, int n_amplitudes)
{
    const int n = n_amplitudes;
    const int n2 = 2 * n;
    const int stride_amp = n_polar * n;
    const int stride_B  = n * n;
    const int hess_sz   = n2 * n2;
    constexpr int kBlockSize = 256;

    // 估算chunk大小（取1/4空闲内存）
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    // per-event: B(n² complex) + localHess(4n² double)
    size_t per_event = stride_B * sizeof(cuComplex) + hess_sz * sizeof(double);
    int max_chunk = (free_mem / 4) / per_event;
    if (max_chunk > nEvents) max_chunk = nEvents;
    if (max_chunk < 1) max_chunk = 1;

    // 一次性分配chunk缓冲区 (B + hessian_chunk)
    cuComplex *d_B;
    double *d_hessian_chunk;
    cudaMalloc(&d_B, max_chunk * stride_B * sizeof(cuComplex));
    cudaMalloc(&d_hessian_chunk, max_chunk * hess_sz * sizeof(double));

    cublasHandle_t handle;
    cublasCreate(&handle);
    cuComplex alpha = make_cuComplex(1.0f, 0.0f);
    cuComplex beta  = make_cuComplex(0.0f, 0.0f);

    for (int off = 0; off < nEvents; off += max_chunk) {
        int chunk = (off + max_chunk <= nEvents) ? max_chunk : (nEvents - off);
        const cuComplex* A_chunk = d_amp + off * stride_amp;
        const double* w_chunk = (d_weights != nullptr) ? d_weights + off : nullptr;

        // 1. B_k = A^H * A (cublas col-major → 读为row-major = A^H A)
        cublasCgemmStridedBatched(handle,
            CUBLAS_OP_N, CUBLAS_OP_C, n, n, n_polar,
            &alpha, A_chunk, n, stride_amp, A_chunk, n, stride_amp,
            &beta, d_B, n, stride_B, chunk);
        int grid = (chunk * stride_B + kBlockSize - 1) / kBlockSize;
        conjKernel<<<grid, kBlockSize>>>(d_B, chunk * stride_B);

        // 2. Per-event Hessian (含S和Bu计算，old convention [[R,C],[-C,R]])
        perEventHessianKernel<<<chunk, kBlockSize>>>(d_B, d_vector, w_chunk,
            d_hessian_chunk, chunk, n);

        // 3. 归约 per-event → 全局（每chunk仅 4n² 次atomicAdd）
        reduceHessianKernel<<<hess_sz, kBlockSize>>>(d_hessian_chunk, d_hessian,
            chunk, n);
    }

    cublasDestroy(handle);
    cudaFree(d_B); cudaFree(d_hessian_chunk);
}

// ========== phsp Hessian ==========
// 关键：phsp_matrix_未经conjKernel修复（与data部分不同），存储为共轭: sP.y = -C_true
// 因此 Pu = P_stored * v (复数乘法) 恰好得到 tildeP_true @ u:
//   Re(P*v) = R_true*vr + C_true*vi  = (tildeP_true @ u)_top
//   Im(P*v) = R_true*vi - C_true*vr  = (tildeP_true @ u)_bot
// tildeP贡献用 [[R,-C],[C,R]] 约定来补偿共轭存储: sP.y=-C_true → -sP.y=C_true
__global__ void phspHessianKernel(
    const cuComplex* __restrict__ P,      // n×n (共轭存储)
    const cuComplex* __restrict__ v,      // n
    double invT, double weight,
    double* __restrict__ d_hessian,
    int n)
{
    const int n2 = 2 * n;
    __shared__ cuComplex sP[64 * 64];
    __shared__ double sPu_real[128];

    int nn = n * n;
    for (int idx = threadIdx.x; idx < nn; idx += blockDim.x) sP[idx] = P[idx];
    __syncthreads();

    // Pu = P * v (复数乘法), 利用共轭存储得到正确的tildeP_true @ u
    for (int a = threadIdx.x; a < n; a += blockDim.x) {
        cuComplex pu = make_cuComplex(0, 0);
        for (int b = 0; b < n; ++b) {
            cuComplex pb = sP[a * n + b], vb = v[b];
            pu.x += pb.x * vb.x - pb.y * vb.y;
            pu.y += pb.x * vb.y + pb.y * vb.x;
        }
        sPu_real[a] = pu.x;
        sPu_real[a + n] = pu.y;
    }
    __syncthreads();

    double invT2 = invT * invT;
    for (int idx = threadIdx.x; idx < n2 * n2; idx += blockDim.x) {
        int i = idx / n2, j = idx % n2;
        double val = 0.0;

        // -4*w*Pu*Pu^T/T²
        val += -4.0 * weight * sPu_real[i] * sPu_real[j] * invT2;

        // +2*w*tildeP/T, 用 [[R,-C],[C,R]] 约定补偿共轭存储
        if (i < n && j < n)
            val += 2.0 * weight * sP[i * n + j].x * invT;
        else if (i < n && j >= n)
            val += -2.0 * weight * sP[i * n + (j - n)].y * invT;
        else if (i >= n && j < n)
            val += 2.0 * weight * sP[(i - n) * n + j].y * invT;
        else
            val += 2.0 * weight * sP[(i - n) * n + (j - n)].x * invT;

        unsigned long long* ptr = (unsigned long long*)(d_hessian + i * n2 + j);
        unsigned long long old = *ptr, new_val;
        do {
            new_val = __double_as_longlong(__longlong_as_double(old) + val);
        } while (atomicCAS(ptr, old, new_val) != old);
    }
}

void computePhspHessian(
    const cuComplex* d_phsp_matrix, const cuComplex* d_vector,
    double phsp_factor, double weight,
    double* d_hessian, int n)
{
    double invT = 1.0 / phsp_factor;
    phspHessianKernel<<<1, 256>>>(d_phsp_matrix, d_vector, invT, weight, d_hessian, n);
}

// ========== Hessian约束投影: H_orig = J^T * H_ext * J ==========
// J是n_ext×n_orig的对角块矩阵: J[2*eid][2*oid]=cr, J[2*eid+1][2*oid+1]=ci
// 与extendVectorWithConstraints的实/虚部分离比例约定一致
// d_H_in: 2·n_ext × 2·n_ext 扩展Hessian (只读), d_H_out: 2·n_orig × 2·n_orig 输出
__global__ void reduceHessianKernel(
    const double* __restrict__ H_in,             // 2·n_ext × 2·n_ext (只读)
    double* __restrict__ H_out,                  // 2·n_orig × 2·n_orig (输出)
    const int* __restrict__ d_origin_ids,        // n_constraints
    const int* __restrict__ d_ext_ids,           // n_constraints
    const double* __restrict__ d_re_ratios,      // n_constraints (real ratio)
    const double* __restrict__ d_im_ratios,      // n_constraints (imag ratio)
    int n_constraints, int n_orig, int n_ext)
{
    const int n2o = 2 * n_orig;
    const int n2e = 2 * n_ext;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n2o || j >= n2o) return;

    // 1. 读取origin-origin块 (identity mapping)
    double val = H_in[i * n2e + j];

    // 2. 右乘J: 每个constraint列 j_ext → j_orig
    for (int c = 0; c < n_constraints; ++c) {
        int oid = d_origin_ids[c];
        int eid = d_ext_ids[c];
        double cr = d_re_ratios[c];
        double ci = d_im_ratios[c];

        if (j == 2 * oid)
            val += H_in[i * n2e + 2 * eid] * cr;
        if (j == 2 * oid + 1)
            val += H_in[i * n2e + 2 * eid + 1] * ci;
    }

    // 3. 左乘J^T (J是对角阵，J^T=J): 每个constraint行 i_ext → i_orig
    double val2 = val;
    for (int c = 0; c < n_constraints; ++c) {
        int oid = d_origin_ids[c];
        int eid = d_ext_ids[c];
        double cr = d_re_ratios[c];
        double ci = d_im_ratios[c];

        if (i == 2 * oid)
            val2 += cr * H_in[2 * eid * n2e + j];
        if (i == 2 * oid + 1)
            val2 += ci * H_in[(2 * eid + 1) * n2e + j];
    }

    // 4. 交叉项: 行和列都是约束的情况
    for (int ci_idx = 0; ci_idx < n_constraints; ++ci_idx) {
        int oi = d_origin_ids[ci_idx], ei = d_ext_ids[ci_idx];
        double cri = d_re_ratios[ci_idx], cii = d_im_ratios[ci_idx];
        if (i != 2 * oi && i != 2 * oi + 1) continue;

        for (int cj = 0; cj < n_constraints; ++cj) {
            int oj = d_origin_ids[cj], ej = d_ext_ids[cj];
            double crj = d_re_ratios[cj], cij = d_im_ratios[cj];
            if (j != 2 * oj && j != 2 * oj + 1) continue;

            double contrib = 0.0;
            bool i_real = (i == 2 * oi), j_real = (j == 2 * oj);

            if (i_real && j_real)
                contrib = cri * H_in[2 * ei * n2e + 2 * ej] * crj;
            else if (i_real && !j_real)
                contrib = cri * H_in[2 * ei * n2e + 2 * ej + 1] * cij;
            else if (!i_real && j_real)
                contrib = cii * H_in[(2 * ei + 1) * n2e + 2 * ej] * crj;
            else
                contrib = cii * H_in[(2 * ei + 1) * n2e + 2 * ej + 1] * cij;

            val2 += contrib;
        }
    }

    H_out[i * n2o + j] = val2;
}

void reduceHessianWithConstraints(
    const double* d_hessian_ext,
    double* d_hessian_out,
    const int* d_origin_ids, const int* d_ext_ids,
    const double* d_re_ratios, const double* d_im_ratios,
    int n_constraints, int n_orig, int n_ext)
{
    const int n2o = 2 * n_orig;
    dim3 block(16, 16);
    dim3 grid((n2o + 15) / 16, (n2o + 15) / 16);
    reduceHessianKernel<<<grid, block>>>(
        d_hessian_ext, d_hessian_out,
        d_origin_ids, d_ext_ids,
        d_re_ratios, d_im_ratios,
        n_constraints, n_orig, n_ext);
}
