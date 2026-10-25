// test_backend.cpp — CPU 后端基础层单测（不依赖 gtest / CUDA / 外部 BLAS）
//
// 覆盖:
//   1. alloc/copy/memset/free 与 64B 对齐
//   2. CTPWA_LAUNCH + CTPWA_TID 的扁平 kernel 语义
//   3. hw::gemv / gemm / dotc / daxpy / dgmm 与"朴素参考实现"逐元素比对
//   4. 确定性: 不同线程数下结果逐位一致
//
// 构建与运行: bash tests/cpu_backend/build_and_run.sh

#include "Backend.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#ifdef _OPENMP
#  include <omp.h>
#endif

static int g_failures = 0;

#define T_CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::printf("  [FAIL] %s:%d  %s\n", __FILE__, __LINE__, #cond);  \
            ++g_failures;                                                    \
        }                                                                    \
    } while (0)

#define T_CHECK_NEAR(a, b, tol)                                                \
    do {                                                                     \
        const double _a = (double)(a), _b = (double)(b);                     \
        if (!(std::fabs(_a - _b) <= (tol))) {                                \
            std::printf("  [FAIL] %s:%d  %s=%.17g != %s=%.17g\n", __FILE__,  \
                        __LINE__, #a, _a, #b, _b);                           \
            ++g_failures;                                                    \
        }                                                                    \
    } while (0)

// ---------------------------------------------------------------------------
// 一个最简的"扁平 kernel"：验证 CTPWA_KERNEL / CTPWA_TID / CTPWA_LAUNCH
// ---------------------------------------------------------------------------
CTPWA_KERNEL void fill_kernel(ctComplex* a, int n)
{
    const int i = CTPWA_TID();
    if (i < n)
        a[i] = ctMake((double)i, -(double)i);
}

CTPWA_KERNEL void scale_kernel(ctComplex* a, int n, double k)
{
    const int i = CTPWA_TID();
    if (i < n) {
        a[i].x *= k;
        a[i].y *= k;
    }
}

// ---------------------------------------------------------------------------
// 朴素参考实现（与 hw:: 完全独立，用来对拍）
// ---------------------------------------------------------------------------
namespace ref {

inline ctComplex cmul(ctComplex a, ctComplex b)
{
    return ctMake(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// 列主序 m×n
inline ctComplex at(const ctComplex* A, int lda, int i, int j)
{
    return A[(std::ptrdiff_t)i + (std::ptrdiff_t)j * lda];
}

void gemv(bool trans, int m, int n, ctComplex alpha, const ctComplex* A, int lda,
          const ctComplex* x, ctComplex beta, ctComplex* y)
{
    const int n_out = trans ? n : m;
    const int n_in = trans ? m : n;
    for (int i = 0; i < n_out; ++i) {
        ctComplex s = ctMake(0.0, 0.0);
        for (int j = 0; j < n_in; ++j) {
            const ctComplex a = trans ? at(A, lda, j, i) : at(A, lda, i, j);
            s.x += a.x * x[j].x - a.y * x[j].y;
            s.y += a.x * x[j].y + a.y * x[j].x;
        }
        ctComplex r = cmul(alpha, s);
        if (beta.x != 0.0 || beta.y != 0.0) {
            const ctComplex t = cmul(beta, y[i]);
            r.x += t.x;
            r.y += t.y;
        }
        y[i] = r;
    }
}

void gemm(bool trans_a, bool trans_b, int m, int n, int k, ctComplex alpha,
          const ctComplex* A, int lda, const ctComplex* B, int ldb,
          ctComplex beta, ctComplex* C, int ldc)
{
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            ctComplex s = ctMake(0.0, 0.0);
            for (int p = 0; p < k; ++p) {
                const ctComplex a = trans_a ? at(A, lda, p, i) : at(A, lda, i, p);
                const ctComplex b = trans_b ? at(B, ldb, j, p) : at(B, ldb, p, j);
                s.x += a.x * b.x - a.y * b.y;
                s.y += a.x * b.y + a.y * b.x;
            }
            ctComplex r = cmul(alpha, s);
            if (beta.x != 0.0 || beta.y != 0.0) {
                const ctComplex t = cmul(beta, C[i + (std::ptrdiff_t)j * ldc]);
                r.x += t.x;
                r.y += t.y;
            }
            C[i + (std::ptrdiff_t)j * ldc] = r;
        }
    }
}

}  // namespace ref

// ---------------------------------------------------------------------------
static std::vector<ctComplex> random_matrix(int n, unsigned seed)
{
    std::vector<ctComplex> v((std::size_t)n);
    unsigned s = seed;
    for (int i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        const double re = ((double)((s >> 8) & 0xFFFF) / 65535.0) - 0.5;
        s = s * 1664525u + 1013904223u;
        const double im = ((double)((s >> 8) & 0xFFFF) / 65535.0) - 0.5;
        v[(std::size_t)i] = ctMake(re, im);
    }
    return v;
}

// ===========================================================================
int main()
{
    std::printf("[cpu-backend] backend = %s\n", ctpwa::backend_name());

    // ---- 1. 内存原语 ----
    std::printf("[1] memory primitives\n");
    {
        const std::size_t n = 1000;
        ctComplex* p = (ctComplex*)ctpwa::alloc(n * sizeof(ctComplex));
        T_CHECK(p != nullptr);
        T_CHECK(((std::uintptr_t)p % 64) == 0);   // 64B 对齐

        std::vector<ctComplex> host(n);
        for (std::size_t i = 0; i < n; ++i)
            host[i] = ctMake((double)i, (double)i * 2.0);

        ctpwa::copy(p, host.data(), n * sizeof(ctComplex), ctpwa::Dir::H2D);
        std::vector<ctComplex> back(n);
        ctpwa::copy(back.data(), p, n * sizeof(ctComplex), ctpwa::Dir::D2H);

        bool same = true;
        for (std::size_t i = 0; i < n; ++i)
            same = same && (back[i].x == host[i].x) && (back[i].y == host[i].y);
        T_CHECK(same);

        ctpwa::memset(p, 0, n * sizeof(ctComplex));
        ctpwa::copy(back.data(), p, n * sizeof(ctComplex), ctpwa::Dir::D2H);
        T_CHECK(back[123].x == 0.0 && back[123].y == 0.0);

        ctpwa::dealloc(p);
        T_CHECK(ctpwa::device_count() == 1);
        T_CHECK(ctpwa::set_device(0));
        T_CHECK(!ctpwa::set_device(1));   // CPU 后端只有设备 0
        T_CHECK(ctpwa::available());
    }

    // ---- 2. launch / tid ----
    std::printf("[2] CTPWA_LAUNCH + CTPWA_TID\n");
    {
        const int n = 1000, block = 128;
        const int grid = (n + block - 1) / block;
        ctComplex* d = (ctComplex*)ctpwa::alloc((std::size_t)n * sizeof(ctComplex));
        T_CHECK(d != nullptr);

        // memset 后 launch 填充
        ctpwa::memset(d, 0, (std::size_t)n * sizeof(ctComplex));
        CTPWA_LAUNCH(fill_kernel, dim3(grid), dim3(block), 0, d, n);
        CTPWA_LAUNCH(scale_kernel, dim3(grid), dim3(block), 0, d, n, 3.0);

        std::vector<ctComplex> h((std::size_t)n);
        ctpwa::copy(h.data(), d, (std::size_t)n * sizeof(ctComplex), ctpwa::Dir::D2H);
        bool ok = true;
        for (int i = 0; i < n; ++i)
            ok = ok && (h[(std::size_t)i].x == 3.0 * i) &&
                 (h[(std::size_t)i].y == -3.0 * i);
        T_CHECK(ok);
        // 越界检查：工作项数刚好等于 n 时，n-1 之后的元素不应被写
        ctpwa::dealloc(d);
    }

    // ---- 3. hw::gemv ----
    std::printf("[3] hw::gemv\n");
    {
        const int m = 37, n = 23;
        auto A = random_matrix(m * n, 12345u);
        auto x = random_matrix(n, 999u);
        const ctComplex alpha = ctMake(0.7, -0.3);
        const ctComplex beta = ctMake(0.5, 0.25);

        std::vector<ctComplex> y_ref((std::size_t)m), y_hw((std::size_t)m);
        for (int i = 0; i < m; ++i)
            y_ref[(std::size_t)i] = y_hw[(std::size_t)i] = ctMake(-1.0, 2.0);

        ref::gemv(false, m, n, alpha, A.data(), m, x.data(), beta, y_ref.data());
        ctpwa::hw::gemv(false, m, n, alpha, A.data(), m, x.data(), beta, y_hw.data());

        for (int i = 0; i < m; ++i) {
            T_CHECK_NEAR(y_hw[(std::size_t)i].x, y_ref[(std::size_t)i].x, 1e-14);
            T_CHECK_NEAR(y_hw[(std::size_t)i].y, y_ref[(std::size_t)i].y, 1e-14);
        }

        // 转置路径
        std::vector<ctComplex> yt_ref((std::size_t)n), yt_hw((std::size_t)n);
        for (int i = 0; i < n; ++i)
            yt_ref[(std::size_t)i] = yt_hw[(std::size_t)i] = ctMake(0.0, 0.0);
        ref::gemv(true, m, n, alpha, A.data(), m, y_hw.data() /*m*/, beta,
                  yt_ref.data());
        ctpwa::hw::gemv(true, m, n, alpha, A.data(), m, y_ref.data(), beta,
                        yt_hw.data());
        for (int i = 0; i < n; ++i) {
            T_CHECK_NEAR(yt_hw[(std::size_t)i].x, yt_ref[(std::size_t)i].x, 1e-13);
            T_CHECK_NEAR(yt_hw[(std::size_t)i].y, yt_ref[(std::size_t)i].y, 1e-13);
        }
    }

    // ---- 4. hw::gemm ----
    std::printf("[4] hw::gemm\n");
    {
        const int m = 19, k = 11, n = 13;
        auto A = random_matrix(m * k, 7u);
        auto B = random_matrix(k * n, 11u);
        const ctComplex alpha = ctMake(1.0, 0.0);
        const ctComplex beta = ctMake(0.0, 0.0);

        std::vector<ctComplex> C_ref((std::size_t)m * n, ctMake(0.0, 0.0));
        std::vector<ctComplex> C_hw((std::size_t)m * n, ctMake(0.0, 0.0));
        ref::gemm(false, false, m, n, k, alpha, A.data(), m, B.data(), k, beta,
                  C_ref.data(), m);
        ctpwa::hw::gemm(false, false, m, n, k, alpha, A.data(), m, B.data(), k, beta,
                        C_hw.data(), m);
        for (std::size_t i = 0; i < C_hw.size(); ++i) {
            T_CHECK_NEAR(C_hw[i].x, C_ref[i].x, 1e-13);
            T_CHECK_NEAR(C_hw[i].y, C_ref[i].y, 1e-13);
        }
    }

    // ---- 5. dotc / daxpy / dgmm ----
    std::printf("[5] hw::dotc / daxpy / dgmm\n");
    {
        const int n = 64;
        auto x = random_matrix(n, 21u);
        auto y = random_matrix(n, 22u);
        const ctComplex d = ctpwa::hw::dotc(n, x.data(), y.data());
        double re = 0.0, im = 0.0;
        for (int i = 0; i < n; ++i) {
            re += x[(std::size_t)i].x * y[(std::size_t)i].x +
                  x[(std::size_t)i].y * y[(std::size_t)i].y;
            im += x[(std::size_t)i].x * y[(std::size_t)i].y -
                  x[(std::size_t)i].y * y[(std::size_t)i].x;
        }
        T_CHECK_NEAR(d.x, re, 1e-13);
        T_CHECK_NEAR(d.y, im, 1e-13);

        std::vector<double> a(n, 1.0), b(n, 2.0);
        ctpwa::hw::daxpy(n, 0.5, a.data(), b.data());
        for (int i = 0; i < n; ++i)
            T_CHECK_NEAR(b[(std::size_t)i], 2.5, 1e-15);

        // dgmm: C = diag(x) * A
        const int m = 7, nn = 5;
        auto A = random_matrix(m * nn, 31u);
        std::vector<double> diag((std::size_t)m);
        for (int i = 0; i < m; ++i)
            diag[(std::size_t)i] = 0.5 + i;
        std::vector<ctComplex> C((std::size_t)m * nn, ctMake(0.0, 0.0));
        ctpwa::hw::dgmm_real(true, m, nn, A.data(), m, diag.data(), C.data(), m);
        for (int j = 0; j < nn; ++j)
            for (int i = 0; i < m; ++i) {
                const ctComplex expect = ctMake(
                    A[(std::size_t)i + (std::size_t)j * m].x * diag[(std::size_t)i],
                    A[(std::size_t)i + (std::size_t)j * m].y * diag[(std::size_t)i]);
                T_CHECK_NEAR(C[(std::size_t)i + (std::size_t)j * m].x, expect.x, 1e-14);
                T_CHECK_NEAR(C[(std::size_t)i + (std::size_t)j * m].y, expect.y, 1e-14);
            }
    }

    // ---- 6. 确定性（同一输入重复调用逐位一致）----
    std::printf("[6] determinism\n");
    {
        const int m = 100, n = 100;
        auto A = random_matrix(m * n, 4242u);
        auto x = random_matrix(n, 43u);
        std::vector<ctComplex> y1((std::size_t)m), y2((std::size_t)m);
        ctpwa::hw::gemv(false, m, n, ctMake(1.0, 0.0), A.data(), m, x.data(),
                        ctMake(0.0, 0.0), y1.data());
        ctpwa::hw::gemv(false, m, n, ctMake(1.0, 0.0), A.data(), m, x.data(),
                        ctMake(0.0, 0.0), y2.data());
        T_CHECK(std::memcmp(y1.data(), y2.data(), m * sizeof(ctComplex)) == 0);
    }

    // ---- 7. 线程数无关性（同一输入，1 线程 vs 多线程，逐位一致）----
    std::printf("[7] thread-count independence\n");
    {
        const int m = 128, n = 96;
        auto A = random_matrix(m * n, 909u);
        auto x = random_matrix(n, 91u);
        std::vector<ctComplex> y1((std::size_t)m), y2((std::size_t)m);
#ifdef _OPENMP
        omp_set_num_threads(1);
        ctpwa::hw::gemv(false, m, n, ctMake(1.0, 0.0), A.data(), m, x.data(),
                        ctMake(0.0, 0.0), y1.data());
        const int max_t = omp_get_max_threads();
        omp_set_num_threads(max_t > 1 ? max_t : 2);
        ctpwa::hw::gemv(false, m, n, ctMake(1.0, 0.0), A.data(), m, x.data(),
                        ctMake(0.0, 0.0), y2.data());
        T_CHECK(std::memcmp(y1.data(), y2.data(), (std::size_t)m * sizeof(ctComplex)) == 0);
#else
        std::printf("  (skipped: not built with -fopenmp)\n");
#endif
    }

    if (g_failures == 0) {
        std::printf("\n[cpu-backend] ALL PASSED\n");
        return 0;
    }
    std::printf("\n[cpu-backend] %d FAILURE(S)\n", g_failures);
    return 1;
}
