// ============================================================================
// 自定义节点求值 JIT：符号微分字节码 → CUDA C → NVRTC 运行时编译
//
// 为什么需要：解释器 evalCustomSeg 是栈式逐指令求值（P 段 × 每条指令的
// 分支+数组寻址），free 参数拟合时占 forward 的 98%+。JIT 把字节码直译为
// 直线寄存器代码，内置模型（BWR/BW/Flatte/...）公式按 ResModel.cuh 原文
// 内联，NVRTC 编译成原生 SASS，然后在 (σ, 事件, SL) 网格上物化
// (F, ∂F/∂θ[, ∂²F/∂θ²]) 到 device buffer，consumer kernel 改读 buffer。
//
// 数值等价性：生成代码与解释器逐位一致（同一公式文本、同一运算顺序、
// 主库与 NVRTC 均未开 fast_math → 同样的 libdevice 精确数学函数）。
// 任何生成失败（栈深 > 32 / 未知 opcode / NVRTC 错误）→ 整块回退解释器。
// CTPWA_NO_JIT=1 强制回退（数值对拍）。
// ============================================================================

#include <JITCustom.cuh>
#include <CustomExpr.cuh>
#include <SymbolicDiff.cuh>   // CompositeId
#include <cuda.h>
#include <nvrtc.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <chrono>
#include <map>
#include <sys/stat.h>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

// ============================================================================
// NVRTC 源文件模板（preamble + 生成代码 + 内核）
// 注意：JitNodeFlat / JitArgs 必须与 include/JITCustom.cuh 逐字段一致
// （NVRTC 编译单元内没有 host 头，结构按同一 ABI 规则布局）
// ============================================================================

const char* kPreamble = R"CU(
struct LorentzVector {
    double E, Px, Py, Pz;
    __device__ __forceinline__ LorentzVector() : E(0), Px(0), Py(0), Pz(0) {}
    __device__ __forceinline__ LorentzVector(double e, double px, double py, double pz)
        : E(e), Px(px), Py(py), Pz(pz) {}
    __device__ __forceinline__ double P() const {
        return sqrt(Px * Px + Py * Py + Pz * Pz);
    }
    __device__ __forceinline__ double M() const {
        double m2 = E * E - Px * Px - Py * Py - Pz * Pz;
        return (m2 < 0.0) ? 0.0 : sqrt(m2);
    }
};

struct DeviceMomenta {
    LorentzVector* momenta;
    int n_events;
    int n_particles_per_event;
    __device__ __forceinline__ LorentzVector getMomentum(int event_idx, int particle_idx) const {
        return momenta[event_idx * n_particles_per_event + particle_idx];
    }
};

struct SL { int S; int L; };

struct JitNodeFlat {
    int node_idx;
    int mother_idx, daug1_idx, daug2_idx;
    double mass0, mass1, mass2;
    double bf_d;
    int param_offset;
    int param_count;
    int slice_base;
    int nvals;
};

struct JitArgs {
    const void* d_momenta;
    const void* d_mom_tab;
    const double* all_params;
    const int* res_param_off;
    const void* d_slComb;
    double* out;
    int decayChain_size;
    int nEvents;
    int evt_offset;
    int nSL;
    int nSigma;
    int nJitNodes;
    JitNodeFlat nodes[16];
};

// q(m, m1, m2)（与 AmpGen.cu 的 breakup_momentum 逐位一致）
// 注意: 显式 __noinline__ —— BWR 类模型 d²F 段展开会生成数千次
// jitQ0/jitBf/jitBreakup 调用，若内联会使单函数体膨胀到 ~8 万行，
// LLVM/NVPTX 编译卡死（12.9/13.2 全 arch 实测）。__noinline__ 把
// 「实测不内联」变成「保证不内联」，防止未来编译器版本回归。
__device__ __noinline__ double jitBreakup(double m, double m1, double m2) {
    double q_sq = (m*m - (m1+m2)*(m1+m2)) * (m*m - (m1-m2)*(m1-m2));
    if (q_sq <= 0.0) return 0.0;
    return sqrt(q_sq) / (2.0 * m);
}

// 势垒因子 Bf（ResModel.cuh Bf<double> 原文；pow 与模板版同样落到
// pow(double,double)，主库与 NVRTC 均未开 fast_math → 逐位一致）
// 显式 __noinline__：见 jitBreakup 注释（防内联膨胀）
__device__ __noinline__ double jitBf(int L, double q, double q0, double d) {
    double z = q * d;
    double z0 = q0 * d;
    switch (L) {
    case 0: return 1.0;
    case 1: return sqrt((1.0 + z0 * z0) / (1.0 + z * z));
    case 2: return sqrt((9.0 + 3.0 * z0 * z0 + z0 * z0 * z0 * z0) /
                        (9.0 + 3.0 * z * z + z * z * z * z));
    case 3: return sqrt((pow(z0, 6.0) + 6.0 * pow(z0, 4.0) + 45.0 * z0 * z0 + 225.0) /
                        (pow(z, 6.0) + 6.0 * pow(z, 4.0) + 45.0 * z * z + 225.0));
    case 4: return sqrt((pow(z0, 8.0) + 10.0 * pow(z0, 6.0) + 135.0 * pow(z0, 4.0) +
                         1575.0 * z0 * z0 + 11025.0) /
                        (pow(z, 8.0) + 10.0 * pow(z, 6.0) + 135.0 * pow(z, 4.0) +
                         1575.0 * z * z + 11025.0));
    case 5: return sqrt((pow(z0, 10.0) + 15.0 * pow(z0, 8.0) + 315.0 * pow(z0, 6.0) +
                         6300.0 * pow(z0, 4.0) + 99225.0 * z0 * z0 + 893025.0) /
                        (pow(z, 10.0) + 15.0 * pow(z, 8.0) + 315.0 * pow(z, 6.0) +
                         6300.0 * pow(z, 4.0) + 99225.0 * z * z + 893025.0));
    case 6: return sqrt((pow(z0, 12.0) + 21.0 * pow(z0, 10.0) + 630.0 * pow(z0, 8.0) +
                         17325.0 * pow(z0, 6.0) + 363825.0 * pow(z0, 4.0) +
                         6185025.0 * z0 * z0 + 540326025.0) /
                        (pow(z, 12.0) + 21.0 * pow(z, 10.0) + 630.0 * pow(z, 8.0) +
                         17325.0 * pow(z, 6.0) + 363825.0 * pow(z, 4.0) +
                         6185025.0 * z * z + 540326025.0));
    default: return 1.0;
    }
}

// q0(m0, md1, md2)（computeQ0AD<double> 原文）
// 显式 __noinline__：见 jitBreakup 注释（防内联膨胀）
__device__ __noinline__ double jitQ0(double m0, double md1, double md2) {
    double s_md = md1 + md2;
    double d_md = md1 - md2;
    double m0sq = m0 * m0;
    double q0sq = (m0sq - s_md * s_md) * (m0sq - d_md * d_md) / (4.0 * m0sq);
    if (q0sq < 0.0) q0sq = 0.0;
    return sqrt(q0sq);
}

// 子粒子 q0 链质量（与内核一致：固定质量优先；否则查共振态首个参数；
// 负值/无参数回退事件质量 — R>1 分支语义，R==1 除病态负参数外等价）
__device__ __noinline__ double jitDaugMass(const JitArgs& A, int daug,
                                           double fixed, double kin) {
    double md = fixed;
    if (md <= 0) {
        int off = (A.res_param_off && daug >= 0 && daug < 256)
                      ? A.res_param_off[daug] : -1;
        md = (off >= 0) ? A.all_params[off] : kin;
    }
    if (md <= 0) md = kin;
    return md;
}

// Flatte（ResModel.cuh Flatte<double> 原文；ch 为 (ma,mb) 通道质量对）
// tf-pwa FlatteC 约定: D = m0² - m² - i·m0·Σ gᵢ·(qᵢ/m)，虚部系数 = g_i·m0/2
// 显式 __noinline__：见 jitBreakup 注释（防内联膨胀）
__device__ __noinline__ void jitFlatte(double m, double m0, int n_ch,
                          const double* g, const double* ch,
                          double& re, double& im) {
    double s = m * m;
    double real_part = m0 * m0 - s;
    double i_term_real = 0.0;
    double i_term_imag = 0.0;
    for (int i = 0; i < n_ch; ++i) {
        double m_a = ch[2 * i];
        double m_b = ch[2 * i + 1];
        double sum = m_a + m_b;
        double diff = m_a - m_b;
        double f1_sq = 1.0 - (sum * sum) / s;
        double f1_re, f1_im;
        if (f1_sq >= 0.0) { f1_re = sqrt(f1_sq); f1_im = 0.0; }
        else              { f1_re = 0.0; f1_im = sqrt(0.0 - f1_sq); }
        double f2_sq = 1.0 - (diff * diff) / s;
        double factor2 = (f2_sq > 0.0) ? sqrt(f2_sq) : 0.0;
        double rho_re = f1_re * factor2;
        double rho_im = f1_im * factor2;
        double gw = (m0 / 2.0) * g[i];       // FlatteC 宽度项系数
        i_term_real = i_term_real + gw * rho_re;
        i_term_imag = i_term_imag + gw * rho_im;
    }
    double den_re = real_part + i_term_imag;
    double den_im = 0.0 - i_term_real;
    double den_sq = den_re * den_re + den_im * den_im;
    re = den_re / den_sq;
    im = (0.0 - den_im) / den_sq;
}
)CU";

// ============================================================================
// 代码生成：栈 → 直线寄存器代码
// ============================================================================

// 栈项：表达式（惰性，pop 时固化为临时变量，保证每个子表达式至多求值一次）
struct StackVal {
    std::string re, im;
    bool materialized;   // re/im 已是独立临时变量（可安全多次引用）
};

struct GenCtx {
    std::vector<StackVal> stk;
    std::vector<std::string> lines;
    int t = 0;           // 临时变量计数
    int depth = 0;       // 段内最大栈深
    bool fail = false;
};

// 把表达式固化为 double tN_re/tN_im 临时变量（若尚未）
static void materialize(GenCtx& g, StackVal& v)
{
    if (v.materialized) return;
    std::string name_re = "t" + std::to_string(g.t++) + "_re";
    std::string name_im = "t" + std::to_string(g.t++) + "_im";
    g.lines.push_back("    double " + name_re + " = " + v.re + ";");
    g.lines.push_back("    double " + name_im + " = " + v.im + ";");
    v.re = name_re; v.im = name_im;
    v.materialized = true;
}

static StackVal popVal(GenCtx& g)
{
    if (g.stk.empty()) { g.fail = true; return {"0.0", "0.0", true}; }
    StackVal v = g.stk.back();
    g.stk.pop_back();
    materialize(g, v);
    return v;
}

static void pushVal(GenCtx& g, const std::string& re, const std::string& im)
{
    g.stk.push_back({re, im, false});
    if ((int)g.stk.size() > g.depth) g.depth = (int)g.stk.size();
}

static void pushTemp(GenCtx& g, const std::string& re_expr, const std::string& im_expr)
{
    std::string name_re = "t" + std::to_string(g.t++) + "_re";
    std::string name_im = "t" + std::to_string(g.t++) + "_im";
    g.lines.push_back("    double " + name_re + " = " + re_expr + ";");
    g.lines.push_back("    double " + name_im + " = " + im_expr + ";");
    pushVal(g, name_re, name_im);
    g.stk.back().materialized = true;
}

static const char* varName(int id)
{
    switch (id) {
        case CVAR_M:  return "mm";
        case CVAR_Q:  return "qq";
        case CVAR_Q0: return "q0";
        case CVAR_L:  return "Ld";
        case CVAR_D:  return "F.bf_d";
        case CVAR_MD1: return "md1_q0";
        case CVAR_MD2: return "md2_q0";
        case CVAR_P1P: return "p1_P";
        case CVAR_P1E: return "p1_E";
        case CVAR_P1COSTHETA: return "p1_ct";
        case CVAR_P1PHI: return "p1_phi";
        case CVAR_P2P: return "p2_P";
        case CVAR_P2E: return "p2_E";
        case CVAR_P2COSTHETA: return "p2_ct";
        case CVAR_P2PHI: return "p2_phi";
        default: return nullptr;
    }
}

// 生成一个段（值/一阶/二阶段）。oplist 为 [n_instr][3] 扁平指令。失败 → false。
static bool genSegment(GenCtx& g, const double* instrs, int n_instr, int P)
{
    g.stk.clear();
    // CSE 槽位（COP_STORE/COP_LOAD）：段内命名临时变量表
    std::vector<std::pair<std::string, std::string>> cse_slots;
    for (int i = 0; i < n_instr; ++i) {
        int op = (int)instrs[3 * i];
        double a0 = instrs[3 * i + 1];
        double a1 = instrs[3 * i + 2];
        switch (op) {
            case COP_PUSH_NUM: {
                char buf[64];
                snprintf(buf, sizeof(buf), "%.17g", a0);
                pushVal(g, buf, "0.0");
                break;
            }
            case COP_PUSH_VAR: {
                const char* vn = varName((int)a0);
                if (!vn) { if (getenv("CTPWA_JIT_DEBUG")) fprintf(stderr, "[ctpwa] genSegment: 未知变量 id %d\n", (int)a0); return false; }
                pushVal(g, vn, "0.0");
                break;
            }
            case COP_PUSH_PARAM:
                pushVal(g, "params[" + std::to_string((int)a0) + "]", "0.0");
                break;
            case COP_PUSH_I:
                pushVal(g, "0.0", "1.0");
                break;
            case COP_PUSH_PI:
                pushVal(g, "3.14159265358979323846", "0.0");
                break;
            case COP_STORE: {
                // 非破坏性：栈顶固化为命名临时变量并登记槽位（编译期 CSE）
                int k = (int)a0;
                if (g.stk.empty()) { g.fail = true; return false; }
                auto& v = g.stk.back();
                materialize(g, v);
                if (k >= (int)cse_slots.size()) cse_slots.resize(k + 1);
                cse_slots[k] = {v.re, v.im};
                break;
            }
            case COP_LOAD: {
                // 复用槽位中的命名临时变量
                int k = (int)a0;
                if (k < 0 || k >= (int)cse_slots.size()) { g.fail = true; return false; }
                pushVal(g, cse_slots[k].first, cse_slots[k].second);
                break;
            }
            case COP_ADD: { auto b = popVal(g); auto a = popVal(g);
                pushTemp(g, "(" + a.re + " + " + b.re + ")", "(" + a.im + " + " + b.im + ")"); break; }
            case COP_SUB: { auto b = popVal(g); auto a = popVal(g);
                pushTemp(g, "(" + a.re + " - " + b.re + ")", "(" + a.im + " - " + b.im + ")"); break; }
            case COP_MUL: { auto b = popVal(g); auto a = popVal(g);
                pushTemp(g, "(" + a.re + " * " + b.re + " - " + a.im + " * " + b.im + ")",
                         "(" + a.re + " * " + b.im + " + " + a.im + " * " + b.re + ")"); break; }
            case COP_DIV: { auto b = popVal(g); auto a = popVal(g);
                std::string sname = "t" + std::to_string(g.t++) + "_s";
                g.lines.push_back("    double " + sname + " = (" + b.re + " * " + b.re +
                                  " + " + b.im + " * " + b.im + ");");
                g.lines.push_back("    if (" + sname + " == 0.0) " + sname + " = 1e-30;");
                pushTemp(g, "((" + a.re + " * " + b.re + " + " + a.im + " * " + b.im + ") / " + sname + ")",
                         "((" + a.im + " * " + b.re + " - " + a.re + " * " + b.im + ") / " + sname + ")");
                break; }
            case COP_POW: {
                // 栈序: base 先入 → pop 得 exp, base（与解释器一致）
                auto e = popVal(g); auto b = popVal(g);
                std::string sname = "t" + std::to_string(g.t++) + "_a";
                std::string lr = "t" + std::to_string(g.t++) + "_lr";
                std::string li = "t" + std::to_string(g.t++) + "_li";
                std::string xr = "t" + std::to_string(g.t++) + "_xr";
                std::string xi = "t" + std::to_string(g.t++) + "_xi";
                std::string ex = "t" + std::to_string(g.t++) + "_ex";
                g.lines.push_back("    double " + sname + " = sqrt(" + b.re + " * " + b.re +
                                  " + " + b.im + " * " + b.im + ");");
                g.lines.push_back("    if (" + sname + " == 0.0) " + sname + " = 1e-30;");
                g.lines.push_back("    double " + lr + " = log(" + sname + "), " + li +
                                  " = atan2(" + b.im + ", " + b.re + ");");
                g.lines.push_back("    double " + xr + " = " + e.re + " * " + lr + " - " +
                                  e.im + " * " + li + ", " + xi + " = " + e.re + " * " + li +
                                  " + " + e.im + " * " + lr + ";");
                g.lines.push_back("    double " + ex + " = exp(" + xr + ");");
                pushTemp(g, "(" + ex + " * cos(" + xi + "))", "(" + ex + " * sin(" + xi + "))");
                break; }
            case COP_NEG: { auto a = popVal(g);
                pushTemp(g, "(-" + a.re + ")", "(-" + a.im + ")"); break; }
            case COP_CALL: {
                auto a = popVal(g);
                switch ((int)a0) {
                    case CFUNC_EXP: {
                        std::string e = "t" + std::to_string(g.t++) + "_e";
                        g.lines.push_back("    double " + e + " = exp(" + a.re + ");");
                        pushTemp(g, "(" + e + " * cos(" + a.im + "))", "(" + e + " * sin(" + a.im + "))");
                        break; }
                    case CFUNC_LOG: {
                        std::string sname = "t" + std::to_string(g.t++) + "_a";
                        g.lines.push_back("    double " + sname + " = sqrt(" + a.re + " * " + a.re +
                                          " + " + a.im + " * " + a.im + ");");
                        g.lines.push_back("    if (" + sname + " == 0.0) " + sname + " = 1e-30;");
                        pushTemp(g, "log(" + sname + ")", "atan2(" + a.im + ", " + a.re + ")");
                        break; }
                    case CFUNC_SIN: {
                        std::string e = "t" + std::to_string(g.t++) + "_e";
                        std::string em = "t" + std::to_string(g.t++) + "_em";
                        g.lines.push_back("    double " + e + " = exp(" + a.im + "), " + em +
                                          " = exp(-" + a.im + ");");
                        pushTemp(g, "(sin(" + a.re + ") * (" + e + " + " + em + ") / 2)",
                                 "(cos(" + a.re + ") * (" + e + " - " + em + ") / 2)");
                        break; }
                    case CFUNC_COS: {
                        std::string e = "t" + std::to_string(g.t++) + "_e";
                        std::string em = "t" + std::to_string(g.t++) + "_em";
                        g.lines.push_back("    double " + e + " = exp(" + a.im + "), " + em +
                                          " = exp(-" + a.im + ");");
                        pushTemp(g, "(cos(" + a.re + ") * (" + e + " + " + em + ") / 2)",
                                 "(-sin(" + a.re + ") * (" + e + " - " + em + ") / 2)");
                        break; }
                    case CFUNC_SQRT:
                    case CFUNC_CSQRT: {
                        std::string sname = "t" + std::to_string(g.t++) + "_a";
                        std::string r = "t" + std::to_string(g.t++) + "_r";
                        std::string ii = "t" + std::to_string(g.t++) + "_ii";
                        g.lines.push_back("    double " + sname + " = sqrt(" + a.re + " * " + a.re +
                                          " + " + a.im + " * " + a.im + ");");
                        g.lines.push_back("    double " + r + " = sqrt((" + sname + " + " + a.re + ") / 2);");
                        g.lines.push_back("    double " + ii + " = sqrt((" + sname + " - " + a.re +
                                          ") / 2) * (" + a.im + " < 0 ? -1.0 : 1.0);");
                        pushTemp(g, r, ii);
                        break; }
                    case CFUNC_ABS:
                        pushTemp(g, "sqrt(" + a.re + " * " + a.re + " + " + a.im + " * " + a.im + ")", "0.0");
                        break;
                    case CFUNC_RE:
                        pushTemp(g, a.re, "0.0");
                        break;
                    case CFUNC_IM:
                        pushTemp(g, a.im, "0.0");
                        break;
                    case CFUNC_CONJ:
                        pushTemp(g, a.re, "(-" + a.im + ")");
                        break;
                    default: return false;
                }
                break; }
            case COP_MODEL: {
                switch ((int)a0) {
                    case MODEL_BREAKUP_Q0: {  // [m0, m1, m2]
                        auto m2v = popVal(g); auto m1v = popVal(g); auto m0v = popVal(g);
                        pushTemp(g, "jitQ0(" + m0v.re + ", " + m1v.re + ", " + m2v.re + ")", "0.0");
                        break; }
                    case MODEL_BF: {  // [L, q, q0, d]
                        auto dv = popVal(g); auto q0v = popVal(g); auto qv = popVal(g);
                        auto Lv = popVal(g);
                        pushTemp(g, "jitBf((int)(" + Lv.re + "), " + qv.re + ", " + q0v.re +
                                    ", " + dv.re + ")", "0.0");
                        break; }
                    case MODEL_BW: {  // [m, m0, g0]
                        auto g0v = popVal(g); auto m0v = popVal(g); auto mv = popVal(g);
                        std::string x = "t" + std::to_string(g.t++) + "_x";
                        std::string y = "t" + std::to_string(g.t++) + "_y";
                        std::string s = "t" + std::to_string(g.t++) + "_s";
                        g.lines.push_back("    double " + x + " = " + m0v.re + " * " + m0v.re +
                                          " - " + mv.re + " * " + mv.re + ";");
                        g.lines.push_back("    double " + y + " = " + m0v.re + " * " + g0v.re + ";");
                        g.lines.push_back("    double " + s + " = " + x + " * " + x + " + " + y +
                                          " * " + y + ";");
                        pushTemp(g, "(" + x + " / " + s + ")", "(" + y + " / " + s + ")");
                        break; }
                    case MODEL_BWR: {  // [m, m0, g0, L, q, q0, d]
                        auto dv = popVal(g); auto q0v = popVal(g); auto qv = popVal(g);
                        auto Lv = popVal(g); auto g0v = popVal(g); auto m0v = popVal(g);
                        auto mv = popVal(g);
                        std::string bf = "t" + std::to_string(g.t++) + "_bf";
                        std::string ga = "t" + std::to_string(g.t++) + "_ga";
                        std::string x = "t" + std::to_string(g.t++) + "_x";
                        std::string y = "t" + std::to_string(g.t++) + "_y";
                        std::string s = "t" + std::to_string(g.t++) + "_s";
                        g.lines.push_back("    double " + bf + " = jitBf((int)(" + Lv.re + "), " +
                                          qv.re + ", " + q0v.re + ", " + dv.re + ");");
                        g.lines.push_back("    double " + ga + " = " + g0v.re + " * pow(" + qv.re +
                                          " / " + q0v.re + ", (double)(2 * (int)(" + Lv.re + ") + 1)) * (" +
                                          m0v.re + " / " + mv.re + ") * pow(" + bf + ", 2.0);");
                        g.lines.push_back("    double " + x + " = " + m0v.re + " * " + m0v.re +
                                          " - " + mv.re + " * " + mv.re + ";");
                        g.lines.push_back("    double " + y + " = " + m0v.re + " * " + ga + ";");
                        g.lines.push_back("    double " + s + " = " + x + " * " + x + " + " + y +
                                          " * " + y + ";");
                        pushTemp(g, "(" + x + " / " + s + ")", "(" + y + " / " + s + ")");
                        break; }
                    case MODEL_FLATTE: {  // [m, m0, g0..g_{n-1}, (ma,mb)0..]
                        int n_ch = (int)a1;
                        if (n_ch < 1 || n_ch > 4) return false;
                        std::vector<std::string> chv(2 * n_ch), gv(n_ch);
                        for (int i = 2 * n_ch - 1; i >= 0; --i) chv[i] = popVal(g).re;
                        for (int i = n_ch - 1; i >= 0; --i) gv[i] = popVal(g).re;
                        auto m0v = popVal(g); auto mv = popVal(g);
                        std::string ga = "t" + std::to_string(g.t++) + "_ga";
                        std::string ch = "t" + std::to_string(g.t++) + "_ch";
                        std::string rr = "t" + std::to_string(g.t++) + "_rr";
                        std::string ri = "t" + std::to_string(g.t++) + "_ri";
                        g.lines.push_back("    double " + ga + "[4] = {" + gv[0] +
                                          (n_ch > 1 ? ", " + gv[1] : "") +
                                          (n_ch > 2 ? ", " + gv[2] : "") +
                                          (n_ch > 3 ? ", " + gv[3] : "") + "};");
                        std::string chinit;
                        for (int i = 0; i < 2 * n_ch; ++i) {
                            if (i) chinit += ", ";
                            chinit += chv[i];
                        }
                        g.lines.push_back("    double " + ch + "[8] = {" + chinit + "};");
                        g.lines.push_back("    double " + rr + ", " + ri + ";");
                        g.lines.push_back("    jitFlatte(" + mv.re + ", " + m0v.re + ", " +
                                          std::to_string(n_ch) + ", " + ga + ", " + ch + ", " +
                                          rr + ", " + ri + ");");
                        pushTemp(g, rr, ri);
                        break; }
                    case MODEL_FLATTE_RHO_RE:
                    case MODEL_FLATTE_RHO_IM: {  // [s, ma, mb] → ρ = csqrt(1-(ma+mb)²/s) * √(clip(1-(ma-mb)²/s,0))
                        auto mbv = popVal(g); auto mav = popVal(g); auto sv = popVal(g);
                        std::string S = "t" + std::to_string(g.t++) + "_S";
                        std::string D = "t" + std::to_string(g.t++) + "_D";
                        std::string f1 = "t" + std::to_string(g.t++) + "_f1";
                        std::string f1r = "t" + std::to_string(g.t++) + "_f1r";
                        std::string f1i = "t" + std::to_string(g.t++) + "_f1i";
                        std::string f2 = "t" + std::to_string(g.t++) + "_f2";
                        std::string rr = "t" + std::to_string(g.t++) + "_rr";
                        std::string ri = "t" + std::to_string(g.t++) + "_ri";
                        g.lines.push_back("    double " + S + " = (" + mav.re + " + " + mbv.re +
                                          ") * (" + mav.re + " + " + mbv.re + ");");
                        g.lines.push_back("    double " + D + " = (" + mav.re + " - " + mbv.re +
                                          ") * (" + mav.re + " - " + mbv.re + ");");
                        g.lines.push_back("    double " + f1 + " = 1.0 - " + S + " / " + sv.re + ";");
                        g.lines.push_back("    double " + f1r + ", " + f1i + ";");
                        g.lines.push_back("    if (" + f1 + " >= 0.0) { " + f1r + " = sqrt(" + f1 +
                                          "); " + f1i + " = 0.0; }");
                        g.lines.push_back("    else { " + f1r + " = 0.0; " + f1i + " = sqrt(-" + f1 + "); }");
                        g.lines.push_back("    double " + f2 + " = 1.0 - " + D + " / " + sv.re + ";");
                        g.lines.push_back("    double " + rr + " = " + f1r + " * ((" + f2 +
                                          " > 0.0) ? sqrt(" + f2 + ") : 0.0);");
                        g.lines.push_back("    double " + ri + " = " + f1i + " * ((" + f2 +
                                          " > 0.0) ? sqrt(" + f2 + ") : 0.0);");
                        if ((int)a0 == MODEL_FLATTE_RHO_RE)
                            pushTemp(g, rr, "0.0");
                        else
                            pushTemp(g, ri, "0.0");
                        break; }
                    case MODEL_ONE:
                        pushVal(g, "1.0", "0.0");
                        break;
                    default: return false;
                }
                break; }
            default:
                if (getenv("CTPWA_JIT_DEBUG"))
                    fprintf(stderr, "[ctpwa] genSegment: 未知 op %d (instr %d)\n", op, i);
                return false;
        }
        if (g.fail) { if (getenv("CTPWA_JIT_DEBUG")) fprintf(stderr, "[ctpwa] genSegment: 栈溢出 instr %d\n", i); return false; }
        if (g.depth > 32) { if (getenv("CTPWA_JIT_DEBUG")) fprintf(stderr, "[ctpwa] genSegment: 栈深 %d > 32 instr %d\n", g.depth, i); return false; }   // 解释器 kStack 上限
    }
    return true;
}

// 生成一个节点的函数（grad: 段 0..P；full: 段 0..P + 二阶段）
// aux: [P, n_seg, 段...]。返回 false = 该变体无法生成（调用方降级）。
static bool genNodeFunction(std::string& src, const std::vector<double>& aux,
                            int node_id, bool full)
{
    int P = (int)aux[0];
    if (P < 0 || P > 16) return false;
    int n_seg = (int)aux[1];
    std::vector<std::pair<int, int>> pairs;   // 二阶段 (j,k) 序（j≤k 行主序）
    for (int j = 0; j < P; ++j)
        for (int k = j; k < P; ++k) pairs.push_back({j, k});
    int n_need = 1 + P + (full ? (int)pairs.size() : 0);
    if (n_seg != 1 + P + (int)pairs.size() || n_need > n_seg) return false;

    // 段公共环境（主函数计算，段函数形参传入）
    const char* common_args =
        "const JitArgs& A, const JitNodeFlat& F, int L, double Ld,"
        " double mm, double qq, double md1, double md2,"
        " double m0_q0, double md1_q0, double md2_q0, double q0,"
        " double p1_P, double p1_E, double p1_ct, double p1_phi,"
        " double p2_P, double p2_E, double p2_ct, double p2_phi,"
        " const double* params, double* out";

    // 每段一个独立 __device__ 函数（普通函数，非 __forceinline__：防止被内联
    // 回主函数）。BWR 类模型 d²F 段符号展开可达 2.6 万条指令 → 单函数
    // 8 万行无分支代码，ptxas 对超长单函数编译卡死（实测 sm_80 上
    // nvrtc 12.9/13.2 均 OOM/超时；拆段后 sm_80 数秒编过）。
    std::ostringstream segs;
    int seg_off = 2;
    if (getenv("CTPWA_JIT_DEBUG")) {
        fprintf(stderr, "[ctpwa] genNodeFunction node=%d full=%d P=%d n_seg=%d:",
                node_id, (int)full, P, n_seg);
        for (int s = 0; s < n_seg; ++s) {
            int ni = (int)aux[seg_off];
            fprintf(stderr, " seg%d=%d", s, ni);
            seg_off += 1 + 3 * ni;
        }
        fprintf(stderr, "\n");
        seg_off = 2;   // 复位，正式循环重新走
    }
    for (int s = 0; s < n_need; ++s) {
        int n_instr = (int)aux[seg_off];
        if (n_instr < 0) return false;
        seg_off += 1 + 3 * n_instr;   // 先前进（后续段独立）
        GenCtx g;
        g.lines.clear();
        g.depth = 0;   // 临时变量计数 g.t 跨段单调递增（段间避免重名声明）
        g.fail = false;
        // 段数据从 n_instr 之后开始（与解释器 aux + seg_off + 1 一致；seg_off 已前进）
        if (!genSegment(g, &aux[seg_off - 3 * n_instr], n_instr, P)) return false;
        std::string re, im;
        if (!g.stk.empty()) { re = g.stk.back().re; im = g.stk.back().im; }
        else { re = "0.0"; im = "0.0"; }
        // 段函数体
        segs << "// seg " << s
             << (s == 0 ? ": F" : (s <= P ? ": dF[" + std::to_string(s - 1) + "]"
                                           : ": d2F[" + std::to_string(pairs[s - (P + 1)].first) + "][" +
                                                 std::to_string(pairs[s - (P + 1)].second) + "]"))
             << "\n";
        segs << "__device__ __noinline__ void ctpwa_seg_" << node_id << "_" << s
             << (full ? "_full" : "_grad") << "(" << common_args << ")\n{\n";
        for (const auto& ln : g.lines) segs << "    " << ln << "\n";
        // 输出位置
        if (s == 0) {
            segs << "    out[0] = " << re << "; out[1] = " << im << ";\n";
        } else if (s <= P) {
            segs << "    out[" << (2 + 2 * (s - 1)) << "] = " << re
                 << "; out[" << (2 + 2 * (s - 1) + 1) << "] = " << im << ";\n";
        } else {
            auto [j, k] = pairs[s - (P + 1)];
            int b = 2 + 2 * P + 2 * (j * P + k);
            int bT = 2 + 2 * P + 2 * (k * P + j);
            segs << "    out[" << b << "] = " << re << "; out[" << (b + 1) << "] = " << im << ";\n";
            segs << "    out[" << bT << "] = " << re << "; out[" << (bT + 1) << "] = " << im << ";\n";
        }
        segs << "}\n";
    }

    // 主函数：公共变量计算 + 依次调用段函数（普通函数，不 forceinline）
    std::ostringstream o;
    o << "__device__ void ctpwa_jit_node_" << node_id
      << (full ? "_full" : "_grad")
      << "(JitArgs A, const DeviceMomenta* dm, int evt, int sl, double* out)\n{\n";
    o << "    const JitNodeFlat& F = A.nodes[" << node_id << "];\n";
    o << "    int L = ((const SL*)A.d_slComb)[F.node_idx + sl * A.decayChain_size].L;\n";
    o << "    double Ld = (double)L;\n";
    o << "    LorentzVector pM  = dm->getMomentum(evt, F.mother_idx);\n";
    o << "    LorentzVector pD1 = dm->getMomentum(evt, F.daug1_idx);\n";
    o << "    LorentzVector pD2 = dm->getMomentum(evt, F.daug2_idx);\n";
    o << "    double mm = pM.M();\n";
    o << "    double qq = jitBreakup(mm, pD1.M(), pD2.M());\n";
    o << "    double md1 = pD1.M(), md2 = pD2.M();\n";
    o << "    double m0_q0 = (F.param_count > 0) ? A.all_params[F.param_offset] : 1.0;\n";
    o << "    double md1_q0 = jitDaugMass(A, F.daug1_idx, F.mass1, md1);\n";
    o << "    double md2_q0 = jitDaugMass(A, F.daug2_idx, F.mass2, md2);\n";
    o << "    double q0 = jitBreakup(m0_q0, md1_q0, md2_q0);\n";
    o << "    double p1_P = pD1.P(), p1_E = pD1.E;\n";
    o << "    double p1_ct = (p1_P > 0) ? pD1.Pz / p1_P : 0.0;\n";
    o << "    double p1_phi = atan2(pD1.Py, pD1.Px);\n";
    o << "    double p2_P = pD2.P(), p2_E = pD2.E;\n";
    o << "    double p2_ct = (p2_P > 0) ? pD2.Pz / p2_P : 0.0;\n";
    o << "    double p2_phi = atan2(pD2.Py, pD2.Px);\n";
    o << "    const double* params = A.all_params + F.param_offset;\n";
    for (int s = 0; s < n_need; ++s) {
        o << "    ctpwa_seg_" << node_id << "_" << s
          << (full ? "_full" : "_grad")
          << "(A, F, L, Ld, mm, qq, md1, md2, m0_q0, md1_q0, md2_q0, q0,"
             " p1_P, p1_E, p1_ct, p1_phi, p2_P, p2_E, p2_ct, p2_phi, params, out);\n";
    }
    o << "}\n";
    src += segs.str();
    src += o.str();
    return true;
}

// 组装完整源文件
static std::string buildSource(const std::vector<std::vector<double>>& aux_list,
                               int hessian_target)
{
    std::string src = kPreamble;
    src += "\n// ---- 生成节点函数 ----\n";
    for (size_t i = 0; i < aux_list.size(); ++i) {
        if (!genNodeFunction(src, aux_list[i], (int)i, false)) return "";
        if (hessian_target == (int)i) {
            std::string full_src = src;
            if (!genNodeFunction(full_src, aux_list[i], (int)i, true)) {
                // full 变体生成失败 → 放弃 hessian JIT（grad 保留）
                return "";
            }
            src = std::move(full_src);
        }
    }
    // grad 物化内核
    src += "\nextern \"C\" __global__ void ctpwa_jit_grad(JitArgs A)\n{\n";
    src += "    int sl = blockIdx.x;\n";
    src += "    int evt = threadIdx.x + blockDim.x * blockIdx.y;\n";
    src += "    if (sl >= A.nSL || evt >= A.nEvents) return;\n";
    src += "    const DeviceMomenta* dmb = (const DeviceMomenta*)A.d_momenta;\n";
    src += "    const DeviceMomenta* dmt = (const DeviceMomenta*)A.d_mom_tab;\n";
    src += "    for (int s = 0; s < A.nSigma; ++s) {\n";
    src += "        const DeviceMomenta* dm = (s == 0 || !dmt) ? dmb : &dmt[s];\n";
    src += "        for (int n = 0; n < A.nJitNodes; ++n) {\n";
    src += "            const JitNodeFlat& F = A.nodes[n];\n";
    src += "            double* out = A.out + F.slice_base + ((s * A.nEvents + evt) * A.nSL + sl) * F.nvals;\n";
    for (size_t i = 0; i < aux_list.size(); ++i) {
        src += "            if (n == " + std::to_string(i) + ") ctpwa_jit_node_" +
               std::to_string(i) + "_grad(A, dm, evt, sl, out); else ";
    }
    src += ";\n        }\n    }\n}\n";   // 末尾空语句闭合最后的 else
    // full 物化内核（仅 hessian 目标节点）
    if (hessian_target >= 0) {
        src += "\nextern \"C\" __global__ void ctpwa_jit_full(JitArgs A)\n{\n";
        src += "    int sl = blockIdx.x;\n";
        src += "    int evt = threadIdx.x + blockDim.x * blockIdx.y;\n";
        src += "    if (sl >= A.nSL || evt >= A.nEvents) return;\n";
        src += "    int evt_abs = evt + A.evt_offset;\n";
        src += "    const DeviceMomenta* dmb = (const DeviceMomenta*)A.d_momenta;\n";
        src += "    const DeviceMomenta* dmt = (const DeviceMomenta*)A.d_mom_tab;\n";
        src += "    for (int s = 0; s < A.nSigma; ++s) {\n";
        src += "        const DeviceMomenta* dm = (s == 0 || !dmt) ? dmb : &dmt[s];\n";
        src += "        const JitNodeFlat& F = A.nodes[0];\n";
        src += "        double* out = A.out + F.slice_base + ((s * A.nEvents + evt) * A.nSL + sl) * F.nvals;\n";
        src += "        ctpwa_jit_node_" + std::to_string(hessian_target) + "_full(A, dm, evt_abs, sl, out);\n";
        src += "    }\n}\n";
    }
    return src;
}

// ============================================================================
// NVRTC 编译 + 进程级二进制缓存
// 注意：CUmodule 绑定 context（每 device 的 primary context 不同），不能跨
// GPU 复用 → 缓存编译产物（CUBIN/PTX 字节），prep 时按 gpu 懒加载模块。
// ============================================================================

struct JitBinary {
    std::vector<char> data;
    bool is_ptx = false;
};

static std::map<std::string, std::shared_ptr<JitBinary>>& jitCache()
{
    static std::map<std::string, std::shared_ptr<JitBinary>> cache;
    return cache;
}

// 模块缓存键：aux 内容 + 节点数 + hessian 目标 + SM arch（FNV-1a 64）
static std::string cacheKey(const std::vector<std::vector<double>>& aux_list,
                            int hessian_target, const std::string& arch)
{
    uint64_t h = 1469598103934665603ULL;
    auto mix = [&](uint64_t v) {
        for (int i = 0; i < 8; ++i) {
            h ^= (v >> (8 * i)) & 0xff;
            h *= 1099511628211ULL;
        }
    };
    mix((uint64_t)aux_list.size());
    mix((uint64_t)(uint32_t)hessian_target);
    for (const auto& a : aux_list)
        for (double d : a) {
            uint64_t u;
            memcpy(&u, &d, sizeof(u));
            mix(u);
        }
    char buf[24];
    snprintf(buf, sizeof(buf), "%016llx", (unsigned long long)h);
    return arch + "|" + std::string(buf);
}

// 当前 device 的 SM 架构串（sm_XX；NVRTC 编译选项 + 缓存键）
static std::string jitArch()
{
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    return "sm_" + std::to_string(prop.major) + std::to_string(prop.minor);
}

// 编译源 → 二进制（不加载；加载在 per-gpu prep 时）
static bool compileToBinary(const std::string& src, JitBinary& out, std::string& err)
{
    nvrtcProgram prog = nullptr;
    nvrtcResult r = nvrtcCreateProgram(&prog, src.c_str(), "ctpwa_jit.cu",
                                       0, nullptr, nullptr);
    if (r != NVRTC_SUCCESS) {
        err = std::string("nvrtcCreateProgram: ") + nvrtcGetErrorString(r);
        return false;
    }
    std::string arch = jitArch();
    std::string opt = "--gpu-architecture=" + arch;
    const char* opts[] = { opt.c_str(), "--std=c++17" };
    clock_t t0 = clock();
    r = nvrtcCompileProgram(prog, 2, opts);
    if (getenv("CTPWA_JIT_DEBUG")) {
        fprintf(stderr, "[ctpwa] nvrtcCompileProgram: %.2fs (src %zu B)\n",
                (double)(clock() - t0) / CLOCKS_PER_SEC, src.size());
        FILE* df = fopen("/tmp/ctpwa_jit_src.cu", "w");
        if (df) { fwrite(src.data(), 1, src.size(), df); fclose(df); }
    }
    if (r != NVRTC_SUCCESS) {
        size_t logsz = 0;
        nvrtcGetProgramLogSize(prog, &logsz);
        std::string log(logsz, '\0');
        if (logsz) nvrtcGetProgramLog(prog, &log[0]);
        err = std::string("nvrtcCompileProgram: ") + nvrtcGetErrorString(r) + "\n" + log;
        nvrtcDestroyProgram(&prog);
        return false;
    }
    // 原生 CUBIN 优先（nvrtcGetCUBIN 内部跑 ptxas）；失败退 PTX（加载时 driver JIT）
    size_t sz = 0;
    if (nvrtcGetCUBINSize(prog, &sz) == NVRTC_SUCCESS && sz > 0) {
        out.data.resize(sz);
        if (nvrtcGetCUBIN(prog, out.data.data()) == NVRTC_SUCCESS) {
            out.is_ptx = false;
            nvrtcDestroyProgram(&prog);
            return true;
        }
    }
    size_t psz = 0;
    if (nvrtcGetPTXSize(prog, &psz) == NVRTC_SUCCESS && psz > 0) {
        out.data.resize(psz);
        if (nvrtcGetPTX(prog, out.data.data()) == NVRTC_SUCCESS) {
            out.is_ptx = true;
            nvrtcDestroyProgram(&prog);
            return true;
        }
    }
    nvrtcDestroyProgram(&prog);
    err = "CUBIN 与 PTX 均不可用";
    return false;
}

// 编译看门狗：NVRTC 编译放进独立线程，超过 CTPWA_JIT_TIMEOUT 秒（默认 300）
// 放弃等待并报告失败（→ 整块回退解释器）。编译线程无法强杀，会继续占 CPU
// 直至完成自然退出（shared_ptr 持有状态，无悬挂引用）；这只把「挂死」变成
// 「可控降级」——若未来模型的分割粒度仍触发编译器病态行为，程序照常运行。
static bool compileToBinaryTimeout(const std::string& src, JitBinary& out,
                                   std::string& err)
{
    int timeout_s = 300;
    if (const char* e = getenv("CTPWA_JIT_TIMEOUT")) timeout_s = atoi(e);
    if (timeout_s <= 0) return compileToBinary(src, out, err);

    struct CompileState {
        std::string src;
        bool ok = false;
        std::string err;
        JitBinary bin;
        std::atomic<bool> done{false};
    };
    auto st = std::make_shared<CompileState>();
    st->src = src;
    std::thread t([st] {
        st->ok = compileToBinary(st->src, st->bin, st->err);
        st->done = true;
    });
    const int step_ms = 100;
    int waited = 0;   // 毫秒
    // 注意：上限必须与 waited 同单位（毫秒）。写 timeout_s*1000/step_ms 会把
    // 超时缩小 step_ms 倍（曾实测 30s 配置 0.3s 就误触发）。
    while (!st->done && waited < timeout_s * 1000) {
        std::this_thread::sleep_for(std::chrono::milliseconds(step_ms));
        waited += step_ms;
    }
    if (!st->done) {
        t.detach();   // 线程持有 st（shared_ptr），结束后自清理
        err = "JIT 编译超时（> " + std::to_string(timeout_s) + "s，CTPWA_JIT_TIMEOUT 可调）";
        if (getenv("CTPWA_JIT_DEBUG"))
            fprintf(stderr, "[ctpwa] %s\n", err.c_str());
        return false;
    }
    t.join();
    if (!st->ok) { err = st->err; return false; }
    out = std::move(st->bin);
    return true;
}

// 在当前 context（= 当前 device 的 primary context）加载模块，取内核句柄
static bool loadModuleForGpu(const std::string& key, CUfunction& f_grad,
                             CUfunction& f_full)
{
    auto& cache = jitCache();
    auto it = cache.find(key);
    if (it == cache.end()) return false;
    CUmodule mod = nullptr;
    clock_t t0 = clock();
    CUresult cr = cuModuleLoadData(&mod, it->second->data.data());
    if (getenv("CTPWA_JIT_DEBUG"))
        fprintf(stderr, "[ctpwa] cuModuleLoadData(%s): %.2fs\n",
                it->second->is_ptx ? "PTX" : "CUBIN",
                (double)(clock() - t0) / CLOCKS_PER_SEC);
    if (cr != CUDA_SUCCESS || !mod) return false;
    f_grad = nullptr;
    cr = cuModuleGetFunction(&f_grad, mod, "ctpwa_jit_grad");
    if (cr != CUDA_SUCCESS || !f_grad) return false;
    f_full = nullptr;
    cuModuleGetFunction(&f_full, mod, "ctpwa_jit_full");   // 可缺（无 hessian 目标）
    return true;
}

}  // namespace

// ============================================================================
// host API
// ============================================================================

bool jitEnabled()
{
    static const bool v = ([] {
        const char* e = getenv("CTPWA_NO_JIT");
        return !(e && e[0] == '1');
    })();
    return v;
}

bool jitCompileBlock(JitBlockState& st,
                     const std::vector<JitNodeSpec>& nodes,
                     const std::vector<std::vector<double>>& aux_list,
                     int hessian_target)
{
    st.built = true;
    st.nodes = nodes;
    st.hessian_target = hessian_target;
    if (!jitEnabled()) { if (getenv("CTPWA_JIT_DEBUG")) fprintf(stderr, "[ctpwa] JIT disabled by env\n"); return false; }
    if (nodes.empty() || nodes.size() > 16 || aux_list.size() != nodes.size()) {
        if (getenv("CTPWA_JIT_DEBUG"))
            fprintf(stderr, "[ctpwa] JIT plan: nodes=%zu (需 1..16) aux=%zu\n",
                    nodes.size(), aux_list.size());
        return false;
    }
    if (getenv("CTPWA_JIT_DEBUG")) {
        for (size_t i = 0; i < nodes.size(); ++i)
            fprintf(stderr, "[ctpwa] JIT plan: node %d (node_idx=%d P=%d off=%d) aux[0]=%.0f n_seg=%.0f\n",
                    (int)i, nodes[i].node_idx, nodes[i].param_count, nodes[i].param_offset,
                    aux_list[i][0], aux_list[i][1]);
    }
    // hessian 目标的 full 变体生成失败 → 仅保留 grad（仍可加速 fit 主循环）
    if (hessian_target >= 0) {
        std::string probe = buildSource(aux_list, hessian_target);
        if (probe.empty()) {
            st.hessian_target = -1;
            std::string probe2 = buildSource(aux_list, -1);
            if (probe2.empty()) return false;
        }
    }
    // 探测编译（含 arch；CUBIN 缓存按 key 复用）。失败 → 整块解释器。
    std::string arch = jitArch();
    std::string key = cacheKey(aux_list, st.hessian_target, arch);
    auto& cache = jitCache();
    if (cache.find(key) == cache.end()) {
        // 磁盘缓存（跨进程复用）：$CTPWA_JIT_CACHE_DIR 或 ~/.cache/ctpwa_jit/<arch>/
        // 文件 <key hash>.cubin/.ptx。编译一次（30-80s），之后进程秒级加载。
        std::string hash = key.substr(key.find('|') + 1);
        std::string dir;
        if (const char* cd = getenv("CTPWA_JIT_CACHE_DIR")) dir = cd;
        else dir = std::string(getenv("HOME") ? getenv("HOME") : "/tmp") +
                  "/.cache/ctpwa_jit";
        dir += "/" + arch;
        std::shared_ptr<JitBinary> bin;
        for (const char* ext : {".cubin", ".ptx"}) {
            std::string path = dir + "/" + hash + ext;
            FILE* f = fopen(path.c_str(), "rb");
            if (f) {
                fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
                bin = std::make_shared<JitBinary>();
                bin->data.resize((size_t)n);
                if (n > 0 && fread(bin->data.data(), 1, (size_t)n, f) != (size_t)n) {
                    bin.reset();   // 读失败按未命中处理
                }
                bin->is_ptx = (strcmp(ext, ".ptx") == 0);
                fclose(f);
                if (bin && getenv("CTPWA_JIT_DEBUG"))
                    fprintf(stderr, "[ctpwa] JIT 磁盘缓存命中: %s\n", path.c_str());
                break;
            }
        }
        if (!bin) {
            clock_t t0 = clock();
            std::string src = buildSource(aux_list, st.hessian_target);
            if (getenv("CTPWA_JIT_DEBUG"))
                fprintf(stderr, "[ctpwa] buildSource: %.2fs, src=%zu B\n",
                        (double)(clock() - t0) / CLOCKS_PER_SEC, src.size());
            if (const char* dp = getenv("CTPWA_JIT_DUMP_SRC")) {   // 编译前 dump，卡住也能拿到源码
                FILE* df = fopen(dp, "w");
                if (df) { fwrite(src.data(), 1, src.size(), df); fclose(df); }
            }
            if (src.empty()) return false;
            bin = std::make_shared<JitBinary>();
            std::string err;
            if (!compileToBinaryTimeout(src, *bin, err)) {
                fprintf(stderr, "[ctpwa] JIT 编译失败，该块回退解释器: %s\n", err.c_str());
                return false;
            }
            // 写盘缓存
            const char* ext = bin->is_ptx ? ".ptx" : ".cubin";
            std::string path = dir + "/" + hash + ext;
            size_t pos = 0;
            while ((pos = path.find('/', pos)) != std::string::npos) {
                std::string sub = path.substr(0, pos);
                if (!sub.empty()) mkdir(sub.c_str(), 0755);
                ++pos;
            }
            mkdir(dir.c_str(), 0755);
            FILE* wf = fopen(path.c_str(), "wb");
            if (wf) { fwrite(bin->data.data(), 1, bin->data.size(), wf); fclose(wf); }
            else if (getenv("CTPWA_JIT_DEBUG"))
                fprintf(stderr, "[ctpwa] JIT 缓存写入失败: %s\n", path.c_str());
        }
        cache.emplace(key, bin);
    }
    st.cache_key = key;
    st.enabled = true;
    if (getenv("CTPWA_JIT_DEBUG")) {
        fprintf(stderr, "[ctpwa] JIT 编译成功: nodes=%zu hessian_target=%d key=%s\n",
                nodes.size(), st.hessian_target, key.c_str());
        if (getenv("CTPWA_JIT_DUMP_SRC")) {
            const char* p = getenv("CTPWA_JIT_DUMP_SRC");
            std::string src = buildSource(aux_list, st.hessian_target);
            fprintf(stderr, "=== JIT SRC (dump %s) ===\n%s\n=== END JIT SRC ===\n", p, src.c_str());
        }
    }
    return true;
}

// 公共 prep 逻辑：slice 布局 + 缓冲分配 + args 填充
// full_variant: 只物化 hessian 目标节点（nvals 含二阶），evt_offset 由调用方给
static bool jitPrepareInternal(JitBlockState& st, int gpu,
                               const void* d_momenta, const void* d_mom_tab,
                               const double* all_params, const int* h_res_param_off,
                               const void* d_slComb, int decayChain_size,
                               int nEvents, int evt_offset, int nSL, int nSigma,
                               bool full_variant)
{
    if (!st.enabled) return false;
    auto& ready = full_variant ? st.full_ready : st.grad_ready;
    auto& buf = full_variant ? st.full_buf : st.grad_buf;
    auto& buf_sz = full_variant ? st.full_buf_sz : st.grad_buf_sz;
    auto& args = full_variant ? st.args_full : st.args_grad;
    if ((int)ready.size() <= gpu) {
        ready.resize(gpu + 1, false);
        buf.resize(gpu + 1, nullptr);
        buf_sz.resize(gpu + 1, 0);
        args.resize(gpu + 1);
        if (!full_variant) {
            st.grad_slice.resize(gpu + 1, nullptr);
            st.res_off_dev.resize(gpu + 1, nullptr);
        }
    }
    // 注意：ready[gpu] 只表示"首次 prep 已完成"，不代表参数新鲜——
    // nEvents/evt_offset 每次调用都必须刷新（getHessian 按数据/bkg/phsp
    // 分段调用，各段 nEvents/evt_offset 不同；stale args 会导致 pass-1 用
    // 旧的 nEvents/evt_offset 物化，pass-2 读越界 → 垃圾/nan）。
    // CUmodule 绑定 context → 每个 gpu 各自加载（当前 context = 该 device primary）
    if ((int)st.fn_loaded.size() <= gpu) {
        st.fn_loaded.resize(gpu + 1, false);
        st.f_grad_g.resize(gpu + 1, nullptr);
        st.f_full_g.resize(gpu + 1, nullptr);
    }
    if (!st.fn_loaded[gpu]) {
        CUfunction fg = nullptr, ff = nullptr;
        if (!loadModuleForGpu(st.cache_key, fg, ff)) return false;
        st.f_grad_g[gpu] = fg;
        st.f_full_g[gpu] = ff;
        st.fn_loaded[gpu] = true;
    }

    int64_t total = 0;
    int n = full_variant ? 1 : (int)st.nodes.size();
    JitArgs& A = args[gpu];
    for (int i = 0; i < n; ++i) {
        const JitNodeSpec& sp = full_variant ? st.nodes[st.hessian_target] : st.nodes[i];
        int P = sp.param_count;
        int nv = full_variant ? 2 + 2 * P + 2 * P * P : 2 + 2 * P;
        JitNodeFlat& f = A.nodes[i];
        f.node_idx = sp.node_idx;
        f.mother_idx = sp.mother_idx;
        f.daug1_idx = sp.daug1_idx;
        f.daug2_idx = sp.daug2_idx;
        f.mass0 = sp.mass0;
        f.mass1 = sp.mass1;
        f.mass2 = sp.mass2;
        f.bf_d = sp.bf_d;
        f.param_offset = sp.param_offset;
        f.param_count = P;
        f.slice_base = (int)total;
        f.nvals = nv;
        total += (int64_t)nSigma * nEvents * nSL * nv;
    }
    size_t need = (size_t)total * sizeof(double);
    if (!buf[gpu] || buf_sz[gpu] < need) {
        if (buf[gpu]) cudaFree(buf[gpu]);
        if (cudaMalloc(&buf[gpu], need) != cudaSuccess) { buf[gpu] = nullptr; return false; }
        buf_sz[gpu] = need;
    }

    A.d_momenta = d_momenta;
    A.d_mom_tab = d_mom_tab;
    A.all_params = all_params;
    A.d_slComb = d_slComb;
    A.out = buf[gpu];
    A.decayChain_size = decayChain_size;
    A.nEvents = nEvents;
    A.evt_offset = evt_offset;
    A.nSL = nSL;
    A.nSigma = nSigma;
    A.nJitNodes = n;
    if (getenv("CTPWA_JIT_DEBUG"))
        fprintf(stderr, "[ctpwa] prep(%s): nEvents=%d nSL=%d nSigma=%d total=%.1fMB\n",
                full_variant ? "full" : "grad", nEvents, nSL, nSigma,
                (double)total * 8 / 1048576.0);

    if (!full_variant) {
        // res_param_off 设备副本（粒子 → param_offset；-1 = 无参数）
        if (!st.res_off_dev[gpu]) {
            if (cudaMalloc(&st.res_off_dev[gpu], 256 * sizeof(int)) != cudaSuccess ||
                cudaMemcpy(st.res_off_dev[gpu], h_res_param_off, 256 * sizeof(int),
                           cudaMemcpyHostToDevice) != cudaSuccess) {
                if (st.res_off_dev[gpu]) { cudaFree(st.res_off_dev[gpu]); st.res_off_dev[gpu] = nullptr; }
                return false;
            }
        }
        A.res_param_off = st.res_off_dev[gpu];
        // 链节点 → slice double 偏移（-1 = 解释器）
        std::vector<int> h_slice(decayChain_size, -1);
        for (int i = 0; i < (int)st.nodes.size(); ++i)
            h_slice[st.nodes[i].node_idx] = A.nodes[i].slice_base;
        if (cudaMalloc(&st.grad_slice[gpu], decayChain_size * sizeof(int)) != cudaSuccess ||
            cudaMemcpy(st.grad_slice[gpu], h_slice.data(), decayChain_size * sizeof(int),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
            if (st.grad_slice[gpu]) { cudaFree(st.grad_slice[gpu]); st.grad_slice[gpu] = nullptr; }
            return false;
        }
    } else {
        A.res_param_off = st.res_off_dev[gpu];   // grad 先 prep 过（amp 路径先于 hessian）
    }
    ready[gpu] = true;
    return true;
}

bool jitPrepareGrad(JitBlockState& st, int gpu,
                    const void* d_momenta, const void* d_mom_tab,
                    const double* all_params, const int* h_res_param_off,
                    const void* d_slComb, int decayChain_size,
                    int nEvents, int nSL, int nSigma)
{
    return jitPrepareInternal(st, gpu, d_momenta, d_mom_tab, all_params,
                              h_res_param_off, d_slComb, decayChain_size,
                              nEvents, 0, nSL, nSigma, false);
}

bool jitPrepareFull(JitBlockState& st, int gpu,
                    const void* d_momenta, const void* d_mom_tab,
                    const double* all_params, const int* h_res_param_off,
                    const void* d_slComb, int decayChain_size,
                    int nEvents, int evt_offset, int nSL, int nSigma)
{
    if (st.hessian_target < 0) return false;
    // full 依赖 res_off_dev —— 若 grad 未 prep 过则先补（同参数）
    if ((int)st.grad_ready.size() <= gpu || !st.grad_ready[gpu]) {
        if (!jitPrepareInternal(st, gpu, d_momenta, d_mom_tab, all_params,
                                h_res_param_off, d_slComb, decayChain_size,
                                nEvents, 0, nSL, nSigma, false))
            return false;
    }
    return jitPrepareInternal(st, gpu, d_momenta, d_mom_tab, all_params,
                              h_res_param_off, d_slComb, decayChain_size,
                              nEvents, evt_offset, nSL, nSigma, true);
}

static void launchKernel(const JitBlockState& st, int gpu, const JitArgs& args,
                         CUfunction fn)
{
    // 本地副本 — cuLaunchKernel 同步拷贝参数块，安全
    JitArgs A = args;
    void* params[] = { &A };
    unsigned int gx = (unsigned int)A.nSL;
    unsigned int gy = ((unsigned int)A.nEvents + 255) / 256;
    if (gx == 0 || gy == 0) return;
    CUresult cr = cuLaunchKernel(fn, gx, gy, 1, 256, 1, 1, 0, nullptr, params, nullptr);
    if (cr != CUDA_SUCCESS && getenv("CTPWA_JIT_DEBUG")) {
        const char* nm = nullptr;
        cuGetErrorName(cr, &nm);
        fprintf(stderr, "[ctpwa] cuLaunchKernel 失败: %s\n", nm ? nm : "?");
    }
    if (getenv("CTPWA_JIT_DEBUG") && cr == CUDA_SUCCESS && A.nJitNodes > 0) {
        cudaDeviceSynchronize();
        cudaError_t lerr = cudaGetLastError();
        if (lerr != cudaSuccess)
            fprintf(stderr, "[ctpwa] pass-1 kernel 错误: %s\n", cudaGetErrorString(lerr));
        int nv = A.nodes[0].nvals;
        std::vector<double> head((size_t)A.nSL * std::min(A.nEvents, 4) * nv);
        cudaMemcpy(head.data(), A.out, head.size() * sizeof(double),
                   cudaMemcpyDeviceToHost);
        fprintf(stderr, "[ctpwa] jit head (nvals=%d): ", nv);
        for (int e = 0; e < std::min(A.nEvents, 4); ++e) {
            fprintf(stderr, "\n  evt %d:", e);
            for (int v = 0; v < nv; ++v)
                fprintf(stderr, " %.6g", head[(size_t)e * nv + v]);
        }
        fprintf(stderr, "\n");
    }
}

void jitLaunchGrad(const JitBlockState& st, int gpu)
{
    if (!st.enabled || (int)st.f_grad_g.size() <= gpu || !st.f_grad_g[gpu] ||
        (int)st.grad_ready.size() <= gpu || !st.grad_ready[gpu])
        return;
    launchKernel(st, gpu, st.args_grad[gpu], (CUfunction)st.f_grad_g[gpu]);
}

void jitLaunchFull(const JitBlockState& st, int gpu)
{
    if (!st.enabled || (int)st.f_full_g.size() <= gpu || !st.f_full_g[gpu] ||
        (int)st.full_ready.size() <= gpu || !st.full_ready[gpu])
        return;
    launchKernel(st, gpu, st.args_full[gpu], (CUfunction)st.f_full_g[gpu]);
}

void jitDestroy(JitBlockState& st)
{
    for (auto p : st.grad_buf) if (p) cudaFree(p);
    for (auto p : st.grad_slice) if (p) cudaFree(p);
    for (auto p : st.full_buf) if (p) cudaFree(p);
    for (auto p : st.res_off_dev) if (p) cudaFree(p);
    st.grad_buf.clear(); st.grad_slice.clear(); st.full_buf.clear(); st.res_off_dev.clear();
    st.grad_ready.clear(); st.full_ready.clear(); st.args_grad.clear(); st.args_full.clear();
    st.fn_loaded.clear(); st.f_grad_g.clear(); st.f_full_g.clear();
    st.enabled = false;
    st.cache_key.clear();
}
