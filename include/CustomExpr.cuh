#ifndef CUSTOM_EXPR_CUH
#define CUSTOM_EXPR_CUH

#include <string>
#include <vector>

// ============================================================================
// Custom DSL（用户自定义共振态模型）
//
// 表达式语言（第一期）:
//   变量: m(不变质量), q(breakup), q0(标称), L(角动量), d(势垒半径)
//   参数: params 列表里的名字（全部视为自由参数，P ≤ 3）
//   常数: 数字, 1j(虚数单位), pi
//   运算: + - * / ^ (幂), 一元负号
//   函数: exp log sin cos sqrt pow(abs) abs Re Im conj csqrt
//
// 编译流水线（host 端，CustomModel 构造时一次完成）:
//   表达式 → AST → 符号微分(一阶+二阶, 复数域) → 字节码段
//   aux 数据布局:
//     aux[0] = P                       # 自由参数数
//     aux[1] = n_seg = 1 + P + P(P+1)/2
//     aux[2..] = 每段: [n_instr, (op,arg0,arg1) x n_instr]  (每段 3 doubles)
//   段 0 = 值 F; 段 1..P = ∂F/∂θ_j; 段 P+1.. = ∂²F/∂θ_j∂θ_k (j≤k)
//   每段执行输出 2 个 double: (re, im)
//
// 设备端: evalCustomExpr 逐段解释执行（栈式，复数运算）
// ============================================================================

// 表达式中可用的变量 id
enum CustomVarId : int {
    CVAR_M = 0,   // 不变质量
    CVAR_Q = 1,   // breakup momentum
    CVAR_Q0 = 2,  // 标称 breakup momentum
    CVAR_L = 3,   // 角动量
    CVAR_D = 4,   // 势垒半径
};

// 函数 id
enum CustomFuncId : int {
    CFUNC_EXP = 0,
    CFUNC_LOG = 1,
    CFUNC_SIN = 2,
    CFUNC_COS = 3,
    CFUNC_SQRT = 4,
    CFUNC_ABS = 5,
    CFUNC_RE = 6,
    CFUNC_IM = 7,
    CFUNC_CONJ = 8,
    CFUNC_CSQRT = 9,   // 复平方根
};

// 指令 opcode
enum CustomOp : int {
    COP_PUSH_NUM = 0,   // arg0 = 数值
    COP_PUSH_VAR = 1,   // arg0 = CustomVarId
    COP_PUSH_PARAM = 2, // arg0 = 参数下标
    COP_PUSH_I = 3,     // 虚数单位 1j
    COP_PUSH_PI = 4,    // π
    COP_ADD = 5,
    COP_SUB = 6,
    COP_MUL = 7,
    COP_DIV = 8,
    COP_POW = 9,
    COP_NEG = 10,
    COP_CALL = 11,      // arg0 = CustomFuncId
    COP_MODEL = 12,     // arg0 = CompositeId（一元）；pow 用 COP_POW
};

// 设备端解释器（定义在 src/CustomExpr.cu，ResModel.cu 调用）
// 逐段执行: seg = aux + seg_offset; 输出 out[0]=re, out[1]=im
__device__ void evalCustomSeg(
    const double* seg, int n_instr,
    double m, double q, double q0, int L, double d,
    const double* params, double* out);

// 获取第 s 段在 aux 中的偏移（host 和 device 共用逻辑）
__host__ __device__ inline int customSegOffset(const double* aux, int s)
{
    int P = (int)aux[0];
    int off = 2;
    int seg = 0;
    while (seg < s) {
        int n_instr = (int)aux[off];
        off += 1 + 3 * n_instr;
        ++seg;
    }
    return off;
}

// 全段解释器（src/CustomExpr.cu）：值段 + P 个一阶段 + P(P+1)/2 个二阶段
// 布局与 evalCustomSeg 相同；d2 按 j≤k 段序读取，对称填充 d2[k][j]=d2[j][k]
__device__ void evalCustomAll(
    const double* aux, int aux_offset,
    double m, double q, double q0, int L, double d,
    const double* params, int P,
    double& Fr, double& Fi,
    double* dFr, double* dFi,
    double* d2Fr, double* d2Fi);

// host 端编译入口（src/CustomExpr.cu）
// expr: 表达式字符串; params: 参数名列表
// 返回 aux 段数据 [P, n_seg, 段...]；抛出 std::runtime_error 报告语法/微分错误
std::vector<double> compileCustomExpr(
    const std::string& expr,
    const std::vector<std::string>& params);

#endif // CUSTOM_EXPR_CUH
