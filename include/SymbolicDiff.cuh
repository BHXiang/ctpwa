#ifndef SYMBOLIC_DIFF_CUH
#define SYMBOLIC_DIFF_CUH

#include "CustomExpr.cuh"   // CustomVarId / CustomFuncId / CustomOp 枚举

// ============================================================================
// 符号微分引擎（host-only）
//
// 通用符号微分基础设施: AST → 复数域符号导数 → 代数化简 → 字节码段。
// 由 Custom DSL（src/CustomExpr.cu）与内置共振态模型（src/ResModel.cu /
// src/Resonance.cu 的 buildModelAST）共享。
//
// 所有函数均为 host 端（config 加载时一次性执行），不进入 GPU 编译路径。
// 语义约定:
//   - m/q/q0/L/d（运动学变量, NodeType::Var）视为常数, 导数 0
//   - 参数 θ_k（NodeType::Param）导数为 δ_kp（参数始终 push 为纯实数）
//   - 复数域普通链式法则（非 Wirtinger）
//
// 布局约定（aux[]，与 CustomExpr.cuh 中 evalCustomAll 匹配）:
//   aux[0] = P                     # 自由参数数
//   aux[1] = n_seg = 1 + P + P(P+1)/2
//   aux[2..] = 每段 [n_instr, (op,arg0,arg1)×n_instr]  (每段 3 doubles)
//   段 0 = 值 F; 段 1..P = ∂F/∂θ_j; 段 P+1.. = ∂²F/∂θ_j∂θ_k (j≤k)
// ============================================================================

#include <vector>

// ============================================================================
// 表达式 AST
// ============================================================================

enum class NodeType {
    Num, Var, Param, Add, Sub, Mul, Div, Pow, Neg, Func,
    Composite,  // 黑盒复合节点（Bf/BWR/BW/Flatte/breakup q0/ONE）
                // 前向求值 + 导数规则预注册，不展开为基础运算树
};

// 复合节点 id（模型注册表，见 src/ResModel.cu buildModelAST）
enum CompositeId : int {
    MODEL_BREAKUP_Q0 = 0,  // q0 = breakup(m0, md1, md2)
    MODEL_BF = 1,          // Bf(L, q, q0, d) —— L 为固定常量参数
    MODEL_BW = 2,          // BW(m, m0, w0) = 1/(m0²-m² - i·m0·w0)
    MODEL_BWR = 3,         // BWR(m, m0, g0, L, q, q0, d)（含内部 Bf² 链）
    MODEL_FLATTE = 4,      // Flatte(m, m0, [g_i, (ma,mb)_i]...)
    MODEL_ONE = 5,         // 1 + 0i
    MODEL_FLATTE_RHO_RE = 6,  // Re(ρ_i): Re(csqrt(1-S/s) * sqrt(clip(1-D/s,0))) — deriv=0
    MODEL_FLATTE_RHO_IM = 7,  // Im(ρ_i): Im(csqrt(1-S/s) * sqrt(clip(1-D/s,0))) — deriv=0
};

// AST 节点（host-only；派生规则与编译都在 host 端执行）
struct Node {
    NodeType type;
    double num = 0.0;          // Num
    int var_id = 0;            // Var（CustomVarId，见 CustomExpr.cuh）
    int param_id = 0;          // Param
    int func_id = 0;           // Func（CustomFuncId，见 CustomExpr.cuh）
    int composite_id = 0;      // Composite（CompositeId）
    std::vector<Node> kids;

    Node() : type(NodeType::Num) {}
    static Node makeNum(double v) { Node n; n.type = NodeType::Num; n.num = v; return n; }
    static Node makeVar(int id) { Node n; n.type = NodeType::Var; n.var_id = id; return n; }
    static Node makeParam(int id) { Node n; n.type = NodeType::Param; n.param_id = id; return n; }
    static Node makeOp(NodeType t, Node a, Node b) {
        Node n; n.type = t; n.kids = {a, b}; return n;
    }
    static Node makeUnary(NodeType t, Node a) {
        Node n; n.type = t; n.kids = {a}; return n;
    }
    static Node makeFunc(int fid, Node a) {
        Node n; n.type = NodeType::Func; n.func_id = fid; n.kids = {a}; return n;
    }
    static Node makeComposite(int cid, std::vector<Node> args) {
        Node n; n.type = NodeType::Composite; n.composite_id = cid;
        n.kids = std::move(args); return n;
    }
};

// ============================================================================
// AST 构建便捷别名（host）
// ============================================================================

inline Node astNum(double v) { return Node::makeNum(v); }
inline Node astAdd(Node a, Node b) { return Node::makeOp(NodeType::Add, std::move(a), std::move(b)); }
inline Node astSub(Node a, Node b) { return Node::makeOp(NodeType::Sub, std::move(a), std::move(b)); }
inline Node astMul(Node a, Node b) { return Node::makeOp(NodeType::Mul, std::move(a), std::move(b)); }
inline Node astDiv(Node a, Node b) { return Node::makeOp(NodeType::Div, std::move(a), std::move(b)); }
inline Node astNeg(Node a) { return Node::makeUnary(NodeType::Neg, std::move(a)); }
inline Node astSq(Node a) { return Node::makeOp(NodeType::Mul, a, a); }  // a²
inline Node astI() { return Node::makeFunc(CFUNC_CSQRT, Node::makeNum(-1.0)); }  // 1j

// ============================================================================
// 编译流水线（host）
// ============================================================================

// 符号微分: ∂n/∂θ_p。Composite 节点按注册表规则展开（见 src/CustomExpr.cu）
Node deriv(const Node& n, int p);

// 符号二阶微分: ∂²n/∂θ_p∂θ_q。结构性规则直接对 n 的表达式结构做二阶导
// （不是 deriv(deriv(...)) 的再微分——那会把一阶导展开式二次膨胀，
// 实测 BWR d²F 段膨胀到 2.6 万指令）。Composite 节点委托 modelDeriv2。
Node deriv2(const Node& n, int p, int q);

// 代数化简: 0±x→x, x*1→x, x^1→x, 数值折叠, Neg 折叠；Composite 原样穿越
Node simplify(Node n);

// AST → 字节码段（每指令 3 doubles: op, arg0, arg1；op 见 CustomOp/CustomExpr.cuh）
void compileNode(const Node& n, std::vector<double>& seg);

// ============================================================================
// 模型注册表（实现 src/ResModel.cu）
// ============================================================================

// 复合节点 ∂n/∂θ_p 的符号导数。n.type 必须为 NodeType::Composite。
// 模型导数规则以 AST 模板表达（∂C/∂arg_k 展开为代数式/嵌套 Composite），
// 链式法则所需的其他复合节点导数通过 deriv() 递归。
Node modelDeriv(int composite_id, const Node& n, int p);

// 复合节点 ∂²n/∂θ_p∂θ_q 的符号二阶导数。实现策略（见 src/ResModel.cu）：
// 用复合节点的子节点重建其定义表达式，再对重建树做 deriv2() 结构性微分
// （含一阶导闭式与嵌套复合节点的 modelDeriv2 递归）——避免对一阶导展开式
// 再微分导致的 AST 爆炸。无参数依赖的复合节点返回 0。
Node modelDeriv2(int composite_id, const Node& n, int p, int q);

// ============================================================================
// 内置模型符号微分（host，替代 computeNodeFactor<Var>）
// ============================================================================

#include <Resonance.cuh>  // ResModelType

// q0 = breakup(m0, md1, md2) 中 m0/md1/md2 的依赖描述：
//   FixedMass  → makeNum(质量)（config 固定质量，无导数）
//   M0Param    → makeParam(0)（目标共振态质量参数，带导数；m0 或子粒子=目标时）
//   EventMass  → makeVar(CVAR_MD1/2)（事件质量，运行时值，导数 0）
enum class Q0MassDep : int {
    FixedMass = 0,
    M0Param = 1,
    EventMass = 2,
};

// Flatte 传播子 AST 展开：F = (A − iB)/D（不含 Bf 因子；调用方按需乘）
//   A = m0² - m² + Σ (g_i·m0/2)·Im(ρ_i),  B = -Σ (g_i·m0/2)·Re(ρ_i),  D = A² + B²
//   ρ_i = MODEL_FLATTE_RHO_{RE,IM}(s, ma_i, mb_i)（导数 0，自动处理）
// tf-pwa FlatteC 约定: D = m0² - m² - i·m0·Σ gᵢ·(qᵢ/m)，qᵢ/m = ρᵢ/2
// 供 Custom DSL（Flatte() 函数）与 buildModelAST（Flatte 分支）共用。
Node buildFlatteAST(const Node& m, const Node& m0,
                    const std::vector<Node>& gs,
                    const std::vector<Node>& chs);

// 程序化构建模型 AST → deriv → simplify → compileNode → aux[]
// 返回标准 aux 布局 [P, n_seg, 段...]（与 compileCustomExpr/evalCustomAll 一致）
// P: 自由参数数; L/d: 块级势垒参数; channels: Flatte 道质量列表
// md1_dep/md2_dep: q0 链中两个子粒子质量的依赖（与 AD 版 kernel 的回退规则一致）
// has_bf: 目标节点是否有势垒因子（false 时 AST 不含 Bf 因子）
std::vector<double> buildModelAST(
    ResModelType model_type,
    int L, double d,
    int P, int n_channels,
    const std::vector<double>& channel_masses,
    Q0MassDep md1_dep, double md1_fixed,
    Q0MassDep md2_dep, double md2_fixed,
    bool has_bf = true);

#endif // SYMBOLIC_DIFF_CUH
