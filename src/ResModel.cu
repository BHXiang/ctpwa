#include <ResModel.cuh>
#include <CustomExpr.cuh>
#include <SymbolicDiff.cuh>

#include <vector>

// ============================================================================
// 模型注册表：复合节点符号导数（host，见 SymbolicDiff.cuh）
//
// 每个 CompositeId 的导数规则以 AST 模板表达:
//   ∂C/∂θ_p = Σ_k (∂C/∂arg_k)·deriv(arg_k, p)
// ∂C/∂arg_k 展开为代数式（或用嵌套 Composite 引用），arg_k 的导数经
// deriv() 递归（例如 BWR 中 q0 = MODEL_BREAKUP_Q0 复合节点）。
// ============================================================================

namespace {

// Blatt-Weisskopf 归一多项式系数: N_L(z) = Σ_k c_k·z^{2k}, k = 0..L
// Bf(L,q,q0,d) = sqrt(N_L(q0·d)/N_L(q·d))（与 Bf<T> 实现逐项一致）
const std::vector<std::vector<double>>& bfPolyCoeffs()
{
    static const std::vector<std::vector<double>> C = {
        {1.0},                                                          // L=0
        {1.0, 1.0},                                                     // L=1
        {9.0, 3.0, 1.0},                                                // L=2
        {225.0, 45.0, 6.0, 1.0},                                        // L=3
        {11025.0, 1575.0, 135.0, 10.0, 1.0},                            // L=4
        {893025.0, 99225.0, 6300.0, 210.0, 15.0, 1.0},                  // L=5
        {540326025.0, 6185025.0, 363825.0, 17325.0, 630.0, 21.0, 1.0},  // L=6
    };
    return C;
}

// N_L(z) AST（z 可为任意表达式）
Node bfPoly(int L, const Node& z)
{
    const auto& C = bfPolyCoeffs();
    if (L < 0 || L >= (int)C.size()) return astNum(1.0);
    Node z2 = astSq(z);
    Node pw = astNum(1.0);   // z^{2k}, k=0 时 = 1
    Node sum = astNum(0.0);
    for (int k = 0; k <= L; ++k) {
        sum = astAdd(sum, astMul(astNum(C[L][k]), pw));
        pw = astMul(pw, z2);
    }
    return sum;
}

// N'_L(z) = dN_L/dz = Σ_k c_k·2k·z^{2k-1}
Node bfPolyDeriv(int L, const Node& z)
{
    const auto& C = bfPolyCoeffs();
    if (L < 0 || L >= (int)C.size()) return astNum(0.0);
    Node z2 = astSq(z);
    Node pw = z;           // z^{2k-1}, k=1 时 = z
    Node sum = astNum(0.0);
    for (int k = 1; k <= L; ++k) {
        sum = astAdd(sum, astMul(astNum(C[L][k] * 2.0 * k), pw));
        pw = astMul(pw, z2);
    }
    return sum;
}

// 商法则模板: F = x/s + i·y/s;  ∂Re/∂a = (∂x/∂a·s - x·∂s/∂a)/s²,
// ∂Im/∂a = (∂y/∂a·s - y·∂s/∂a)/s²（BW/BWR 共用）
Node qRe(const Node& x, const Node& s, const Node& s2,
         const Node& dx, const Node& ds)
{
    return astDiv(astSub(astMul(dx, s), astMul(x, ds)), s2);
}
Node qIm(const Node& y, const Node& s, const Node& s2,
         const Node& dy, const Node& ds)
{
    return astDiv(astSub(astMul(dy, s), astMul(y, ds)), s2);
}

}  // namespace

Node modelDeriv(int composite_id, const Node& n, int p)
{
    switch (composite_id) {
        case MODEL_BREAKUP_Q0: {
            // args: [m0, m1, m2]
            // q0 = sqrt(A·B/(4m0²)), A = m0²-(m1+m2)², B = m0²-(m1-m2)²
            // ∂q0/∂m0 = (m0²(A+B) - A·B) / (4·m0³·q0)
            // ∂q0/∂m1 = -(s·B + d·A) / (4·m0²·q0), s = m1+m2, d = m1-m2
            // ∂q0/∂m2 = (d·A - s·B) / (4·m0²·q0)
            Node m0 = n.kids[0], m1 = n.kids[1], m2 = n.kids[2];
            Node s = astAdd(m1, m2), dd = astSub(m1, m2);
            Node A = astSub(astSq(m0), astSq(s));
            Node B = astSub(astSq(m0), astSq(dd));
            Node q0 = n;   // 引用复合节点自身
            Node dq0_dm0 = astDiv(
                astSub(astMul(astSq(m0), astAdd(A, B)), astMul(A, B)),
                astMul(astNum(4.0), astMul(astMul(m0, astSq(m0)), q0)));
            Node dq0_dm1 = astDiv(
                astNeg(astAdd(astMul(s, B), astMul(dd, A))),
                astMul(astNum(4.0), astMul(astSq(m0), q0)));
            Node dq0_dm2 = astDiv(
                astSub(astMul(dd, A), astMul(s, B)),
                astMul(astNum(4.0), astMul(astSq(m0), q0)));
            return astAdd(astAdd(astMul(dq0_dm0, deriv(m0, p)),
                                 astMul(dq0_dm1, deriv(m1, p))),
                          astMul(dq0_dm2, deriv(m2, p)));
        }
        case MODEL_BF: {
            // args: [L, q, q0, d]; L 为 Num（离散，导数 0）
            // Bf = sqrt(N(z0)/N(z)), z = q·d, z0 = q0·d
            // ∂Bf/∂q  = -d·Bf·N'(z) / (2N(z))
            // ∂Bf/∂q0 =  d·Bf·N'(z0) / (2N(z0))
            // ∂Bf/∂d  = (q/d)·∂Bf/∂q + (q0/d)·∂Bf/∂q0
            Node q = n.kids[1], q0 = n.kids[2], d = n.kids[3];
            int L = (int)n.kids[0].num;
            Node z = astMul(q, d), z0 = astMul(q0, d);
            Node Nz = bfPoly(L, z), Nz0 = bfPoly(L, z0);
            Node Npz = bfPolyDeriv(L, z), Npz0 = bfPolyDeriv(L, z0);
            Node BfN = n;   // 引用复合节点自身
            Node dBf_dq = astDiv(astMul(astMul(d, astNeg(BfN)), Npz),
                                 astMul(astNum(2.0), Nz));
            Node dBf_dq0 = astDiv(astMul(astMul(d, BfN), Npz0),
                                  astMul(astNum(2.0), Nz0));
            Node dBf_dd = astAdd(astDiv(astMul(dBf_dq, q), d),
                                 astDiv(astMul(dBf_dq0, q0), d));
            return astAdd(astAdd(astMul(dBf_dq, deriv(q, p)),
                                 astMul(dBf_dq0, deriv(q0, p))),
                          astMul(dBf_dd, deriv(d, p)));
        }
        case MODEL_BW: {
            // args: [m, m0, g0]
            // F = x/s + i·y/s; x = m0²-m², y = m0·g0, s = x²+y²
            Node m = n.kids[0], m0 = n.kids[1], g0 = n.kids[2];
            Node x = astSub(astSq(m0), astSq(m));
            Node y = astMul(m0, g0);
            Node s = astAdd(astSq(x), astSq(y));
            Node s2 = astSq(s);
            // ∂s/∂m0 = 2x·2m0 + 2y·g0;  ∂s/∂g0 = 2y·m0;  ∂s/∂m = 2x·(-2m)
            Node ds_m0 = astAdd(astMul(astMul(astNum(2.0), x), astMul(astNum(2.0), m0)),
                                astMul(astMul(astNum(2.0), y), g0));
            Node ds_g0 = astMul(astMul(astNum(2.0), y), m0);
            Node ds_m = astMul(astMul(astNum(2.0), x), astMul(astNum(-2.0), m));
            // ∂x/∂m0 = 2m0, ∂y/∂m0 = g0;  ∂x/∂g0 = 0, ∂y/∂g0 = m0;  ∂x/∂m = -2m, ∂y/∂m = 0
            Node term_m0 = astAdd(qRe(x, s, s2, astMul(astNum(2.0), m0), ds_m0),
                                  astMul(astI(), qIm(y, s, s2, g0, ds_m0)));
            Node term_g0 = astAdd(qRe(x, s, s2, astNum(0.0), ds_g0),
                                  astMul(astI(), qIm(y, s, s2, m0, ds_g0)));
            Node term_m = astAdd(qRe(x, s, s2, astNeg(astMul(astNum(2.0), m)), ds_m),
                                 astMul(astI(), qIm(y, s, s2, astNum(0.0), ds_m)));
            return astAdd(astAdd(astMul(term_m0, deriv(m0, p)),
                                 astMul(term_g0, deriv(g0, p))),
                          astMul(term_m, deriv(m, p)));
        }
        case MODEL_BWR: {
            // args: [m, m0, g0, L, q, q0, d]
            // F = x/s + i·y/s; x = m0²-m², y = m0·γ, s = x²+y²
            // γ = g0·(q/q0)^(2L+1)·(m0/m)·Bf²（BWR<T> 内部宽度公式）
            Node m = n.kids[0], m0 = n.kids[1], g0 = n.kids[2];
            int L = (int)n.kids[3].num;
            Node q = n.kids[4], q0 = n.kids[5], d = n.kids[6];
            Node BfN = Node::makeComposite(MODEL_BF, {n.kids[3], q, q0, d});
            Node pw = Node::makeOp(NodeType::Pow, astDiv(q, q0), astNum(2.0 * L + 1));
            Node gamma = astMul(astMul(astMul(g0, pw), astDiv(m0, m)), astSq(BfN));
            Node x = astSub(astSq(m0), astSq(m));
            Node y = astMul(m0, gamma);
            Node s = astAdd(astSq(x), astSq(y));
            Node s2 = astSq(s);
            // ∂γ/∂θ（对数微分）:
            //   γ·[∂g0/∂θ/g0 + (2L+1)(∂q/∂θ/q - ∂q0/∂θ/q0) + ∂m0/∂θ/m0
            //      - ∂m/∂θ/m + 2·∂Bf/∂θ/Bf]
            Node dgamma = astAdd(
                astMul(deriv(g0, p), astDiv(gamma, g0)),
                astMul(gamma, astMul(astNum(2.0 * L + 1),
                    astSub(astDiv(deriv(q, p), q), astDiv(deriv(q0, p), q0)))));
            dgamma = astAdd(dgamma, astMul(gamma, astDiv(deriv(m0, p), m0)));
            dgamma = astSub(dgamma, astMul(gamma, astDiv(deriv(m, p), m)));
            dgamma = astAdd(dgamma, astMul(gamma,
                astMul(astNum(2.0), astDiv(modelDeriv(MODEL_BF, BfN, p), BfN))));
            // 商法则
            Node dx = astSub(astMul(astMul(astNum(2.0), m0), deriv(m0, p)),
                             astMul(astMul(astNum(2.0), m), deriv(m, p)));
            Node dy = astAdd(astMul(deriv(m0, p), gamma), astMul(m0, dgamma));
            Node ds = astAdd(astMul(astMul(astNum(2.0), x), dx),
                             astMul(astMul(astNum(2.0), y), dy));
            return astAdd(qRe(x, s, s2, dx, ds), astMul(astI(), qIm(y, s, s2, dy, ds)));
        }
        case MODEL_FLATTE:
            // Phase 6: csqrt 分支导数（COP_SELECT 条件指令）
            return astNum(0.0);
        case MODEL_ONE:
            return astNum(0.0);
        default:
            return astNum(0.0);
    }
}

// ============================================================================
// buildModelAST: 程序化构建内置模型的符号微分 AST → aux[]（替代 computeNodeFactor<Var>）
// ============================================================================

std::vector<double> buildModelAST(
    ResModelType model_type,
    int L, double d,
    int P,                              // 自由参数数
    int n_channels,                     // Flatte: 道数; 其他: 0
    const std::vector<double>& channel_masses)  // Flatte: [m_a0,m_b0, m_a1,m_b1, ...]
{
    Node ast;
    switch (model_type) {
        case ResModelType::BW:
            // F = MODEL_BW(m, θ0, θ1)    // θ0=m0, θ1=w0
            ast = Node::makeComposite(MODEL_BW, {
                Node::makeVar(CVAR_M),
                Node::makeParam(0),
                Node::makeParam(1)
            });
            break;
        case ResModelType::BWR: {
            // F = MODEL_BWR(m, θ0, θ1, L, q, q0, d) × MODEL_BF(L, q, q0, d)
            Node bwr = Node::makeComposite(MODEL_BWR, {
                Node::makeVar(CVAR_M),
                Node::makeParam(0),       // m0
                Node::makeParam(1),       // g0
                Node::makeNum((double)L),
                Node::makeVar(CVAR_Q),
                Node::makeVar(CVAR_Q0),
                Node::makeNum(d)
            });
            Node bf = Node::makeComposite(MODEL_BF, {
                Node::makeNum((double)L),
                Node::makeVar(CVAR_Q),
                Node::makeVar(CVAR_Q0),
                Node::makeNum(d)
            });
            ast = Node::makeOp(NodeType::Mul, bwr, bf);
            break;
        }
        case ResModelType::ONE: {
            // F = MODEL_BF(L, q, q0, d)
            ast = Node::makeComposite(MODEL_BF, {
                Node::makeNum((double)L),
                Node::makeVar(CVAR_Q),
                Node::makeVar(CVAR_Q0),
                Node::makeNum(d)
            });
            break;
        }
        case ResModelType::Flatte: {
            // F = MODEL_FLATTE(m, θ0, θ1..θ_n, ch0a,ch0b, ch1a,ch1b,...) × MODEL_BF(L, q, q0, d)
            // args: [m, m0, g0..g_{n-1}, (ma,mb)0..(ma,mb)_{n-1}]
            std::vector<Node> fargs;
            fargs.push_back(Node::makeVar(CVAR_M));
            fargs.push_back(Node::makeParam(0));  // m0
            for (int i = 0; i < n_channels; ++i)
                fargs.push_back(Node::makeParam(1 + i));  // g_i
            for (int i = 0; i < n_channels; ++i) {
                fargs.push_back(Node::makeNum(channel_masses[2 * i]));
                fargs.push_back(Node::makeNum(channel_masses[2 * i + 1]));
            }
            Node flatte = Node::makeComposite(MODEL_FLATTE, fargs);
            Node bf = Node::makeComposite(MODEL_BF, {
                Node::makeNum((double)L),
                Node::makeVar(CVAR_Q),
                Node::makeVar(CVAR_Q0),
                Node::makeNum(d)
            });
            ast = Node::makeOp(NodeType::Mul, flatte, bf);
            break;
        }
        default:
            // Hist/Custom: 不经过此路径
            return {};
    }

    // 符号微分 + 编译
    std::vector<Node> segs;
    segs.push_back(simplify(ast));                          // 段 0: 值 F
    for (int j = 0; j < P; ++j)
        segs.push_back(simplify(deriv(ast, j)));            // 段 1..P: ∂F/∂θ_j
    for (int j = 0; j < P; ++j)
        for (int k = j; k < P; ++k)
            segs.push_back(simplify(deriv(deriv(ast, j), k))); // ∂²F/∂θ_j∂θ_k

    // 布局: [P, n_seg, 段...]（与 compileCustomExpr / evalCustomAll 一致）
    std::vector<double> aux;
    aux.push_back((double)P);
    aux.push_back((double)segs.size());
    for (const auto& seg_tree : segs) {
        std::vector<double> seg;
        compileNode(seg_tree, seg);
        aux.push_back((double)(seg.size() / 3));
        aux.insert(aux.end(), seg.begin(), seg.end());
    }
    return aux;
}
