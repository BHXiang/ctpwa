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
            if (n.kids[0].type == NodeType::Var) {
                // L 为 CVAR_L（buildModelAST 顶点 Bf: 内核运行时逐波 L）:
                // 编译期无法展开 bfPoly → 用运行期对数导数黑盒
                // ∂lnBf/∂q  = l_q = MODEL_BF_DLNQ, ∂lnBf/∂q0 = l_q0 = MODEL_BF_DLNQ0
                Node lq = Node::makeComposite(MODEL_BF_DLNQ, n.kids);
                Node lq0 = Node::makeComposite(MODEL_BF_DLNQ0, n.kids);
                Node dBf_dq = astMul(n, lq);
                Node dBf_dq0 = astMul(n, lq0);
                // ∂Bf/∂d = (q/d)·Bf·l_q + (q0/d)·Bf·l_q0
                Node dBf_dd = astAdd(
                    astMul(astMul(dBf_dq, q), astDiv(astNum(1.0), d)),
                    astMul(astMul(dBf_dq0, q0), astDiv(astNum(1.0), d)));
                return astAdd(astAdd(astMul(dBf_dq, deriv(q, p)),
                                     astMul(dBf_dq0, deriv(q0, p))),
                              astMul(dBf_dd, deriv(d, p)));
            }
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
        case MODEL_FLATTE_RHO_RE:
        case MODEL_FLATTE_RHO_IM:
            // ρ_i 仅依赖 s(=m²) 和道质量（常数），与拟合参数无关
            return astNum(0.0);
        case MODEL_FLATTE:
            // 由 buildModelAST 分解为 RHO_RE/IM + 算术 → deriv 自动处理
            return astNum(0.0);
        case MODEL_ONE:
            return astNum(0.0);
        default:
            return astNum(0.0);
    }
}

// ============================================================================
// 模型注册表二阶导：∂²C/∂θ_p∂θ_q
//
// 实现策略：用复合节点的子节点重建其定义表达式，再对重建树做 deriv2()
// 结构性二阶微分。重建树里嵌套的复合节点（BWR 的 Bf、q0 的 breakup 链）
// 经 deriv2 → modelDeriv2 递归，逐层紧凑展开——相比「对一阶导展开式
// 再微分」（deriv(deriv())）不会二次膨胀（实测 BWR d²F 段 26306 → 数百指令）。
// ============================================================================

Node modelDeriv2(int composite_id, const Node& n, int p, int q)
{
    switch (composite_id) {
        case MODEL_BREAKUP_Q0: {
            // args: [m0, m1, m2]; q0 = sqrt(A·B)/(2m0),
            // A = m0²-(m1+m2)², B = m0²-(m1-m2)²
            Node m0 = n.kids[0], m1 = n.kids[1], m2 = n.kids[2];
            Node s = astAdd(m1, m2), dd = astSub(m1, m2);
            Node A = astSub(astSq(m0), astSq(s));
            Node B = astSub(astSq(m0), astSq(dd));
            Node q0def = astDiv(Node::makeFunc(CFUNC_SQRT, astMul(A, B)),
                                astMul(astNum(2.0), m0));
            return deriv2(q0def, p, q);
        }
        case MODEL_BF: {
            // args: [L, q, q0, d]; Bf = sqrt(N_L(q0·d)/N_L(q·d))
            Node qarg = n.kids[1], q0 = n.kids[2], d = n.kids[3];
            if (n.kids[0].type == NodeType::Var) {
                // L 为 CVAR_L（运行时逐波）→ 运行期对数导数黑盒:
                // d²Bf/dp·dq = Bf·[Σ_v l_v·v''_pq + Σ_{v,w}(l_v·l_w + l_vw)·v'_p·w'_q]
                // v,w ∈ {q, q0}; d 为 Num 常量；l_qq0 = ∂²lnBf/∂q∂q0 = 0
                Node BfN = n;
                Node lq = Node::makeComposite(MODEL_BF_DLNQ, n.kids);
                Node lq0 = Node::makeComposite(MODEL_BF_DLNQ0, n.kids);
                Node lqq = Node::makeComposite(MODEL_BF_D2LNQQ, n.kids);
                Node lq0q0 = Node::makeComposite(MODEL_BF_D2LNQ0Q0, n.kids);
                Node qp = deriv(qarg, p), qq = deriv(qarg, q);
                Node q0p = deriv(q0, p), q0q = deriv(q0, q);
                Node qppq = deriv2(qarg, p, q), q0ppq = deriv2(q0, p, q);
                Node d2 = astAdd(astMul(lq, qppq), astMul(lq0, q0ppq));
                // q 方向: (l_q² + l_qq)·q'_p·q'_q（q=CVAR_Q → 该项简化为 0）
                d2 = astAdd(d2, astMul(astAdd(astSq(lq), lqq), astMul(qp, qq)));
                // q0 方向: (l_q0² + l_q0q0)·q0'_p·q0'_q
                d2 = astAdd(d2, astMul(astAdd(astSq(lq0), lq0q0), astMul(q0p, q0q)));
                // 交叉项: l_q·l_q0·(q'_p·q0'_q + q0'_p·q'_q)
                d2 = astAdd(d2, astMul(astMul(lq, lq0),
                                       astAdd(astMul(qp, q0q), astMul(q0p, qq))));
                return astMul(BfN, d2);
            }
            Node z = astMul(qarg, d), z0 = astMul(q0, d);
            Node bfdef = Node::makeFunc(CFUNC_SQRT,
                astDiv(bfPoly((int)n.kids[0].num, z0), bfPoly((int)n.kids[0].num, z)));
            return deriv2(bfdef, p, q);
        }
        case MODEL_BW:
        case MODEL_BWR: {
            // args: BW [m, m0, g0]; BWR [m, m0, g0, L, q, q0, d]
            // F = x/s + i·y/s; x = m0²-m², s = x²+y²
            //   BW:  y = m0·g0
            //   BWR: y = m0·γ, γ = g0·(q/q0)^(2L+1)·(m0/m)·Bf²
            // 二阶导闭式（对数微分 + 二阶商法则模板）：
            //   d²(x/s) = x''/s − x·s''/s² − (x'_p·s'_q + x'_q·s'_p)/s²
            //             + 2·x·s'_p·s'_q/s³
            // 交叉项（p≠q）必须用 p、q 两个方向的组合（x'_p·s'_q + x'_q·s'_p，
            // s'_p·s'_q），不能只用 p 方向——那仅对对角段（p==q）成立。
            // 与 modelDeriv 一阶规则同风格（dγ = γ·U 对数微分），
            // d²γ = γ·(U_p·U_q + U_pq)——避免 deriv(deriv()) 的二次膨胀。
            bool is_bwr = (composite_id == MODEL_BWR);
            Node m = n.kids[0], m0 = n.kids[1], g0 = n.kids[2];
            Node m0p = deriv(m0, p), m0q = deriv(m0, q);
            Node g0p = deriv(g0, p), g0q = deriv(g0, q);
            Node gamma, dgp, dgq, d2gamma;
            if (is_bwr) {
                int L = (int)n.kids[3].num;
                Node qarg = n.kids[4], q0 = n.kids[5], d = n.kids[6];
                Node BfN = Node::makeComposite(MODEL_BF, {n.kids[3], qarg, q0, d});
                Node q0p = deriv(q0, p), q0q = deriv(q0, q);
                Node d2q0 = deriv2(q0, p, q);
                Node Bfp = modelDeriv(MODEL_BF, BfN, p);
                Node Bfq = modelDeriv(MODEL_BF, BfN, q);
                Node d2Bf = modelDeriv2(MODEL_BF, BfN, p, q);
                Node pw = Node::makeOp(NodeType::Pow, astDiv(qarg, q0),
                                       astNum(2.0 * L + 1));
                gamma = astMul(astMul(astMul(g0, pw), astDiv(m0, m)), astSq(BfN));
                // U_p = dγ/γ 的 p 方向 = dg0/g0 + dm0/m0 − (2L+1)·dq0/q0 + 2·dBf/Bf
                //（q、m 为 Var，导数 0）；U_q 同理换 q 方向
                auto logDeriv = [&](const Node& dg, const Node& dm,
                                    const Node& dq, const Node& db) {
                    return astAdd(astSub(astAdd(astDiv(dg, g0), astDiv(dm, m0)),
                                         astMul(astNum(2.0 * L + 1), astDiv(dq, q0))),
                                  astMul(astNum(2.0), astDiv(db, BfN)));
                };
                Node Up = logDeriv(g0p, m0p, q0p, Bfp);
                Node Uq = logDeriv(g0q, m0q, q0q, Bfq);
                // U_pq = −dg0_p·dg0_q/g0² − dm0_p·dm0_q/m0²
                //        + (2L+1)(dq0_p·dq0_q/q0² − d²q0/q0)
                //        + 2·(d²Bf/Bf − dBf_p·dBf_q/Bf²)
                Node Upq = astAdd(astAdd(astNeg(astDiv(astMul(g0p, g0q), astSq(g0))),
                                         astNeg(astDiv(astMul(m0p, m0q), astSq(m0)))),
                                  astAdd(astMul(astNum(2.0 * L + 1),
                                                astSub(astDiv(astMul(q0p, q0q), astSq(q0)),
                                                       astDiv(d2q0, q0))),
                                         astMul(astNum(2.0),
                                                astSub(astDiv(d2Bf, BfN),
                                                       astDiv(astMul(Bfp, Bfq), astSq(BfN))))));
                dgp = astMul(gamma, Up);
                dgq = astMul(gamma, Uq);
                d2gamma = astMul(gamma, astAdd(astMul(Up, Uq), Upq));
            } else {
                gamma = g0;   // BW: y = m0·g0（复用下方 d2y 公式，dγ=dg, d²γ=0）
                dgp = g0p;
                dgq = g0q;
                d2gamma = astNum(0.0);
            }
            Node x = astSub(astSq(m0), astSq(m));
            Node y = astMul(m0, gamma);
            Node s = astAdd(astSq(x), astSq(y));
            Node s2 = astSq(s), s3 = astMul(s2, s);
            // x = m0² − m²: x'_p = 2·m0·m0'_p, x'' = 2·m0'_p·m0'_q（m0 为 Param，二阶 0）
            Node xp = astMul(astMul(astNum(2.0), m0), m0p);
            Node xq = astMul(astMul(astNum(2.0), m0), m0q);
            Node d2x = astMul(astNum(2.0), astMul(m0p, m0q));
            // y = m0·γ: y'_p = m0'_p·γ + m0·γ'_p
            Node yp = astAdd(astMul(m0p, gamma), astMul(m0, dgp));
            Node yq = astAdd(astMul(m0q, gamma), astMul(m0, dgq));
            Node d2y = astAdd(astAdd(astMul(m0p, dgq), astMul(m0q, dgp)),
                              astMul(m0, d2gamma));
            // s = x² + y²: s'_p = 2x·x'_p + 2y·y'_p
            Node sp = astAdd(astMul(astMul(astNum(2.0), x), xp),
                             astMul(astMul(astNum(2.0), y), yp));
            Node sq = astAdd(astMul(astMul(astNum(2.0), x), xq),
                             astMul(astMul(astNum(2.0), y), yq));
            Node d2s = astAdd(astAdd(astMul(astNum(2.0), astMul(xp, xq)),
                                     astMul(astMul(astNum(2.0), x), d2x)),
                              astAdd(astMul(astNum(2.0), astMul(yp, yq)),
                                     astMul(astMul(astNum(2.0), y), d2y)));
            // 商法则最后一项是 +2x·s_p·s_q/s³（注意不是减号；减号会导致
            // θθ Hessian 块完全错误——旧手写实现带入，曾用 FD 交叉验证发现）
            Node d2re = astSub(astSub(astAdd(astDiv(d2x, s),
                                             astDiv(astMul(astMul(astNum(2.0), x), astMul(sp, sq)), s3)),
                                      astDiv(astMul(x, d2s), s2)),
                               astDiv(astAdd(astMul(xp, sq), astMul(xq, sp)), s2));
            Node d2im = astSub(astSub(astAdd(astDiv(d2y, s),
                                             astDiv(astMul(astMul(astNum(2.0), y), astMul(sp, sq)), s3)),
                                      astDiv(astMul(y, d2s), s2)),
                               astDiv(astAdd(astMul(yp, sq), astMul(yq, sp)), s2));
            return astAdd(d2re, astMul(astI(), d2im));
        }
        case MODEL_FLATTE_RHO_RE:
        case MODEL_FLATTE_RHO_IM:
        case MODEL_FLATTE:
        case MODEL_ONE:
            // 无参数依赖（或已在 AST 层分解）→ 二阶导为 0
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
    int L, int Lmin, double d,
    int P,                              // 自由参数数
    int n_channels,                     // Flatte: 道数; 其他: 0
    const std::vector<double>& channel_masses,  // Flatte: [m_a0,m_b0, m_a1,m_b1, ...]
    Q0MassDep md1_dep, double md1_fixed,
    Q0MassDep md2_dep, double md2_fixed,
    bool has_bf)                        // 目标节点是否有势垒因子（false 时 AST 不含 Bf）
{
    // q0 链辅助：把质量依赖描述转成 AST 节点
    auto massNode = [&](Q0MassDep dep, double val, int cv) -> Node {
        switch (dep) {
            case Q0MassDep::FixedMass: return Node::makeNum(val);
            case Q0MassDep::M0Param:   return Node::makeParam(0);
            default:                   return Node::makeVar(cv);
        }
    };
    // q0 = breakup(m0, md1, md2) 复合节点（导数规则见 modelDeriv MODEL_BREAKUP_Q0）
    // m0: 目标共振态质量（BWR/BW/Flatte 是参数 0；ONE 无参数 → 1.0 回退）
    Node m0_ast = (P > 0) ? Node::makeParam(0) : Node::makeNum(1.0);
    Node q0_ast = Node::makeComposite(MODEL_BREAKUP_Q0, {
        m0_ast,
        massNode(md1_dep, md1_fixed, CVAR_MD1),
        massNode(md2_dep, md2_fixed, CVAR_MD2)
    });

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
            // F = MODEL_BWR(m, θ0, θ1, Lmin, q, q0, d) × [MODEL_BF(L_runtime, q, q0, d)]
            // （has_bf=false 时不含 Bf 因子）
            // has_bf=false 时传播子内部宽度也不含 Bf（MODEL_BWR 的 gamma 含 Bf²）
            // → d kid 传 0.0 使 Bf≡1，与 bf_d=0.0 完全等价
            // 宽度约定与 computeNodeFactor 一致: BWR Γ(Lmin) 与波无关恒用节点
            // SL 列表最小 L（烘焙）; 顶点 Bf 用 CVAR_L（内核运行时逐波 sl.L）
            Node bwr = Node::makeComposite(MODEL_BWR, {
                Node::makeVar(CVAR_M),
                Node::makeParam(0),       // m0
                Node::makeParam(1),       // g0
                Node::makeNum((double)Lmin),
                Node::makeVar(CVAR_Q),
                q0_ast,
                Node::makeNum(has_bf ? d : 0.0)
            });
            if (has_bf) {
                Node bf = Node::makeComposite(MODEL_BF, {
                    Node::makeVar(CVAR_L),
                    Node::makeVar(CVAR_Q),
                    q0_ast,
                    Node::makeNum(d)
                });
                ast = Node::makeOp(NodeType::Mul, bwr, bf);
            } else {
                ast = bwr;
            }
            break;
        }
        case ResModelType::ONE: {
            // F = [MODEL_BF(L_runtime, q, q0, d)]（has_bf=false 时 F = 1）
            // 顶点 Bf 用 CVAR_L: 与 computeNodeFactor 的 Bf(sl.L, ...) 逐波一致
            if (has_bf) {
                ast = Node::makeComposite(MODEL_BF, {
                    Node::makeVar(CVAR_L),
                    Node::makeVar(CVAR_Q),
                    q0_ast,
                    Node::makeNum(d)
                });
            } else {
                ast = astNum(1.0);
            }
            break;
        }
        case ResModelType::Flatte: {
            // Flatte 分解: F = (A - iB)/D × Bf（展开见 buildFlatteAST，
            // ρ_i 仅依赖 s=m² 和道质量 → 导数 0 → 链式法则自动处理）
            std::vector<Node> gs, chs;
            for (int i = 0; i < n_channels; ++i) {
                gs.push_back(Node::makeParam(1 + i));
                chs.push_back(astNum(channel_masses[2 * i]));
                chs.push_back(astNum(channel_masses[2 * i + 1]));
            }
            Node flatte = buildFlatteAST(Node::makeVar(CVAR_M),
                                         Node::makeParam(0), gs, chs);

            // （has_bf=false 时不含 Bf 因子）; 顶点 Bf L 用运行时逐波（CVAR_L）
            if (has_bf) {
                Node bf = Node::makeComposite(MODEL_BF, {
                    Node::makeVar(CVAR_L),
                    Node::makeVar(CVAR_Q),
                    q0_ast,
                    Node::makeNum(d)
                });
                ast = Node::makeOp(NodeType::Mul, flatte, bf);
            } else {
                ast = flatte;
            }
            break;
        }
        case ResModelType::GS: {
            // Gounaris-Sakurai for ρ→ππ: F = (1 + d·g₀/m₀) / (m₀² - m² + f - i·m₀·Γ) × Bf
            // Decomposed into MODEL_BREAKUP_Q0 (q₀) + arithmetic (h, d, h₀, h'₀, Γ, f)
            double mpi = (n_channels > 0) ? channel_masses[0] : 0.1396;
            double pi = 3.14159265358979323846;

            Node m0 = Node::makeParam(0);
            Node g0 = Node::makeParam(1);
            Node m  = Node::makeVar(CVAR_M);
            Node q  = Node::makeVar(CVAR_Q);
            // q0_ast = MODEL_BREAKUP_Q0(m0, md1, md2) — 已在上方构造，对等质量 pion 导数正确

            auto astLog = [](Node a) { return Node::makeFunc(CFUNC_LOG, a); };
            // h(m) = (2/π)*(q/m)*log((m+2q)/(2mpi))
            Node hm = astMul(astNum(2.0/pi),
                astMul(astDiv(q, m),
                    astLog(astDiv(astAdd(m, astMul(astNum(2.0), q)),
                                  astNum(2.0*mpi)))));

            // d = (3/π)*(mpi²/q0²)*log((m0+2q0)/(2mpi)) + m0/(2πq0) - mpi²*m0/(πq0³)
            auto astPow = [](Node a, int n) {
                Node r = a;
                for (int i = 1; i < n; ++i) r = astMul(r, a);
                return r;
            };
            Node q0_2 = astSq(q0_ast);
            Node q0_3 = astMul(q0_2, q0_ast);
            Node mpi2 = astNum(mpi * mpi);
            Node logArg = astDiv(astAdd(m0, astMul(astNum(2.0), q0_ast)), astNum(2.0*mpi));
            Node dval = astSub(
                astAdd(astMul(astMul(astNum(3.0/pi), astDiv(mpi2, q0_2)), astLog(logArg)),
                       astDiv(m0, astMul(astNum(2.0*pi), q0_ast))),
                astDiv(astMul(mpi2, m0), astMul(astNum(pi), q0_3)));

            // h0 = (2/π)*(q0/m0)*log((m0+2q0)/(2mpi))
            Node h0 = astMul(astNum(2.0/pi),
                astMul(astDiv(q0_ast, m0), astLog(logArg)));

            // dh0 = h0*(1/(8q0²) - 1/(2m0²)) + 1/(2π·m0²)
            Node dh0 = astAdd(
                astMul(h0, astSub(astDiv(astNum(1.0), astMul(astNum(8.0), q0_2)),
                                  astDiv(astNum(1.0), astMul(astNum(2.0), astSq(m0))))),
                astDiv(astNum(1.0), astMul(astNum(2.0*pi), astSq(m0))));

            // Γ(m) = g0 * (q/q0)³ * (m0/m)
            Node qratio = astDiv(q, q0_ast);
            Node gam = astMul(astMul(g0, astPow(qratio, 3)), astDiv(m0, m));

            // f(m) = g0 * m0²/q0³ * (q²*(hm-h0) + (m0²-m²)*q0²*dh0)
            Node f = astMul(astMul(g0, astDiv(astSq(m0), q0_3)),
                astAdd(astMul(astSq(q), astSub(hm, h0)),
                       astMul(astMul(astSub(astSq(m0), astSq(m)), q0_2), dh0)));

            // Num = 1 + d * g0 / m0
            Node Num = astAdd(astNum(1.0), astDiv(astMul(dval, g0), m0));

            // DenRe = m0² - m² + f, DenIm = -m0 * gam
            Node DenRe = astAdd(astSub(astSq(m0), astSq(m)), f);
            Node DenIm = astNeg(astMul(m0, gam));
            Node DenSq = astAdd(astSq(DenRe), astSq(DenIm));

            // F = Num / (DenRe + i*DenIm)
            //   = Num*DenRe/DenSq + i*(-Num*DenIm/DenSq)
            Node Fre_node = astDiv(astMul(Num, DenRe), DenSq);
            Node Fim_node = astNeg(astDiv(astMul(Num, DenIm), DenSq));
            Node gs = astAdd(Fre_node, astMul(Fim_node, astI()));

            // （has_bf=false 时不含 Bf 因子）; 顶点 Bf L 用运行时逐波（CVAR_L）
            if (has_bf) {
                Node bf = Node::makeComposite(MODEL_BF, {
                    Node::makeVar(CVAR_L), q, q0_ast, astNum(d)
                });
                ast = Node::makeOp(NodeType::Mul, gs, bf);
            } else {
                ast = gs;
            }
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
            // 直接二阶微分（deriv(deriv()) 会二次膨胀，BWR d²F 段实测 2.6 万指令）
            segs.push_back(simplify(deriv2(ast, j, k))); // ∂²F/∂θ_j∂θ_k

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
