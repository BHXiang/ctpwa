#include "CustomExpr.cuh"
#include "SymbolicDiff.cuh"
#include "ResModel.cuh"   // COP_MODEL: Bf/BW/BWR/Flatte/computeQ0AD（double 实例化）

#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

// ============================================================================
// 复数运算（host 端，符号微分用）
// ============================================================================

namespace {

struct C {
    double re, im;
    C(double r = 0.0, double i = 0.0) : re(r), im(i) {}
    C operator+(const C& o) const { return {re + o.re, im + o.im}; }
    C operator-(const C& o) const { return {re - o.re, im - o.im}; }
    C operator-() const { return {-re, -im}; }
    C operator*(const C& o) const {
        return {re * o.re - im * o.im, re * o.im + im * o.re};
    }
    C operator/(const C& o) const {
        double s = o.re * o.re + o.im * o.im;
        if (s == 0.0) s = 1e-30;
        return {(re * o.re + im * o.im) / s, (im * o.re - re * o.im) / s};
    }
    bool operator==(const C& o) const { return re == o.re && im == o.im; }
    double abs() const { return std::sqrt(re * re + im * im); }
};

C cexp(const C& z) {
    double e = std::exp(z.re);
    return {e * std::cos(z.im), e * std::sin(z.im)};
}
C clog(const C& z) {
    double a = z.abs();
    if (a == 0.0) a = 1e-30;
    return {std::log(a), std::atan2(z.im, z.re)};
}
C csin(const C& z) {
    double e = std::exp(z.im), em = std::exp(-z.im);
    return {std::sin(z.re) * (e + em) / 2, std::cos(z.re) * (e - em) / 2};
}
C ccos(const C& z) {
    double e = std::exp(z.im), em = std::exp(-z.im);
    return {std::cos(z.re) * (e + em) / 2, -std::sin(z.re) * (e - em) / 2};
}
C csqrt_c(const C& z) {
    double a = z.abs();
    double r = std::sqrt((a + z.re) / 2);
    double i = std::sqrt((a - z.re) / 2) * (z.im < 0 ? -1.0 : 1.0);
    return {r, i};
}
C cpow(const C& b, const C& e) {
    // b^e = exp(e * log(b))
    return cexp(e * clog(b));
}

// ============================================================================
// 求值（host 参考实现，测试用）
// ============================================================================

C evalNode(const Node& n, double m, double q, double q0, int L, double d,
           const std::vector<C>& params)
{
    switch (n.type) {
        case NodeType::Num: return {n.num, 0.0};
        case NodeType::Var:
            switch (n.var_id) {
                case CVAR_M: return {m, 0.0};
                case CVAR_Q: return {q, 0.0};
                case CVAR_Q0: return {q0, 0.0};
                case CVAR_L: return {(double)L, 0.0};
                case CVAR_D: return {d, 0.0};
            }
            return {0, 0};
        case NodeType::Param: return params[n.param_id];
        case NodeType::Add: return evalNode(n.kids[0], m, q, q0, L, d, params) + evalNode(n.kids[1], m, q, q0, L, d, params);
        case NodeType::Sub: return evalNode(n.kids[0], m, q, q0, L, d, params) - evalNode(n.kids[1], m, q, q0, L, d, params);
        case NodeType::Mul: return evalNode(n.kids[0], m, q, q0, L, d, params) * evalNode(n.kids[1], m, q, q0, L, d, params);
        case NodeType::Div: return evalNode(n.kids[0], m, q, q0, L, d, params) / evalNode(n.kids[1], m, q, q0, L, d, params);
        case NodeType::Pow: return cpow(evalNode(n.kids[0], m, q, q0, L, d, params), evalNode(n.kids[1], m, q, q0, L, d, params));
        case NodeType::Neg: return -evalNode(n.kids[0], m, q, q0, L, d, params);
        case NodeType::Func: {
            C a = evalNode(n.kids[0], m, q, q0, L, d, params);
            switch (n.func_id) {
                case CFUNC_EXP: return cexp(a);
                case CFUNC_LOG: return clog(a);
                case CFUNC_SIN: return csin(a);
                case CFUNC_COS: return ccos(a);
                case CFUNC_SQRT: return csqrt_c(a);
                case CFUNC_ABS: return {a.abs(), 0.0};
                case CFUNC_RE: return {a.re, 0.0};
                case CFUNC_IM: return {a.im, 0.0};
                case CFUNC_CONJ: return {a.re, -a.im};
                case CFUNC_CSQRT: return csqrt_c(a);
            }
            return {0, 0};
        }
        case NodeType::Composite: {
            // 黑盒模型求值（host 参考实现，与 COP_MODEL 指令一致）
            auto ev = [&](const Node& k) { return evalNode(k, m, q, q0, L, d, params); };
            switch (n.composite_id) {
                case MODEL_BREAKUP_Q0: {  // [m0, m1, m2]
                    double m0v = ev(n.kids[0]).re, m1v = ev(n.kids[1]).re, m2v = ev(n.kids[2]).re;
                    return {computeQ0AD(m0v, m1v, m2v), 0.0};
                }
                case MODEL_BF: {  // [L, q, q0, d]
                    double qv = ev(n.kids[1]).re, q0v = ev(n.kids[2]).re, dv = ev(n.kids[3]).re;
                    return {Bf<double>((int)n.kids[0].num, qv, q0v, dv), 0.0};
                }
                case MODEL_BW: {  // [m, m0, g0]
                    double mv = ev(n.kids[0]).re, m0v = ev(n.kids[1]).re, g0v = ev(n.kids[2]).re;
                    auto z = BW<double>(mv, m0v, g0v);
                    return {z.real(), z.imag()};
                }
                case MODEL_BWR: {  // [m, m0, g0, L, q, q0, d]
                    double mv = ev(n.kids[0]).re, m0v = ev(n.kids[1]).re, g0v = ev(n.kids[2]).re;
                    double qv = ev(n.kids[4]).re, q0v = ev(n.kids[5]).re, dv = ev(n.kids[6]).re;
                    auto z = BWR<double>(mv, m0v, g0v, (int)n.kids[3].num, qv, q0v, dv);
                    return {z.real(), z.imag()};
                }
                case MODEL_FLATTE: {  // [m, m0, g0..g_{n-1}, (ma,mb)0..]
                    double mv = ev(n.kids[0]).re, m0v = ev(n.kids[1]).re;
                    int n_ch = ((int)n.kids.size() - 2) / 3;
                    double g[4], ch[8];
                    for (int i = 0; i < n_ch; ++i) g[i] = ev(n.kids[2 + i]).re;
                    for (int i = 0; i < 2 * n_ch; ++i) ch[i] = ev(n.kids[2 + n_ch + i]).re;
                    auto z = Flatte<double>(mv, m0v, n_ch, g, ch);
                    return {z.real(), z.imag()};
                }
                case MODEL_FLATTE_RHO_RE:
                case MODEL_FLATTE_RHO_IM: {  // [s, ma, mb]
                    double sv = ev(n.kids[0]).re, mav = ev(n.kids[1]).re, mbv = ev(n.kids[2]).re;
                    double S = (mav + mbv) * (mav + mbv);
                    double D = (mav - mbv) * (mav - mbv);
                    double f1_sq = 1.0 - S / sv;
                    double f1_re, f1_im;
                    if (f1_sq >= 0.0) { f1_re = sqrt(f1_sq); f1_im = 0.0; }
                    else              { f1_re = 0.0; f1_im = sqrt(-f1_sq); }
                    double f2_sq = 1.0 - D / sv;
                    double f2 = (f2_sq > 0.0) ? sqrt(f2_sq) : 0.0;
                    if (n.composite_id == MODEL_FLATTE_RHO_RE)
                        return {f1_re * f2, 0.0};
                    else
                        return {f1_im * f2, 0.0};
                }
                case MODEL_ONE:
                    return {1.0, 0.0};
            }
            return {0, 0};
        }
    }
    return {0, 0};
}

// ============================================================================
// 解析器（tokenizer + 递归下降）
// ============================================================================

struct Token {
    enum Type { Num, Ident, Op, LParen, RParen, Comma, End } type;
    double num = 0.0;
    std::string str;
};

std::vector<Token> tokenize(const std::string& s)
{
    std::vector<Token> toks;
    size_t i = 0;
    while (i < s.size()) {
        char c = s[i];
        if (isspace(c)) { ++i; continue; }
        if (isdigit(c) || c == '.') {
            // 虚数单位: 1j / 1i
            if (s[i] == '1' && i + 1 < s.size() && (s[i + 1] == 'j' || s[i + 1] == 'i')) {
                toks.push_back({Token::Ident, 0.0, "1j"});
                i += 2;
                continue;
            }
            size_t j = i;
            while (j < s.size() && (isdigit(s[j]) || s[j] == '.' || s[j] == 'e' || s[j] == 'E' ||
                   ((s[j] == '+' || s[j] == '-') && j > i && (s[j-1] == 'e' || s[j-1] == 'E')))) ++j;
            toks.push_back({Token::Num, std::stod(s.substr(i, j - i)), ""});
            i = j;
            continue;
        }
        if (isalpha(c) || c == '_') {
            size_t j = i;
            while (j < s.size() && (isalnum(s[j]) || s[j] == '_')) ++j;
            std::string id = s.substr(i, j - i);
            // 复数单位: "1j" 或 "i" 或 "1i"
            if (id == "1j" || id == "1i") {
                toks.push_back({Token::Ident, 0.0, "1j"});
            } else {
                toks.push_back({Token::Ident, 0.0, id});
            }
            i = j;
            continue;
        }
        if (c == '+' || c == '-' || c == '*' || c == '/' || c == '^') {
            toks.push_back({Token::Op, 0.0, std::string(1, c)});
            ++i; continue;
        }
        if (c == '(') { toks.push_back({Token::LParen, 0.0, ""}); ++i; continue; }
        if (c == ')') { toks.push_back({Token::RParen, 0.0, ""}); ++i; continue; }
        if (c == ',') { toks.push_back({Token::Comma, 0.0, ""}); ++i; continue; }
        throw std::runtime_error("CustomExpr: unexpected char '" + std::string(1, c) + "'");
    }
    toks.push_back({Token::End, 0.0, ""});
    return toks;
}

class Parser {
  public:
    Parser(const std::vector<Token>& toks,
           const std::map<std::string, int>& params)
        : toks_(toks), params_(params), pos_(0) {}

    Node parse()
    {
        Node n = parseExpr();
        if (toks_[pos_].type != Token::End)
            throw std::runtime_error("CustomExpr: trailing tokens");
        return n;
    }

  private:
    const std::vector<Token>& toks_;
    const std::map<std::string, int>& params_;
    size_t pos_;

    const Token& peek() const { return toks_[pos_]; }
    const Token& next() { return toks_[pos_++]; }

    Node parseExpr()
    {
        Node left = parseTerm();
        while (peek().type == Token::Op &&
               (peek().str == "+" || peek().str == "-")) {
            std::string op = next().str;
            Node right = parseTerm();
            left = Node::makeOp(op == "+" ? NodeType::Add : NodeType::Sub, left, right);
        }
        return left;
    }

    Node parseTerm()
    {
        Node left = parseFactor();
        while (peek().type == Token::Op &&
               (peek().str == "*" || peek().str == "/")) {
            std::string op = next().str;
            Node right = parseFactor();
            left = Node::makeOp(op == "*" ? NodeType::Mul : NodeType::Div, left, right);
        }
        return left;
    }

    Node parseFactor()
    {
        // 幂是右结合的: a^b^c = a^(b^c)
        Node base = parseUnary();
        if (peek().type == Token::Op && peek().str == "^") {
            next();
            Node exp = parseFactor();
            return Node::makeOp(NodeType::Pow, base, exp);
        }
        return base;
    }

    Node parseUnary()
    {
        if (peek().type == Token::Op && peek().str == "-") {
            next();
            return Node::makeUnary(NodeType::Neg, parseUnary());
        }
        if (peek().type == Token::Op && peek().str == "+") {
            next();
            return parseUnary();
        }
        return parsePrimary();
    }

    // 解析括号内参数表达式列表: '(' expr (',' expr)* ')'
    std::vector<Node> parseArgList()
    {
        if (peek().type != Token::LParen)
            throw std::runtime_error("CustomExpr: function needs (");
        next();
        std::vector<Node> args;
        if (peek().type == Token::RParen) { next(); return args; }
        while (true) {
            args.push_back(parseExpr());
            if (peek().type == Token::Comma) { next(); continue; }
            break;
        }
        if (peek().type != Token::RParen)
            throw std::runtime_error("CustomExpr: missing )");
        next();
        return args;
    }

    // q0 复合节点：m0 = 第一个参数（与内核运行时 m0_q0 = all_params[param_offset]
    // 一致）；无参数时回退 1.0。md1/md2 为运行时子粒子质量（导数 0）。
    Node q0Node()
    {
        return Node::makeComposite(MODEL_BREAKUP_Q0, {
            params_.empty() ? Node::makeNum(1.0) : Node::makeParam(0),
            Node::makeVar(CVAR_MD1),
            Node::makeVar(CVAR_MD2)});
    }

    Node parsePrimary()
    {
        const Token& t = peek();
        if (t.type == Token::Num) {
            next();
            return Node::makeNum(t.num);
        }
        if (t.type == Token::Ident) {
            next();
            if (t.str == "1j") {
                // 虚数单位: 用 0+1j 表示 —— 以 csqrt(-1) 构造? 用 Num 0 + 特殊?
                // 直接构造一个 "虚数" —— 用一个固定函数? 简化: pi 常数 + 1j 用 Num(-1) 的 csqrt?
                // 1j 无法用纯 AST 表达 → 引入 COP_PUSH_I 指令 → AST 用特殊标记:
                // 用 Var? 不行。用 Func CSQRT(-1)? csqrt(-1)=j ✓ 但表达式膨胀。
                // 更简单: 1j 作为独立的 Node 类型? 当前 AST 无此类型。
                // 方案: 用 csqrt(Num(-1)) 表示 1j —— 求值/微分都正确 (csqrt(-1)=j)
                return Node::makeFunc(CFUNC_CSQRT, Node::makeNum(-1.0));
            }
            if (t.str == "pi") return Node::makeNum(3.14159265358979323846);
            // 变量
            if (t.str == "m") return Node::makeVar(CVAR_M);
            if (t.str == "q") return Node::makeVar(CVAR_Q);
            if (t.str == "q0" && peek().type != Token::LParen)
                // q0 变量 = breakup(θ_0, md1, md2)：标称质量取第一个参数
                // （与内核运行时 m0_q0 = all_params[param_offset] 一致），
                // 因此 ∂q0/∂θ_0 ≠ 0 —— 用 MODEL_BREAKUP_Q0 复合节点追踪
                // 导数，而非当作常数变量（旧行为导数缺失 ∂F/∂q0·∂q0/∂θ_0）。
                // q0(...) 带括号时是函数调用（见下方模型函数分支）
                return q0Node();
            if (t.str == "L") return Node::makeVar(CVAR_L);
            if (t.str == "d") return Node::makeVar(CVAR_D);
            if (t.str == "p1p") return Node::makeVar(CVAR_P1P);
            if (t.str == "p1e") return Node::makeVar(CVAR_P1E);
            if (t.str == "p1costheta") return Node::makeVar(CVAR_P1COSTHETA);
            if (t.str == "p1phi") return Node::makeVar(CVAR_P1PHI);
            if (t.str == "p2p") return Node::makeVar(CVAR_P2P);
            if (t.str == "p2e") return Node::makeVar(CVAR_P2E);
            if (t.str == "p2costheta") return Node::makeVar(CVAR_P2COSTHETA);
            if (t.str == "p2phi") return Node::makeVar(CVAR_P2PHI);
            // 函数
            static const std::map<std::string, int> funcs = {
                {"exp", CFUNC_EXP}, {"log", CFUNC_LOG}, {"ln", CFUNC_LOG},
                {"sin", CFUNC_SIN}, {"cos", CFUNC_COS},
                {"sqrt", CFUNC_SQRT}, {"abs", CFUNC_ABS},
                {"Re", CFUNC_RE}, {"real", CFUNC_RE},
                {"Im", CFUNC_IM}, {"imag", CFUNC_IM},
                {"conj", CFUNC_CONJ}, {"csqrt", CFUNC_CSQRT},
            };
            auto fit = funcs.find(t.str);
            if (fit != funcs.end()) {
                if (peek().type != Token::LParen)
                    throw std::runtime_error("CustomExpr: function '" + t.str + "' needs (");
                next();
                Node arg = parseExpr();
                if (peek().type != Token::RParen)
                    throw std::runtime_error("CustomExpr: missing )");
                next();
                return Node::makeFunc(fit->second, arg);
            }
            // 模型函数（多参数）：BW / BWR / Bf / q0 / Flatte
            // 约定：质量/宽度/耦合常数传参数（p0, p1, ...），
            // m/q/q0/L/d 由包装内部补全，语义与内置模型完全一致
            if (t.str == "BW" || t.str == "BWR" || t.str == "Bf" ||
                t.str == "Flatte" ||
                (t.str == "q0" && peek().type == Token::LParen)) {
                std::vector<Node> args = parseArgList();
                if (t.str == "BW") {
                    // BW(m, m0, w0)：包装成 BW(m0, w0)，m 自动补全
                    if (args.size() != 2)
                        throw std::runtime_error(
                            "CustomExpr: BW(m0, w0) needs exactly 2 args");
                    return Node::makeComposite(MODEL_BW,
                        {Node::makeVar(CVAR_M), args[0], args[1]});
                }
                if (t.str == "BWR") {
                    // BWR(m, m0, g0, L, q, q0, d)：包装成 BWR(m0, g0[, L, d])
                    // L/d 缺省时与内置模型默认一致（L=1, d=3.0）；
                    // L/d 必须是字面量（modelDeriv 按 (int)num 取整）
                    if (args.size() < 2 || args.size() > 4)
                        throw std::runtime_error(
                            "CustomExpr: BWR(m0, g0[, L, d]) needs 2~4 args");
                    if (args.size() >= 3 && args[2].type != NodeType::Num)
                        throw std::runtime_error(
                            "CustomExpr: BWR L must be a literal (e.g. 1)");
                    if (args.size() >= 4 && args[3].type != NodeType::Num)
                        throw std::runtime_error(
                            "CustomExpr: BWR d must be a literal (e.g. 3.0)");
                    double Lv = (args.size() >= 3) ? args[2].num : 1.0;
                    double dv = (args.size() >= 4) ? args[3].num : 3.0;
                    return Node::makeComposite(MODEL_BWR, {
                        Node::makeVar(CVAR_M), args[0], args[1],
                        Node::makeNum(Lv), Node::makeVar(CVAR_Q), q0Node(),
                        Node::makeNum(dv)});
                }
                if (t.str == "Bf") {
                    // Bf(L, q, q0, d)：L 必须是字面量
                    if (args.size() != 4)
                        throw std::runtime_error(
                            "CustomExpr: Bf(L, q, q0, d) needs exactly 4 args");
                    if (args[0].type != NodeType::Num)
                        throw std::runtime_error(
                            "CustomExpr: Bf L must be a literal (e.g. 1)");
                    return Node::makeComposite(MODEL_BF, args);
                }
                if (t.str == "q0") {
                    // q0(x) → breakup(x, md1, md2)；q0(x, md1, md2) → 显式
                    if (args.size() == 1)
                        return Node::makeComposite(MODEL_BREAKUP_Q0,
                            {args[0], Node::makeVar(CVAR_MD1),
                             Node::makeVar(CVAR_MD2)});
                    if (args.size() == 3)
                        return Node::makeComposite(MODEL_BREAKUP_Q0, args);
                    throw std::runtime_error(
                        "CustomExpr: q0(x) or q0(x, md1, md2) expected");
                }
                // Flatte(m0, g0..g_{n-1}, (ma,mb)0..) = 1 + 3n 个参数
                // 展开为 AST（MODEL_FLATTE 复合节点的 modelDeriv 为 0，
                // 无法从 DSL 使用，须经 buildFlatteAST 展开成可微的树）
                if (t.str == "Flatte") {
                    if (args.size() < 4 || (args.size() - 1) % 3 != 0)
                        throw std::runtime_error(
                            "CustomExpr: Flatte(m0, g..., (ma,mb)...) needs "
                            "1+3n args (n>=1)");
                    int n_ch = (int)(args.size() - 1) / 3;
                    std::vector<Node> gs(args.begin() + 1,
                                         args.begin() + 1 + n_ch);
                    std::vector<Node> chs(args.begin() + 1 + n_ch, args.end());
                    return buildFlatteAST(Node::makeVar(CVAR_M), args[0],
                                          gs, chs);
                }
            }
            // 参数
            auto pit = params_.find(t.str);
            if (pit != params_.end())
                return Node::makeParam(pit->second);
            throw std::runtime_error("CustomExpr: unknown identifier '" + t.str + "'");
        }
        if (t.type == Token::LParen) {
            next();
            Node n = parseExpr();
            if (peek().type != Token::RParen)
                throw std::runtime_error("CustomExpr: missing )");
            next();
            return n;
        }
        throw std::runtime_error("CustomExpr: unexpected token");
    }
};

}  // namespace

// ============================================================================
// 符号微分（复数域链式法则，全局 API，见 SymbolicDiff.cuh）
// 对任意树 T 求 ∂/∂θ_p: m/q/q0/L/d 视为常数, 参数 θ_k 的导数为 δ_kp
// Composite 节点: dC/dθ = Σ_k (∂C/∂arg_k)·deriv(arg_k, p)，其中
// ∂C/∂arg_k 由模型注册的导数规则产生（注册见下，Phase 2 填充）
// ============================================================================

Node deriv(const Node& n, int p)
{
    switch (n.type) {
        case NodeType::Num:
        case NodeType::Var:
            return Node::makeNum(0.0);
        case NodeType::Param:
            return Node::makeNum(n.param_id == p ? 1.0 : 0.0);
        case NodeType::Add:
            return Node::makeOp(NodeType::Add, deriv(n.kids[0], p), deriv(n.kids[1], p));
        case NodeType::Sub:
            return Node::makeOp(NodeType::Sub, deriv(n.kids[0], p), deriv(n.kids[1], p));
        case NodeType::Mul:
            return Node::makeOp(NodeType::Add,
                Node::makeOp(NodeType::Mul, deriv(n.kids[0], p), n.kids[1]),
                Node::makeOp(NodeType::Mul, n.kids[0], deriv(n.kids[1], p)));
        case NodeType::Div:
            // (a'b - ab')/b²
            return Node::makeOp(NodeType::Div,
                Node::makeOp(NodeType::Sub,
                    Node::makeOp(NodeType::Mul, deriv(n.kids[0], p), n.kids[1]),
                    Node::makeOp(NodeType::Mul, n.kids[0], deriv(n.kids[1], p))),
                Node::makeOp(NodeType::Mul, n.kids[1], n.kids[1]));
        case NodeType::Pow: {
            // d(a^b) = a^b * (b'·log(a) + b·a'/a)
            Node da = deriv(n.kids[0], p), db = deriv(n.kids[1], p);
            Node loga = Node::makeFunc(CFUNC_LOG, n.kids[0]);
            Node term = Node::makeOp(NodeType::Add,
                Node::makeOp(NodeType::Mul, db, loga),
                Node::makeOp(NodeType::Mul, n.kids[1],
                    Node::makeOp(NodeType::Div, da, n.kids[0])));
            return Node::makeOp(NodeType::Mul, n, term);
        }
        case NodeType::Neg:
            return Node::makeUnary(NodeType::Neg, deriv(n.kids[0], p));
        case NodeType::Func: {
            Node a = n.kids[0], da = deriv(a, p);
            switch (n.func_id) {
                case CFUNC_EXP:
                    return Node::makeOp(NodeType::Mul, n, da);
                case CFUNC_LOG:
                    return Node::makeOp(NodeType::Div, da, a);
                case CFUNC_SIN:
                    return Node::makeOp(NodeType::Mul,
                        Node::makeFunc(CFUNC_COS, a), da);
                case CFUNC_COS:
                    return Node::makeUnary(NodeType::Neg,
                        Node::makeOp(NodeType::Mul,
                            Node::makeFunc(CFUNC_SIN, a), da));
                case CFUNC_SQRT:
                case CFUNC_CSQRT:
                    return Node::makeOp(NodeType::Div, da,
                        Node::makeOp(NodeType::Mul, Node::makeNum(2.0), n));
                case CFUNC_ABS: {
                    // d|z| = Re(conj(z)·dz)/|z|
                    Node conjz = Node::makeFunc(CFUNC_CONJ, a);
                    Node prod = Node::makeOp(NodeType::Mul, conjz, da);
                    return Node::makeOp(NodeType::Div,
                        Node::makeFunc(CFUNC_RE, prod), n);
                }
                case CFUNC_RE:
                    return Node::makeFunc(CFUNC_RE, da);
                case CFUNC_IM:
                    return Node::makeFunc(CFUNC_IM, da);
                case CFUNC_CONJ:
                    return Node::makeFunc(CFUNC_CONJ, da);
            }
            return Node::makeNum(0.0);
        }
        case NodeType::Composite:
            // 模型注册表（src/ResModel.cu）: ∂C/∂θ_p = Σ_k (∂C/∂arg_k)·deriv(arg_k, p)
            return modelDeriv(n.composite_id, n, p);
    }
    return Node::makeNum(0.0);
}

// ============================================================================
// 符号二阶微分 ∂²n/∂θ_p∂θ_q（结构性规则，避免 deriv(deriv()) 的 AST 爆炸）
// ============================================================================

Node deriv2(const Node& n, int p, int q)
{
    switch (n.type) {
        case NodeType::Num:
        case NodeType::Var:
        case NodeType::Param:
            return Node::makeNum(0.0);
        case NodeType::Add:
            return Node::makeOp(NodeType::Add,
                deriv2(n.kids[0], p, q), deriv2(n.kids[1], p, q));
        case NodeType::Sub:
            return Node::makeOp(NodeType::Sub,
                deriv2(n.kids[0], p, q), deriv2(n.kids[1], p, q));
        case NodeType::Neg:
            return Node::makeUnary(NodeType::Neg, deriv2(n.kids[0], p, q));
        case NodeType::Mul: {
            // d²(ab) = d²a·b + (da_p·db_q + da_q·db_p) + a·d²b
            //（一阶导项必须用 p、q 两个方向：da_p·db_q + da_q·db_p，
            //  仅 da_p·db_p 只对对角 p==q 成立）
            Node a = n.kids[0], b = n.kids[1];
            Node ap = deriv(a, p), aq = deriv(a, q);
            Node bp = deriv(b, p), bq = deriv(b, q);
            return Node::makeOp(NodeType::Add,
                Node::makeOp(NodeType::Add,
                    Node::makeOp(NodeType::Mul, deriv2(a, p, q), b),
                    Node::makeOp(NodeType::Add,
                        Node::makeOp(NodeType::Mul, ap, bq),
                        Node::makeOp(NodeType::Mul, aq, bp))),
                Node::makeOp(NodeType::Mul, a, deriv2(b, p, q)));
        }
        case NodeType::Div: {
            // d²(a/b) = d²a/b − (da_p·db_q + da_q·db_p)/b²
            //           + 2a·db_p·db_q/b³ − a·d²b/b²
            Node a = n.kids[0], b = n.kids[1];
            Node ap = deriv(a, p), aq = deriv(a, q);
            Node bp = deriv(b, p), bq = deriv(b, q);
            Node b2 = astSq(b);
            return Node::makeOp(NodeType::Sub,
                Node::makeOp(NodeType::Add,
                    Node::makeOp(NodeType::Sub,
                        Node::makeOp(NodeType::Div, deriv2(a, p, q), b),
                        Node::makeOp(NodeType::Div,
                            Node::makeOp(NodeType::Add,
                                Node::makeOp(NodeType::Mul, ap, bq),
                                Node::makeOp(NodeType::Mul, aq, bp)),
                            b2)),
                    Node::makeOp(NodeType::Div,
                        Node::makeOp(NodeType::Mul,
                            Node::makeOp(NodeType::Mul, a,
                                Node::makeOp(NodeType::Mul, bp, bq)),
                            Node::makeNum(2.0)),
                        Node::makeOp(NodeType::Mul, b2, b))),
                Node::makeOp(NodeType::Div,
                    Node::makeOp(NodeType::Mul, a, deriv2(b, p, q)), b2));
        }
        case NodeType::Pow: {
            // d²(a^b) = n·(L_p·L_q + L_pq)，对数微分（L_p/L_q 为 p/q 方向
            // 对数导数 L = db·log a + b·da/a；L_pq 为其 p,q 二阶混合导数）：
            //   L_p = db_p·log a + b·da_p/a
            //   L_q = db_q·log a + b·da_q/a
            //   L_pq = d²b·log a + (db_p·da_q + db_q·da_p)/a
            //          + b·d²a/a − b·da_p·da_q/a²
            Node a = n.kids[0], b = n.kids[1];
            Node ap = deriv(a, p), aq = deriv(a, q);
            Node bp = deriv(b, p), bq = deriv(b, q);
            Node loga = Node::makeFunc(CFUNC_LOG, a);
            Node Lp = Node::makeOp(NodeType::Add,
                Node::makeOp(NodeType::Mul, bp, loga),
                Node::makeOp(NodeType::Mul, b, Node::makeOp(NodeType::Div, ap, a)));
            Node Lq = Node::makeOp(NodeType::Add,
                Node::makeOp(NodeType::Mul, bq, loga),
                Node::makeOp(NodeType::Mul, b, Node::makeOp(NodeType::Div, aq, a)));
            Node Lpq = Node::makeOp(NodeType::Add,
                Node::makeOp(NodeType::Mul, deriv2(b, p, q), loga),
                Node::makeOp(NodeType::Sub,
                    Node::makeOp(NodeType::Add,
                        Node::makeOp(NodeType::Div,
                            Node::makeOp(NodeType::Add,
                                Node::makeOp(NodeType::Mul, bp, aq),
                                Node::makeOp(NodeType::Mul, bq, ap)), a),
                        Node::makeOp(NodeType::Div,
                            Node::makeOp(NodeType::Mul, b, deriv2(a, p, q)), a)),
                    Node::makeOp(NodeType::Div,
                        Node::makeOp(NodeType::Mul, b,
                            Node::makeOp(NodeType::Mul, ap, aq)),
                        astSq(a))));
            return Node::makeOp(NodeType::Mul, n,
                Node::makeOp(NodeType::Add, astMul(Lp, Lq), Lpq));
        }
        case NodeType::Func: {
            Node a = n.kids[0];
            Node ap = deriv(a, p), aq = deriv(a, q);
            Node d2a = deriv2(a, p, q);
            switch (n.func_id) {
                case CFUNC_EXP:
                    // d²e^a = e^a·(d²a + da_p·da_q)
                    return Node::makeOp(NodeType::Mul, n,
                        Node::makeOp(NodeType::Add, d2a, astMul(ap, aq)));
                case CFUNC_LOG:
                    // d²log a = d²a/a − da_p·da_q/a²
                    return Node::makeOp(NodeType::Sub,
                        Node::makeOp(NodeType::Div, d2a, a),
                        Node::makeOp(NodeType::Div, astMul(ap, aq), astSq(a)));
                case CFUNC_SIN: {
                    // d²sin a = cos a·d²a − sin a·da_p·da_q
                    Node cosn = Node::makeFunc(CFUNC_COS, a);
                    return Node::makeOp(NodeType::Sub,
                        Node::makeOp(NodeType::Mul, cosn, d2a),
                        Node::makeOp(NodeType::Mul, n, astMul(ap, aq)));
                }
                case CFUNC_COS: {
                    // d²cos a = −sin a·d²a − cos a·da_p·da_q
                    Node sinn = Node::makeFunc(CFUNC_SIN, a);
                    return Node::makeOp(NodeType::Sub,
                        Node::makeUnary(NodeType::Neg,
                            Node::makeOp(NodeType::Mul, sinn, d2a)),
                        Node::makeOp(NodeType::Mul, n, astMul(ap, aq)));
                }
                case CFUNC_SQRT:
                case CFUNC_CSQRT:
                    // d²√a = d²a/(2√a) − da_p·da_q/(4·a·√a)
                    return Node::makeOp(NodeType::Sub,
                        Node::makeOp(NodeType::Div, d2a,
                            Node::makeOp(NodeType::Mul, Node::makeNum(2.0), n)),
                        Node::makeOp(NodeType::Div, astMul(ap, aq),
                            Node::makeOp(NodeType::Mul,
                                Node::makeOp(NodeType::Mul,
                                    Node::makeNum(4.0), a), n)));
                case CFUNC_ABS: {
                    // d²|z| = [Re(conj·d²z) + Re(conj(da_p)·da_q)]/|z|
                    //         − Re(conj·da_p)·Re(conj·da_q)/|z|³
                    Node conjz = Node::makeFunc(CFUNC_CONJ, a);
                    Node tp = Node::makeFunc(CFUNC_RE,
                        Node::makeOp(NodeType::Mul, conjz, ap));
                    Node tq = Node::makeFunc(CFUNC_RE,
                        Node::makeOp(NodeType::Mul, conjz, aq));
                    Node dzsq = Node::makeFunc(CFUNC_RE,
                        Node::makeOp(NodeType::Mul,
                            Node::makeFunc(CFUNC_CONJ, ap), aq));
                    Node d2term = Node::makeFunc(CFUNC_RE,
                        Node::makeOp(NodeType::Mul, conjz, d2a));
                    return Node::makeOp(NodeType::Sub,
                        Node::makeOp(NodeType::Div,
                            Node::makeOp(NodeType::Add, d2term, dzsq), n),
                        Node::makeOp(NodeType::Div, astMul(tp, tq),
                            Node::makeOp(NodeType::Mul, astSq(n), n)));
                }
                case CFUNC_RE:
                    return Node::makeFunc(CFUNC_RE, d2a);
                case CFUNC_IM:
                    return Node::makeFunc(CFUNC_IM, d2a);
                case CFUNC_CONJ:
                    return Node::makeFunc(CFUNC_CONJ, d2a);
            }
            return Node::makeNum(0.0);
        }
        case NodeType::Composite:
            // 模型注册表（src/ResModel.cu）: 重建定义表达式 + 结构性二阶导
            return modelDeriv2(n.composite_id, n, p, q);
    }
    return Node::makeNum(0.0);
}

// 化简: 0±x → x, x±0 → x, 0*x → 0, x*1 → x, 1*x → x, 0/x → 0, x/1 → x,
//       x^1 → x, 1^x → 1, 0^x → 0, -( -x ) → x, 数值折叠
bool isZero(const Node& n) { return n.type == NodeType::Num && n.num == 0.0; }
bool isOne(const Node& n) { return n.type == NodeType::Num && n.num == 1.0; }
bool isConst(const Node& n) { return n.type == NodeType::Num; }

Node simplify(Node n)
{
    for (auto& k : n.kids) k = simplify(k);
    switch (n.type) {
        case NodeType::Add:
            if (isZero(n.kids[0])) return n.kids[1];
            if (isZero(n.kids[1])) return n.kids[0];
            if (isConst(n.kids[0]) && isConst(n.kids[1]))
                return Node::makeNum(n.kids[0].num + n.kids[1].num);
            break;
        case NodeType::Sub:
            if (isZero(n.kids[1])) return n.kids[0];
            if (isConst(n.kids[0]) && isConst(n.kids[1]))
                return Node::makeNum(n.kids[0].num - n.kids[1].num);
            break;
        case NodeType::Mul:
            if (isZero(n.kids[0]) || isZero(n.kids[1])) return Node::makeNum(0.0);
            if (isOne(n.kids[0])) return n.kids[1];
            if (isOne(n.kids[1])) return n.kids[0];
            if (isConst(n.kids[0]) && isConst(n.kids[1]))
                return Node::makeNum(n.kids[0].num * n.kids[1].num);
            break;
        case NodeType::Div:
            if (isZero(n.kids[0])) return Node::makeNum(0.0);
            if (isOne(n.kids[1])) return n.kids[0];
            if (isConst(n.kids[0]) && isConst(n.kids[1]) && n.kids[1].num != 0.0)
                return Node::makeNum(n.kids[0].num / n.kids[1].num);
            break;
        case NodeType::Pow:
            if (isOne(n.kids[1])) return n.kids[0];
            if (isOne(n.kids[0])) return Node::makeNum(1.0);
            if (isZero(n.kids[0])) return Node::makeNum(0.0);
            if (isConst(n.kids[0]) && isConst(n.kids[1]))
                return Node::makeNum(std::pow(n.kids[0].num, n.kids[1].num));
            break;
        case NodeType::Neg:
            if (n.kids[0].type == NodeType::Neg) return n.kids[0].kids[0];
            if (isConst(n.kids[0])) return Node::makeNum(-n.kids[0].num);
            if (isZero(n.kids[0])) return Node::makeNum(0.0);
            break;
        case NodeType::Composite:
            // 黑盒节点原样穿越（导数规则在 deriv 的 Composite case 展开）
            break;
        default: break;
    }
    return n;
}

// ============================================================================
// Flatte 传播子 AST 展开（全局 API，见 SymbolicDiff.cuh）
// 与 buildModelAST 的 Flatte 分支同一数学；DSL Flatte() 函数共用。
// ============================================================================

Node buildFlatteAST(const Node& m, const Node& m0,
                    const std::vector<Node>& gs,
                    const std::vector<Node>& chs)
{
    // F = (A − iB)/D
    // A = m0² - m² + Σ g_i·Im(ρ_i),  B = -Σ g_i·Re(ρ_i),  D = A² + B²
    // ρ_i = MODEL_FLATTE_RHO_{RE,IM}(s, ma_i, mb_i)：仅依赖 s=m² 和道质量
    // （导数 0 → 链式法则自动处理）
    // tf-pwa FlatteC 约定（与 ResModel.cuh Flatte<T> 一致）:
    // D = m0² - m² - i·m0·Σ gᵢ·(qᵢ/m)，qᵢ/m = ρᵢ/2 → 虚部系数 = g_i·m0/2
    Node s = astSq(m);                       // s = m²
    Node A = astSq(m0);                      // A = m0²
    Node B = astNum(0.0);                    // B = -Σ g_i·R_i（初始 0）
    Node half_m0 = astMul(m0, astNum(0.5));  // FlatteC: m0/2
    for (size_t i = 0; i < gs.size(); ++i) {
        Node Ri = Node::makeComposite(MODEL_FLATTE_RHO_RE,
            {s, chs[2 * i], chs[2 * i + 1]});
        Node Ii = Node::makeComposite(MODEL_FLATTE_RHO_IM,
            {s, chs[2 * i], chs[2 * i + 1]});
        Node gw = astMul(gs[i], half_m0);    // FlatteC 宽度项系数 g_i·m0/2
        A = astAdd(A, astMul(gw, Ii));       // A += (g_i·m0/2)·I_i
        B = astSub(B, astMul(gw, Ri));       // B -= (g_i·m0/2)·R_i
    }
    A = astSub(A, s);                        // A = m0² - s + Σ g_i·I_i
    Node D = astAdd(astSq(A), astSq(B));     // D = A² + B²
    return astAdd(astDiv(A, D), astMul(astNeg(astDiv(B, D)), astI()));
}

// ============================================================================
// 字节码编译（AST → 指令段，全局 API，见 SymbolicDiff.cuh）
// ============================================================================

// ============================================================================
// 编译期公共子表达式消除（CSE）
//
// deriv2/modelDeriv2 输出的 AST 是纯表达式树，共享子表达式在树中重复出现
// （无共享指针，每次引用整棵复制）。直线字节码里重复求值这些子树是 d²F
// 段膨胀的主因（BWR d²F 实测 1.5 万+ 指令）。做法：
//   - 结构相同且足够大（≥ kMinCse 节点）的子树只编译一次，结尾发
//     COP_STORE（非破坏性复制栈顶到槽位，不扰动操作数栈）；
//   - 后续出现处直接发 COP_LOAD 压栈复用。
// 段内槽位有限（解释器 tre/tim 容量 64），耗尽后自动退化为重算。
// ============================================================================

namespace {

constexpr int kMinCse = 4;       // 小于该节点数的子树不 CSE（省 STORE/LOAD 开销）
constexpr int kMaxCseSlots = 63; // 与 evalCustomSeg 的 kSlots=64 对应（留 1 余量）

int nodeCount(const Node& n)
{
    int c = 1;
    for (const auto& k : n.kids) c += nodeCount(k);
    return c;
}

// 结构哈希（FNV-1a 双混合，防碰撞误复用）
uint64_t nodeHashMix(const Node& n, uint64_t seed)
{
    uint64_t h = seed;
    auto mix = [&](uint64_t v) {
        h ^= v;
        h *= 1099511628211ULL;
    };
    mix((uint64_t)n.type);
    {
        uint64_t bits;
        memcpy(&bits, &n.num, sizeof(bits));
        mix(bits);
    }
    mix((uint64_t)n.var_id);
    mix((uint64_t)n.param_id);
    mix((uint64_t)n.func_id);
    mix((uint64_t)n.composite_id);
    for (const auto& k : n.kids) h = nodeHashMix(k, h);
    return h;
}

struct CseState {
    std::map<uint64_t, int> seen;   // 双哈希混合值 → 槽位
    int next_slot = 0;
};

void compileNodeRec(const Node& n, std::vector<double>& seg, CseState& cs)
{
    auto emit = [&](double op, double a0, double a1) {
        seg.push_back(op); seg.push_back(a0); seg.push_back(a1);
    };

    // 叶子直接编译（单条指令，CSE 无收益）
    if (n.type == NodeType::Num || n.type == NodeType::Var ||
        n.type == NodeType::Param) {
        if (n.type == NodeType::Num) emit(COP_PUSH_NUM, n.num, 0);
        else if (n.type == NodeType::Var) emit(COP_PUSH_VAR, n.var_id, 0);
        else emit(COP_PUSH_PARAM, n.param_id, 0);
        return;
    }

    // 足够大的子树：先查 CSE 表
    uint64_t h = 0;
    bool cse_candidate = (nodeCount(n) >= kMinCse) && cs.next_slot < kMaxCseSlots;
    if (cse_candidate) {
        h = nodeHashMix(n, 0xcbf29ce484222325ULL) ^ nodeHashMix(n, 0x9e3779b97f4a7c15ULL);
        auto it = cs.seen.find(h);
        if (it != cs.seen.end()) {
            emit(COP_LOAD, it->second, 0);   // 复用已有槽位
            return;
        }
    }

    switch (n.type) {
        case NodeType::Add: compileNodeRec(n.kids[0], seg, cs); compileNodeRec(n.kids[1], seg, cs); emit(COP_ADD, 0, 0); break;
        case NodeType::Sub: compileNodeRec(n.kids[0], seg, cs); compileNodeRec(n.kids[1], seg, cs); emit(COP_SUB, 0, 0); break;
        case NodeType::Mul: compileNodeRec(n.kids[0], seg, cs); compileNodeRec(n.kids[1], seg, cs); emit(COP_MUL, 0, 0); break;
        case NodeType::Div: compileNodeRec(n.kids[0], seg, cs); compileNodeRec(n.kids[1], seg, cs); emit(COP_DIV, 0, 0); break;
        case NodeType::Pow: compileNodeRec(n.kids[0], seg, cs); compileNodeRec(n.kids[1], seg, cs); emit(COP_POW, 0, 0); break;
        case NodeType::Neg: compileNodeRec(n.kids[0], seg, cs); emit(COP_NEG, 0, 0); break;
        case NodeType::Func:
            compileNodeRec(n.kids[0], seg, cs);
            emit(COP_CALL, n.func_id, 0);
            break;
        case NodeType::Composite:
            // 参数按注册表顺序入栈: 先编译 kids[0]（第一个参数先入）
            for (const auto& k : n.kids) compileNodeRec(k, seg, cs);
            emit(COP_MODEL, n.composite_id, 0);
            break;
        default:
            emit(COP_PUSH_NUM, 0.0, 0);
            break;
    }

    // 非破坏性登记：栈顶值复制进槽位，后续 COP_LOAD 复用。
    // 注意：必须在递归完成后再检查槽位上限——cse_candidate 在递归前
    // 求值，子节点递归期间 next_slot 会继续增长（BWR d²F 段实测会越过
    // 63 到 88+），递归前检查会发出越界槽位，解释器 kSlots=64 直接 OOB
    // 写崩溃（JIT genSegment 会 resize 所以不受影响）。
    if (cse_candidate && cs.next_slot < kMaxCseSlots) {
        emit(COP_STORE, cs.next_slot, 0);
        cs.seen[h] = cs.next_slot;
        ++cs.next_slot;
    }
}

}  // namespace

void compileNode(const Node& n, std::vector<double>& seg)
{
    CseState cs;
    compileNodeRec(n, seg, cs);
}

// ============================================================================
// 设备解释器
// ============================================================================

__device__ void evalCustomSeg(
    const double* seg, int n_instr,
    double m, double q, double q0, int L, double d,
    double md1, double md2,
    double p1p, double p1e, double p1costheta, double p1phi,
    double p2p, double p2e, double p2costheta, double p2phi,
    const double* params, double* out)
{
    // 栈式复数求值
    constexpr int kStack = 32;
    double sre[kStack], sim[kStack];
    int sp = 0;
    // CSE 临时槽（COP_STORE/COP_LOAD，编译期公共子表达式消除）
    constexpr int kSlots = 64;
    double tre[kSlots], tim[kSlots];
    const double* pc = seg;

    auto push = [&](double re, double im) {
        if (sp >= kStack) { out[0] = 0; out[1] = 0; return; }
        sre[sp] = re; sim[sp] = im; ++sp;
    };
    auto pop = [&](double& re, double& im) {
        if (sp <= 0) { re = 0; im = 0; return; }
        --sp; re = sre[sp]; im = sim[sp];
    };

    for (int i = 0; i < n_instr; ++i) {
        int op = (int)pc[0];
        double a0 = pc[1], a1 = pc[2];
        pc += 3;
        switch (op) {
            case COP_PUSH_NUM: push(a0, 0.0); break;
            case COP_PUSH_VAR: {
                double v = 0.0;
                switch ((int)a0) {
                    case CVAR_M: v = m; break;
                    case CVAR_Q: v = q; break;
                    case CVAR_Q0: v = q0; break;
                    case CVAR_L: v = (double)L; break;
                    case CVAR_D: v = d; break;
                    case CVAR_MD1: v = md1; break;
                    case CVAR_MD2: v = md2; break;
                    case CVAR_P1P: v = p1p; break;
                    case CVAR_P1E: v = p1e; break;
                    case CVAR_P1COSTHETA: v = p1costheta; break;
                    case CVAR_P1PHI: v = p1phi; break;
                    case CVAR_P2P: v = p2p; break;
                    case CVAR_P2E: v = p2e; break;
                    case CVAR_P2COSTHETA: v = p2costheta; break;
                    case CVAR_P2PHI: v = p2phi; break;
                }
                push(v, 0.0); break;
            }
            case COP_PUSH_PARAM: push(params[(int)a0], 0.0); break;
            case COP_PUSH_I: push(0.0, 1.0); break;
            case COP_PUSH_PI: push(3.14159265358979323846, 0.0); break;
            case COP_STORE: {
                // 非破坏性：栈顶复制进槽位（后续 COP_LOAD 复用）
                int sk = (int)a0;
                if (sk < 0 || sk >= kSlots) break;   // 防御：编译器保证 < kMaxCseSlots
                if (sp <= 0) break;
                tre[sk] = sre[sp - 1];
                tim[sk] = sim[sp - 1];
                break;
            }
            case COP_LOAD: {
                int k = (int)a0;
                if (k < 0 || k >= kSlots) break;     // 防御：编译器保证 < kMaxCseSlots
                push(tre[k], tim[k]);
                break;
            }
            case COP_ADD: { double br, bi, ar, ai; pop(br, bi); pop(ar, ai); push(ar + br, ai + bi); break; }
            case COP_SUB: { double br, bi, ar, ai; pop(br, bi); pop(ar, ai); push(ar - br, ai - bi); break; }
            case COP_MUL: { double br, bi, ar, ai; pop(br, bi); pop(ar, ai); push(ar * br - ai * bi, ar * bi + ai * br); break; }
            case COP_DIV: {
                double br, bi, ar, ai;
                pop(br, bi); pop(ar, ai);
                double s = br * br + bi * bi;
                if (s == 0.0) s = 1e-30;
                push((ar * br + ai * bi) / s, (ai * br - ar * bi) / s);
                break;
            }
            case COP_POW: {
                // 栈顺序: base 先入 → pop 得 exp, base
                double er, ei, br, bi;
                pop(er, ei); pop(br, bi);
                // b^e = exp(e·log(b))
                double a = sqrt(br * br + bi * bi);
                if (a == 0.0) a = 1e-30;
                double lr = log(a), li = atan2(bi, br);
                double xr = er * lr - ei * li, xi = er * li + ei * lr;
                double ex = exp(xr);
                push(ex * cos(xi), ex * sin(xi));
                break;
            }
            case COP_NEG: { double ar, ai; pop(ar, ai); push(-ar, -ai); break; }
            case COP_MODEL: {
                // 内置共振态黑盒求值。参数在栈上按注册表顺序
                // （kids[0] 最先入栈），arg1 = Flatte 的道数（其他模型 0）
                switch ((int)a0) {
                    case MODEL_BREAKUP_Q0: {  // [m0, m1, m2]
                        double m2v, m1v, m0v, _t;
                        pop(m2v, _t); pop(m1v, _t); pop(m0v, _t);
                        push(computeQ0AD(m0v, m1v, m2v), 0.0);
                        break;
                    }
                    case MODEL_BF: {  // [L, q, q0, d]
                        double dv, q0v, qv, Lv, _t;
                        pop(dv, _t); pop(q0v, _t); pop(qv, _t); pop(Lv, _t);
                        push(Bf<double>((int)Lv, qv, q0v, dv), 0.0);
                        break;
                    }
                    case MODEL_BW: {  // [m, m0, g0]
                        double g0v, m0v, mv, _t;
                        pop(g0v, _t); pop(m0v, _t); pop(mv, _t);
                        auto z = BW<double>(mv, m0v, g0v);
                        push(z.real(), z.imag());
                        break;
                    }
                    case MODEL_BWR: {  // [m, m0, g0, L, q, q0, d]
                        double dv, q0v, qv, Lv, g0v, m0v, mv, _t;
                        pop(dv, _t); pop(q0v, _t); pop(qv, _t);
                        pop(Lv, _t); pop(g0v, _t); pop(m0v, _t); pop(mv, _t);
                        auto z = BWR<double>(mv, m0v, g0v, (int)Lv, qv, q0v, dv);
                        push(z.real(), z.imag());
                        break;
                    }
                    case MODEL_FLATTE: {  // [m, m0, g0..g_{n-1}, (ma,mb)0..]
                        int n_ch = (int)a1;
                        double g[4], ch[8], _t;
                        for (int i = 2 * n_ch - 1; i >= 0; --i) pop(ch[i], _t);
                        for (int i = n_ch - 1; i >= 0; --i) pop(g[i], _t);
                        double m0v, mv;
                        pop(m0v, _t); pop(mv, _t);
                        auto z = Flatte<double>(mv, m0v, n_ch, g, ch);
                        push(z.real(), z.imag());
                        break;
                    }
                    case MODEL_FLATTE_RHO_RE:
                    case MODEL_FLATTE_RHO_IM: {
                        // [s, ma, mb] → ρ = csqrt(1-(ma+mb)²/s) * √(clip(1-(ma-mb)²/s,0))
                        double mb_im, ma_im, s_im;
                        double mbv, mav, sv;
                        pop(mbv, mb_im); pop(mav, ma_im); pop(sv, s_im);
                        double S = (mav + mbv) * (mav + mbv);
                        double D = (mav - mbv) * (mav - mbv);
                        double f1_sq = 1.0 - S / sv;
                        double f1_re, f1_im;
                        if (f1_sq >= 0.0) { f1_re = sqrt(f1_sq); f1_im = 0.0; }
                        else              { f1_re = 0.0; f1_im = sqrt(-f1_sq); }
                        double f2_sq = 1.0 - D / sv;
                        double f2 = (f2_sq > 0.0) ? sqrt(f2_sq) : 0.0;
                        double rho_re = f1_re * f2;
                        double rho_im = f1_im * f2;
                        if ((int)a0 == MODEL_FLATTE_RHO_RE)
                            push(rho_re, 0.0);
                        else
                            push(rho_im, 0.0);
                        break;
                    }
                    case MODEL_ONE:
                        push(1.0, 0.0);
                        break;
                    default:
                        push(0.0, 0.0);
                }
                break;
            }
            case COP_CALL: {
                double ar, ai;
                pop(ar, ai);
                switch ((int)a0) {
                    case CFUNC_EXP: { double e = exp(ar); push(e * cos(ai), e * sin(ai)); break; }
                    case CFUNC_LOG: {
                        double a = sqrt(ar * ar + ai * ai);
                        if (a == 0.0) a = 1e-30;
                        push(log(a), atan2(ai, ar));
                        break;
                    }
                    case CFUNC_SIN: {
                        double e = exp(ai), em = exp(-ai);
                        push(sin(ar) * (e + em) / 2, cos(ar) * (e - em) / 2);
                        break;
                    }
                    case CFUNC_COS: {
                        double e = exp(ai), em = exp(-ai);
                        push(cos(ar) * (e + em) / 2, -sin(ar) * (e - em) / 2);
                        break;
                    }
                    case CFUNC_SQRT:
                    case CFUNC_CSQRT: {
                        double a = sqrt(ar * ar + ai * ai);
                        double r = sqrt((a + ar) / 2);
                        double ii = sqrt((a - ar) / 2) * (ai < 0 ? -1.0 : 1.0);
                        push(r, ii);
                        break;
                    }
                    case CFUNC_ABS: push(sqrt(ar * ar + ai * ai), 0.0); break;
                    case CFUNC_RE: push(ar, 0.0); break;
                    case CFUNC_IM: push(ai, 0.0); break;
                    case CFUNC_CONJ: push(ar, -ai); break;
                }
                break;
            }
        }
    }
    if (sp > 0) { --sp; out[0] = sre[sp]; out[1] = sim[sp]; }
    else { out[0] = 0; out[1] = 0; }
}

__device__ void evalCustomAll(
    const double* aux, int aux_offset,
    double m, double q, double q0, int L, double d,
    double md1, double md2,
    double p1p, double p1e, double p1costheta, double p1phi,
    double p2p, double p2e, double p2costheta, double p2phi,
    const double* params, int P,
    double& Fr, double& Fi,
    double* dFr, double* dFi,
    double* d2Fr, double* d2Fi,
    bool compute_2nd)
{
    int seg_off = aux_offset + 2;
    double out[2];
    // 值段
    int n_instr = (int)aux[seg_off];
    evalCustomSeg(aux + seg_off + 1, n_instr, m, q, q0, L, d, md1, md2,
                  p1p, p1e, p1costheta, p1phi, p2p, p2e, p2costheta, p2phi, params, out);
    Fr = out[0]; Fi = out[1];
    seg_off += 1 + 3 * n_instr;
    // 一阶段
    for (int j = 0; j < P; ++j) {
        n_instr = (int)aux[seg_off];
        evalCustomSeg(aux + seg_off + 1, n_instr, m, q, q0, L, d, md1, md2,
                      p1p, p1e, p1costheta, p1phi, p2p, p2e, p2costheta, p2phi, params, out);
        dFr[j] = out[0]; dFi[j] = out[1];
        seg_off += 1 + 3 * n_instr;
    }
    // 二阶段（段序 j≤k，对称填充）
    if (compute_2nd)
    for (int j = 0; j < P; ++j)
        for (int k = j; k < P; ++k) {
            n_instr = (int)aux[seg_off];
            evalCustomSeg(aux + seg_off + 1, n_instr, m, q, q0, L, d, md1, md2,
                          p1p, p1e, p1costheta, p1phi, p2p, p2e, p2costheta, p2phi, params, out);
            d2Fr[j * P + k] = out[0]; d2Fi[j * P + k] = out[1];
            d2Fr[k * P + j] = out[0]; d2Fi[k * P + j] = out[1];
            seg_off += 1 + 3 * n_instr;
        }
}


// ============================================================================
// host 编译入口
// ============================================================================

std::vector<double> compileCustomExpr(
    const std::string& expr,
    const std::vector<std::string>& params)
{
    if (params.size() > 16)
        throw std::runtime_error("CustomExpr: at most 16 free params supported");

    std::map<std::string, int> param_map;
    for (size_t i = 0; i < params.size(); ++i) param_map[params[i]] = (int)i;

    auto toks = tokenize(expr);
    Parser parser(toks, param_map);
    Node root = simplify(parser.parse());
    int P = (int)params.size();

    // 值段 + 一阶段 + 二阶段（j ≤ k）
    std::vector<Node> segs;
    segs.push_back(root);
    for (int j = 0; j < P; ++j)
        segs.push_back(simplify(deriv(root, j)));
    for (int j = 0; j < P; ++j)
        for (int k = j; k < P; ++k)
            // 直接二阶微分（对一阶导展开式再微分会让 AST 爆炸：
            // 实测 BWR d²F 段 2.6 万指令 → deriv2 结构规则后数百指令）
            segs.push_back(simplify(deriv2(root, j, k)));

    // 布局: [P, n_seg, seg...]
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
