#include "CustomExpr.cuh"
#include "SymbolicDiff.cuh"
#include "ResModel.cuh"   // COP_MODEL: Bf/BW/BWR/Flatte/computeQ0AD（double 实例化）

#include <cmath>
#include <cstdio>
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
            if (t.str == "q0") return Node::makeVar(CVAR_Q0);
            if (t.str == "L") return Node::makeVar(CVAR_L);
            if (t.str == "d") return Node::makeVar(CVAR_D);
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
// 字节码编译（AST → 指令段，全局 API，见 SymbolicDiff.cuh）
// ============================================================================

void compileNode(const Node& n, std::vector<double>& seg)
{
    auto emit = [&](double op, double a0, double a1) {
        seg.push_back(op); seg.push_back(a0); seg.push_back(a1);
    };
    switch (n.type) {
        case NodeType::Num: emit(COP_PUSH_NUM, n.num, 0); return;
        case NodeType::Var: emit(COP_PUSH_VAR, n.var_id, 0); return;
        case NodeType::Param: emit(COP_PUSH_PARAM, n.param_id, 0); return;
        case NodeType::Add: compileNode(n.kids[0], seg); compileNode(n.kids[1], seg); emit(COP_ADD, 0, 0); return;
        case NodeType::Sub: compileNode(n.kids[0], seg); compileNode(n.kids[1], seg); emit(COP_SUB, 0, 0); return;
        case NodeType::Mul: compileNode(n.kids[0], seg); compileNode(n.kids[1], seg); emit(COP_MUL, 0, 0); return;
        case NodeType::Div: compileNode(n.kids[0], seg); compileNode(n.kids[1], seg); emit(COP_DIV, 0, 0); return;
        case NodeType::Pow: compileNode(n.kids[0], seg); compileNode(n.kids[1], seg); emit(COP_POW, 0, 0); return;
        case NodeType::Neg: compileNode(n.kids[0], seg); emit(COP_NEG, 0, 0); return;
        case NodeType::Func:
            compileNode(n.kids[0], seg);
            emit(COP_CALL, n.func_id, 0);
            return;
        case NodeType::Composite:
            // 参数按注册表顺序入栈: 先编译 kids[0]（第一个参数先入）
            for (const auto& k : n.kids) compileNode(k, seg);
            emit(COP_MODEL, n.composite_id, 0);
            return;
    }
}

// ============================================================================
// 设备解释器
// ============================================================================

__device__ void evalCustomSeg(
    const double* seg, int n_instr,
    double m, double q, double q0, int L, double d,
    double md1, double md2,
    const double* params, double* out)
{
    // 栈式复数求值
    constexpr int kStack = 32;
    double sre[kStack], sim[kStack];
    int sp = 0;
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
                }
                push(v, 0.0); break;
            }
            case COP_PUSH_PARAM: push(params[(int)a0], 0.0); break;
            case COP_PUSH_I: push(0.0, 1.0); break;
            case COP_PUSH_PI: push(3.14159265358979323846, 0.0); break;
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
    const double* params, int P,
    double& Fr, double& Fi,
    double* dFr, double* dFi,
    double* d2Fr, double* d2Fi)
{
    int seg_off = aux_offset + 2;
    double out[2];
    // 值段
    int n_instr = (int)aux[seg_off];
    evalCustomSeg(aux + seg_off + 1, n_instr, m, q, q0, L, d, md1, md2, params, out);
    Fr = out[0]; Fi = out[1];
    seg_off += 1 + 3 * n_instr;
    // 一阶段
    for (int j = 0; j < P; ++j) {
        n_instr = (int)aux[seg_off];
        evalCustomSeg(aux + seg_off + 1, n_instr, m, q, q0, L, d, md1, md2, params, out);
        dFr[j] = out[0]; dFi[j] = out[1];
        seg_off += 1 + 3 * n_instr;
    }
    // 二阶段（段序 j≤k，对称填充）
    for (int j = 0; j < P; ++j)
        for (int k = j; k < P; ++k) {
            n_instr = (int)aux[seg_off];
            evalCustomSeg(aux + seg_off + 1, n_instr, m, q, q0, L, d, md1, md2, params, out);
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
            segs.push_back(simplify(deriv(deriv(root, j), k)));

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
