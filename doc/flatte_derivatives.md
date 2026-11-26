# Flatte Propagator: Symbolic Formulas for P6 Implementation

## Propagator

```
F(s) = (A - iB) / D    where D = A² + B²

A = m₀² - s + Σ_j g_j · Im(ρ_j)
B = -Σ_j g_j · Re(ρ_j)                     (note: B is negative)
```

## Phase Space Factor ρ_j(s)

For channel j with masses m_a, m_b:

```
S_j = (m_a + m_b)²
D_j = (m_a - m_b)²

f1 = csqrt(1 - S_j/s)       — complex sqrt, branch at s = S_j
f2 = √(max(0, 1 - D_j/s))   — real sqrt, clamped to 0 below D_j

ρ_j = f1 · f2 = R_j + i·I_j
```

Branch behavior:
| Region | s range | f1 | f2 | ρ_j |
|---|---|---|---|---|
| Below D_j | s < D_j | 0 + i·√(S_j/s-1) | 0 | 0 |
| Between | D_j < s < S_j | 0 + i·√(S_j/s-1) | √(1-D_j/s) | i · √((S_j-s)(s-D_j))/s |
| Above S_j | s > S_j | √(1-S_j/s) + 0i | √(1-D_j/s) | √((s-S_j)(s-D_j))/s |

## Key Insight

**ρ_j does NOT depend on fit parameters (m₀, g_j).**
It depends only on s (CVAR_M, event mass squared) and channel masses (constants).
Therefore: **∂ρ_j/∂m₀ = 0, ∂ρ_j/∂g_k = 0**, and no branch-cut handling
is needed in the derivative computation.

## First Derivatives

### By m₀

```
∂A/∂m₀ = 2m₀
∂B/∂m₀ = 0
∂D/∂m₀ = 4m₀A

∂F_re/∂m₀ = 2m₀(B² - A²) / D²
∂F_im/∂m₀ = 4m₀AB / D²
```

### By g_k (coupling for channel k)

Let R_k = Re(ρ_k), I_k = Im(ρ_k).

```
∂A/∂g_k = I_k
∂B/∂g_k = -R_k
∂D/∂g_k = 2(A·I_k - B·R_k)

∂F_re/∂g_k = (I_k(B² - A²) + 2ABR_k) / D²
∂F_im/∂g_k = (R_k(A² - B²) + 2ABI_k) / D²
```

## Second Derivatives

### ∂²/∂m₀²

```
∂²F_re/∂m₀² = 2(B²-A²)/D² - 16m₀²A(B²-A²)/D³ - 8m₀²A/D²
∂²F_im/∂m₀² = 4AB/D² + 8m₀²B/D² - 32m₀²A²B/D³
```

### ∂²/∂m₀∂g_k

```
∂²F_re/∂m₀∂g_k = (2m₀/D²)(BR_k - 2AI_k) - (8m₀A/D³)(I_k(B²-A²) + 2ABR_k)
∂²F_im/∂m₀∂g_k = (2m₀/D²)(2AR_k + BI_k) - (8m₀A/D³)(R_k(A²-B²) + 2ABI_k)
```

### ∂²/∂g_k∂g_l

Apply quotient rule to first derivative of g_k, differentiating by g_l.
General form:

```
∂(dF/dg_k)/∂g_l = [∂²A/∂g_k∂g_l·D + ∂A/∂g_k·∂D/∂g_l - ∂A/∂g_l·∂D/∂g_k - A·∂²D/∂g_k∂g_l] / D²
                 - 2 [∂A/∂g_k·D - A·∂D/∂g_k] · ∂D/∂g_l / D³
```

Since ∂²A/∂g_k∂g_l = 0 (A is linear in g), ∂²B/∂g_k∂g_l = 0:

For k = l:
```
∂²D/∂g_k² = 2(I_k² + R_k²)

∂²F_re/∂g_k² = [I_k·∂D/∂g_k - I_k·∂D/∂g_k - A·2(I_k²+R_k²)] / D²
              - 2[I_k·D - A·∂D/∂g_k]·∂D/∂g_k / D³
```

For k ≠ l:
```
∂²D/∂g_k∂g_l = 2(I_k·I_l + R_k·R_l)

∂²F_re/∂g_k∂g_l = [I_k·∂D/∂g_l + I_l·∂D/∂g_k - A·2(I_k·I_l+R_k·R_l)] / D²
                 - 2[I_k·D - A·∂D/∂g_k]·∂D/∂g_l / D³
```

## General Quotient-Rule Framework

All derivatives follow the same pattern. Define:

```
F_re = A / D
F_im = -B / D      (where B is negative, so F_im > 0)

dF_re/dp = (dA/dp · D - A · dD/dp) / D²
dF_im/dp = (-dB/dp · D + B · dD/dp) / D²

d²F/dpdq = derivative of dF/dp by q (apply quotient rule again)
```

Where the elementary derivatives are:

| | ∂/∂m₀ | ∂/∂g_k |
|---|---|---|
| ∂A | 2m₀ | I_k |
| ∂B | 0 | -R_k |
| ∂D | 4m₀A | 2(A·I_k - B·R_k) |
| ∂²A | 2 | 0 |
| ∂²B | 0 | 0 |
| ∂²D/(∂m₀∂g_k) | 4m₀·I_k | — |
| ∂²D/(∂g_k∂g_l) | — | 2(I_k·I_l + R_k·R_l) |

## Implementation Strategy for P6

1. **Runtime**: Compute R_j, I_j per-channel from s = m² (CVAR_M) and channel masses,
   using the branch-aware csqrt. This is equivalent to the existing Flatte<T> code.

2. **Bytecode (aux)**: Generate segments that compute F, dF, d2F using the formulas above.
   Since ρ_j doesn't depend on fit params, no branch handling needed in derivatives.

3. **Options**:
   - A) Add COP_MODEL_FLATTE to evalCustomSeg, hand-code the full eval
   - B) Generate bytecode using existing ops (CVAR_M, arithmetic), with a new COP_CSQRT
        opcode whose deriv returns 0 (since argument depends only on s, not fit params)
   - C) Keep the existing Flatte<double> runtime call and add derivative formulas
        as explicit bytecode segments in buildModelAST

   Option C is recommended: use Flatte<double> for segment 0 (value), generate explicit
   derivative bytecode using the formulas above for segments 1..P, P+1..
