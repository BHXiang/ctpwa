# Interpolation 类模型统一设计

## 1. 共性分析

| | Hist | Linear Interp | Spline |
|---|---|---|---|
| 输入 | bin 高度数组 | 采样点 y 值 | 采样点 y 值 |
| 插值公式 | F(x)=y[i] (分段常数) | y[i] + (y[i+1]-y[i])×f | 三次多项式 |
| 光滑性 | C⁻¹ (不连续) | C⁰ (连续) | C² (光滑) |
| 有无自由参数 | 极少 | 极少 | 极少 |
| 需要梯度？ | 需要也是 trivial | 同上 | 同上 |
| aux 格式 | [m_min,m_max,N,vals] | 同上 | 同上（或+系数） |

**共性远大于差异** → 用一个统一模型 + `method` 参数区分。

## 2. 用户接口

```yaml
Resonances:
  bg_shape:
    model: Interp                    # 统一名称
    parameters: []                   # 无自由参数
    options:
      file: "./data/shape.dat"       # 数据文件 (两列: m, val)
      method: linear                 # hist | linear | spline
      bins: 100                      # 仅 hist: 直方图 bin 数
```

或者替代 `file`，支持内联数据：
```yaml
    options:
      points: [[0.3, 0.1], [0.5, 0.5], [1.0, 0.8], [1.5, 0.3]]
      method: spline
```

## 3. Device 端 aux 格式

统一 aux 布局（全部 double）：

```
[method_id, N, x_0, ..., x_{N-1}, y_0, ..., y_{N-1}]
   0          1    2           N+1       N+2          2N+1
```

- `method_id`: 0=hist, 1=linear, 2=spline
- `N`: 点数
- `x_i`: 横坐标（等距时可省略，存 x_min + dx）
- `y_i`: 纵坐标

对于**等距 bin**（最常见场景），压缩为：

```
[method_id | (N<<8), x_min, dx, y_0, ..., y_{N-1}]
```

用 `method_id` 的符号位编码是否等距：`method_id > 0` = 等距，`method_id < 0` = 非等距。

## 4. Kernel 实现

```cpp
// computeCustomAmpsKernel 中，R==1 路径:
if (target_res.eval_mode == EvalMode::Direct) {
    interpEval(aux + target_res.aux_offset, mm,
               Fr, Fi, dFr, dFi, P, target_rp);
}

// interpEval: 统一插值求值
__device__ void interpEval(const double* tab, double x,
    double& Fr, double& Fi, double* dFr, double* dFi, int P,
    const double* params)
{
    int hdr = (int)tab[0];
    int method = abs(hdr) & 0xFF;
    int N = (int)tab[1];
    bool uniform = (hdr > 0);

    // 定位 x 所在区间 [i, i+1]
    int i; double frac;
    if (uniform) {
        double xmin = tab[2], dx = tab[3];
        const double* y = tab + 4;
        double pos = (x - xmin) / dx;
        i = (int)pos;
        if (i < 0) { i = 0; frac = 0.0; }
        else if (i >= N-1) { i = N-2; frac = 1.0; }
        else { frac = pos - i; }
        switch (method) {
        case 0: // hist
            Fr = y[i]; break;
        case 1: // linear
            Fr = y[i] + (y[i+1] - y[i]) * frac; break;
        case 2: // spline (Catmull-Rom)
            Fr = catmullRom(y, i, frac, N); break;
        }
    } else {
        const double* xv = tab + 2;
        const double* yv = tab + 2 + N;
        // binary search for x in xv[0..N-1]
        i = binarySearch(xv, x, N);
        frac = (x - xv[i]) / (xv[i+1] - xv[i]);
        // same switch on method...
    }

    Fi = 0.0;
    // 梯度: P>0 时参数是 bin 高度 (极少见)
    if (P > 0) {
        for (int j = 0; j < P; ++j) dFr[j] = dFi[j] = 0.0;
        if (method == 0 && i < P)      dFr[i] = 1.0;               // hist: ∂/∂bin_i = 1
        if (method == 1 && i < P)      dFr[i] = 1.0 - frac;        // linear: 权重
        if (method == 1 && i+1 < P)    dFr[i+1] = frac;
        // spline 梯度类似（多项式系数对控制点的导数）
    }
}
```

## 5. 与符号微分轨的关系

```
computeCustomAmpsKernel
  │
  ├─ eval_mode == Symbolic → evalCustomAll(aux 字节码)
  │                           (BWR/BW/Flatte/GS/Custom/ONE)
  │
  └─ eval_mode == Direct   → interpEval(aux 插值表)
                              (Interp 模型, method=hist|linear|spline)
```

两条轨**共享同一个 O×F 和 d_dF_tab 写入**，只是 F/dF 的来源不同。

## 6. 扩展性

未来加新插值方法只需：
1. `interpEval` 里加一个 `case method:`
2. config 里加对应的 `method` 名字

例如：
- `method: akima` → Akima 插值
- `method: gauss` → 高斯平滑核插值
- `method: kmatrix` → K-Matrix（需要参数量大时升至符号轨）

## 7. 向后兼容

现有 `model: Hist` 的 config 自动映射到 `model: Interp, method: linear`（因为当前 `lookupHistTable` 实现的就是线性插值）。
