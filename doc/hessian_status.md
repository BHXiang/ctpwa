# Hessian 多共振态修复状态报告

## 已验证（正确）

1. **AutoDiff BWR Hessian** (`Var<double,N,true>`): 与 PyTorch autograd 精确一致 (<1e-15)
2. **Per-event Hessian 公式**: 在 Python 中与 PyTorch 完全匹配
   ```
   g[j] = -2·bf·Re(D^j · Σ conj(w)·T)
   H = g·g^T - 2·I_inv·bf²·Σ|T|²·Re(conj(D_k)·D_j) - 2·bf·Re(Σ conj(w)·T · D²_jk)
   ```
3. **单共振态**: getHessian 与 PyTorch 匹配 (rel < 1e-7)
4. **θθ 交叉块公式**: H_rs = g_r·g_s^T (只有外积，无 d²S 修正项)

## 当前代码状态 (hessian-research 分支)

### 已修改
- `resonanceHessianBlockKernel` Step 5: 替换为 per-event 聚合公式
- `computeResonanceHessian`: 替换为直接调用 BlockKernel（移除 3-pass）
- `getHessian`: 添加 computeEffectiveCoupling 调用
- `d_v[site+sl_idx]`: double-offset 修复（3个kernel）

### 未解决
1. **符号/权重**: data 的 sign 是 +1 还是 -1 需要和 NLL 公式对齐
2. **bkg 权重**: 每个 bkg event 需要独立的 w_evt 权重传入 kernel
3. **PHSP 对 θθ 的贡献**: 完全缺失
4. **vθ 混合块**: 暂填零（单链模型下正确，多链需要实现）

## 建议方案

### 短期（最小修改让当前代码工作）
在 `getHessian` 里 per-event 循环调用 `computeResonanceHessian` 时传入正确的 w_evt:
- data: w_evt = -1
- bkg: w_evt 从 bkg_weights_ 读取（正权重 → 负贡献）
- 然后将 kernel 里的 `sign` 参数统一为 +1，外积和修正都用 `w_evt * (...)` 格式

### 长期（推荐架构）
```
computeResonanceHessian() {
    for each event:
        1. 计算 S = A*v, I = |S|², w = S/I
        2. for each resonance r:
            计算 g_r, H_rr (per-event)
            H_total += w_evt * H_rr  
            存储 g_r 到 per-event buffer
        3. for each pair (r,s) where r≠s:
            H_total += w_evt * g_r ⊗ g_s^T
        4. PHSP 贡献: H += n_data * ∂²log(phsp_factor)/∂θ²
}
```

### 如果重新开始
用 Python 先写完整的多共振态 NLL + Hessian，验证公式完全对上后，再逐块翻译成 CUDA kernel。避免在 C++ 里反复编译调试。
