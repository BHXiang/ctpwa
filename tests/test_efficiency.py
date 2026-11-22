"""getEfficiency / getFitFractions 测试（CPU 与 CUDA 后端都跑）。

约定: phsp(如 cut 后 MC)=带效率样本, phsp_truth=无效率 MC truth。
- eff.yml 把 phsp 与 phsp_truth 指向同一文件 ⇒ 效率解析结果严格 ≡ 1。
- simple.yml 缺 phsp_truth ⇒ 返回空张量 [0,2]（自然跳过，不抛异常）。

回归（CPU 后端踩过的两个坑）:
1. `getFitFractions(v, hessian)` / `getEfficiency(v, hessian)` 的 hessian 检查曾写死
   `is_cuda()` → CPU 构建只要传非空 Hessian 必抛 "hessian_in must be float64 CUDA"；
2. `phsp_truth` 的早退排在 Tensor 检查之后 → 没有 phsp_truth 的配置也会因为传了
   一个 dtype/device 不合适的 Hessian 而抛错，而它本该安静返回空。

参数一律在 CPU 上生成再 `.to(DEVICE)`（`torch.Generator(device=...)` 的随机流
在两个后端不同，直接在目标设备上生成会让两端比到不同参数点）。
"""

from pathlib import Path

import torch
import pytest

import ctpwa

DEVICE = torch.device("cpu") if ctpwa.backend() == "cpu" else torch.device("cuda:0")


def _complex_dtype():
    cb = int(ctpwa.build_info().get("complex_bytes", 16))
    return torch.float64 if cb == 16 else torch.float32


def _coupling(ana, params):
    """按 .so 编译精度构造耦合向量(complex64/128)，放到当前后端设备。"""
    dt = _complex_dtype()
    n = ana.getNVector()
    return torch.complex(params[:n].to(dt), params[n:2 * n].to(dt)).contiguous().to(DEVICE)


def _params(ana, seed=42):
    """[Re(v), Im(v), θ]，在 CPU 生成后搬到目标设备（保证跨后端同值）。"""
    n_vec, n_theta = ana.getNVector(), ana.getNFreeTheta()
    p = torch.zeros(2 * n_vec + n_theta, dtype=torch.float64)
    p[0] = 1.0
    g = torch.Generator(device="cpu").manual_seed(seed)
    if n_vec > 1:
        p[1:n_vec] = torch.rand(n_vec - 1, generator=g) * 0.5
        p[n_vec + 1:2 * n_vec] = (torch.rand(n_vec - 1, generator=g) - 0.5) * 0.5
    if n_theta > 0:
        p[2 * n_vec:] = ana.getFreeResParams()[0].double().cpu()
    return p.to(DEVICE)


def test_efficiency_same_file_center_is_one(make_analysis):
    """phsp 与 phsp_truth 同一文件时，每个分波效率必须严格为 1。"""
    ana = make_analysis("eff")
    params = _params(ana)
    eff = ana.getEfficiency(_coupling(ana, params))
    assert eff.dtype == torch.float64
    assert eff.ndim == 2 and eff.shape[1] == 2 and eff.shape[0] >= 1
    center, error = eff[:, 0].cpu(), eff[:, 1].cpu()
    assert torch.all(torch.abs(center - 1.0) < 1e-4), center.tolist()
    assert torch.all(error > 0), error.tolist()
    assert torch.all(error < 0.5), error.tolist()


def test_efficiency_missing_phsp_truth_returns_empty(make_analysis):
    """simple.yml 无 phsp_truth → 返回空张量 [0,2]（不抛异常，调用方自然跳过）。"""
    ana = make_analysis("simple")
    params = _params(ana)
    eff = ana.getEfficiency(_coupling(ana, params))
    assert tuple(eff.shape) == (0, 2)


def test_efficiency_double_arg_overload(make_analysis):
    """双参重载 (vector, hessian) 与单参结果一致（hessian 传空时等价）。"""
    ana = make_analysis("eff")
    params = _params(ana, seed=7)
    cpl = _coupling(ana, params)
    eff1 = ana.getEfficiency(cpl)
    eff2 = ana.getEfficiency(cpl, torch.empty(0, dtype=torch.float64, device=DEVICE))
    assert torch.allclose(eff1, eff2, atol=1e-10)


def test_fit_fractions_accepts_backend_hessian(make_analysis):
    """回归：hessian 检查必须走后端感知判断（CPU 构建传 CPU float64 Hessian 不能抛）。

    以前这里写死 `hessian_in.is_cuda()`，CPU 构建必抛
    "hessian_in must be float64 CUDA"。
    """
    ana = make_analysis("eff")
    params = _params(ana)
    cpl = _coupling(ana, params)
    H = ana.getHessian(params).double()          # 统一 Hessian (2n+θ)²
    ff = ana.getFitFractions(cpl, H)              # 内部取左上 [2n,2n] 耦合块
    assert ff.dtype == torch.float64
    assert ff.ndim == 2 and ff.shape[1] == 2 and ff.shape[0] >= 1
    center = ff[:, 0].double().cpu()
    assert abs(float(center.sum()) - 1.0) < 1e-5, center.tolist()
    assert torch.all(torch.isfinite(center))


def test_fit_fractions_missing_phsp_truth_early_return(make_analysis):
    """回归：缺 phsp_truth 时早退必须在 Tensor 检查**之前**返回空。

    调用方探测"有没有份额"时不该被 dtype/device 挡住：这里故意传错误 dtype 的
    Hessian（float32）和非法的空 vector，仍应安静返回 [0,2] 而不是抛错。
    """
    ana = make_analysis("simple")                # simple.yml 无 phsp_truth
    bad_h = torch.zeros(4, 4, dtype=torch.float32, device=DEVICE)
    v = torch.zeros(max(ana.getNVector(), 1), dtype=torch.complex128, device=DEVICE)
    ff = ana.getFitFractions(v, bad_h)
    assert tuple(ff.shape) == (0, 2)
    eff = ana.getEfficiency(v, bad_h)
    assert tuple(eff.shape) == (0, 2)


# ============================================================
# 逐事件权重: phsp_truth_weights / phsp_weights
#   truth 积分 = Σ w|A|², 归一化 = Σ w, MC 统计误差二阶矩 = Σ(w|A|²)²
# ============================================================
_TESTS_DIR = Path(__file__).resolve().parent
_CONFIGS = _TESTS_DIR / "configs"
_DATA = _TESTS_DIR / "data"


def _write_cfg(tmp_path, base, overrides, name):
    """复制 base.yml，替换 Data 段已有键 / 在 phsp_truth 后插入新键。"""
    pending = dict(overrides)
    out = []
    for line in (_CONFIGS / f"{base}.yml").read_text().splitlines():
        key = None
        if line.startswith("  ") and ":" in line:
            key = line[2:].split(":", 1)[0].strip()
        if key in pending:
            out.append(f"  {key}: {pending.pop(key)}")
        else:
            out.append(line)
    if pending:                        # 新键挂在 phsp_truth 那一行后面
        idx = next(i for i, l in enumerate(out) if l.startswith("  phsp_truth:"))
        for k, v in pending.items():
            out.insert(idx + 1, f"  {k}: {v}")
    path = tmp_path / name
    path.write_text("\n".join(out) + "\n")
    return path


def _truth_events(n_per_event=3):
    """eff.yml 的 Data.order 是 3 个末态粒子 → test_phsp.dat 每 3 行一个事例。"""
    lines = (_DATA / "test_phsp.dat").read_text().splitlines()
    return [lines[i:i + n_per_event] for i in range(0, len(lines), n_per_event)]


def test_truth_weights_uniform_equals_unweighted(tmp_path, make_analysis):
    """常数 truth 权重只改整体标度（Σw 相消）→ 份额/效率连误差都必须不变。

    用 eff2（双波）而不是 eff：单波时 FF≡1，任何权重都"相等"，测不出东西。
    """
    base = make_analysis("eff2")
    p0 = _params(base, seed=11)
    c0 = _coupling(base, p0)
    ff0, eff0 = base.getFitFractions(c0), base.getEfficiency(c0)

    ana = ctpwa.analysis(str(_write_cfg(tmp_path, "eff2",
                                        {"phsp_truth_weights": "[2.5]"},
                                        "eff2_truthw.yml")))
    p1 = _params(ana, seed=11)
    c1 = _coupling(ana, p1)
    ff1, eff1 = ana.getFitFractions(c1), ana.getEfficiency(c1)

    assert torch.allclose(ff0, ff1, rtol=1e-6, atol=1e-9), (ff0, ff1)
    assert torch.allclose(eff0, eff1, rtol=1e-6, atol=1e-9), (eff0, eff1)


def test_phsp_weights_uniform_equals_unweighted(tmp_path, make_analysis):
    """常数 phsp 权重进入效率分子（此前只进 NLL 归一化）→ 比值仍应不变。"""
    base = make_analysis("eff2")
    p0 = _params(base, seed=3)
    eff0 = base.getEfficiency(_coupling(base, p0))

    ana = ctpwa.analysis(str(_write_cfg(tmp_path, "eff2",
                                        {"phsp_weights": "[0.4]"},
                                        "eff2_phspw.yml")))
    p1 = _params(ana, seed=3)
    eff1 = ana.getEfficiency(_coupling(ana, p1))

    assert torch.allclose(eff0, eff1, rtol=1e-6, atol=1e-9), (eff0, eff1)


def test_truth_weights_equal_repeated_events(tmp_path, make_analysis):
    """整型权重 w ⇔ 把该事例重复 w 次：份额/效率的中心值必须一致。

    这验证了 Σ w|A|² / Σ w 的加权口径（含 Jacobian 扰动积分）实现正确。
    用 eff2（双波）才能让 FF 依赖 truth 分布；否则 FF≡1，权重被丢掉也测不出来
    —— 所以另外断言加权结果确实偏离等权结果。
    注意 MC 统计误差**不**等价：加权按重要性权重的二阶矩 Σ(w|A|²)²，
    重复事例则是多重数口径 Σ w|A|⁴ —— 本测试只比较中心值与 FF 参数误差。
    """
    events = _truth_events()
    w = [1 + (i % 3) for i in range(len(events))]

    wpath = tmp_path / "truth_w.dat"
    wpath.write_text("\n".join(str(x) for x in w) + "\n")
    dup = tmp_path / "truth_dup.dat"
    with dup.open("w") as f:
        for ev, wi in zip(events, w):
            for _ in range(wi):
                f.write("\n".join(ev) + "\n")

    ana_w = ctpwa.analysis(str(_write_cfg(
        tmp_path, "eff2", {"phsp_truth_weights": f'[dat, "{wpath}"]'}, "eff2_w.yml")))
    ana_d = ctpwa.analysis(str(_write_cfg(
        tmp_path, "eff2", {"phsp_truth": f'[dat, "{dup}"]'}, "eff2_dup.yml")))
    base = make_analysis("eff2")

    pw, pd = _params(ana_w, seed=5), _params(ana_d, seed=5)
    ffw = ana_w.getFitFractions(_coupling(ana_w, pw))
    ffd = ana_d.getFitFractions(_coupling(ana_d, pd))
    effw = ana_w.getEfficiency(_coupling(ana_w, pw))
    effd = ana_d.getEfficiency(_coupling(ana_d, pd))
    ff0 = base.getFitFractions(_coupling(base, _params(base, seed=5)))

    # 权重必须真的被用上：非均匀权重相对等权确实把 FF 移开了
    assert float((ffw[:, 0] - ff0[:, 0]).abs().max()) > 1e-4
    # 中心值: Σ w·f 与 Σ_{重复} f 只差浮点累加顺序
    assert torch.allclose(ffw[:, 0], ffd[:, 0], rtol=1e-8, atol=1e-12)
    assert torch.allclose(effw[:, 0], effd[:, 0], rtol=1e-8, atol=1e-12)
    # FF 参数误差: 同一个 Hessian × 同一批扰动积分 → 也一致
    assert torch.allclose(ffw[:, 1], ffd[:, 1], rtol=1e-6, atol=1e-9)
    assert torch.all(torch.isfinite(ffw)) and torch.all(torch.isfinite(effw))


def test_truth_weights_bad_file_raises(tmp_path):
    """权重文件缺失或长度不符必须报错 —— 静默按 0 权重会把份额算成全 0。"""
    ana = ctpwa.analysis(str(_write_cfg(tmp_path, "eff", {
        "phsp_truth_weights": f'[dat, "{tmp_path / "missing_w.dat"}"]'},
        "eff_missing_w.yml")))
    params = _params(ana)
    with pytest.raises(RuntimeError):
        ana.getFitFractions(_coupling(ana, params))

    short = tmp_path / "short_w.dat"
    short.write_text("1\n2\n3\n")
    ana2 = ctpwa.analysis(str(_write_cfg(tmp_path, "eff", {
        "phsp_truth_weights": f'[dat, "{short}"]'}, "eff_short_w.yml")))
    params2 = _params(ana2)
    with pytest.raises(RuntimeError):
        ana2.getEfficiency(_coupling(ana2, params2))


def test_truth_weights_root_matches_dat(tmp_path, monkeypatch):
    """ROOT truth（四动量 + weight 分支）⇔ 同一批事例的 DAT 版本。

    覆盖"用模型/生成器产生的 MC 直接当 PHSP MC"的完整链路:
    `PhaseSpaceGenerator.write(..., weights=w)` 写出的 "mc" 树同时带末态四动量
    分支与 weight 分支，config 的 phsp_truth / phsp_truth_weights 都指向它。
    批大小压到 700 → 2000 事例分 3 批，顺带回归 ROOT 路径的 offset 分批读取。
    """
    monkeypatch.setenv("CTPWA_TRUTH_BATCH", "700")
    gen = ctpwa.PhaseSpaceGenerator(str(_CONFIGS / "eff2.yml"))
    ev, w = gen.generateWeighted(2000, seed=7)

    root = tmp_path / "truth.root"
    gen.write(ev, str(root), w)          # tree "mc": 末态分支 + weight
    dat = tmp_path / "truth.dat"
    gen.write(ev, str(dat))              # 同一批事例，无权重
    wdat = tmp_path / "truth_w.dat"
    wdat.write_text("\n".join(f"{x:.17g}" for x in w.tolist()) + "\n")

    ana_root = ctpwa.analysis(str(_write_cfg(tmp_path, "eff2", {
        "phsp_truth": f'[ROOT, "{root}", "mc"]',
        "phsp_truth_weights": f'[ROOT, "{root}", "mc", "weight"]'}, "eff2_root.yml")))
    ana_dat = ctpwa.analysis(str(_write_cfg(tmp_path, "eff2", {
        "phsp_truth": f'[dat, "{dat}"]',
        "phsp_truth_weights": f'[dat, "{wdat}"]'}, "eff2_dat.yml")))

    pr, pd = _params(ana_root, seed=13), _params(ana_dat, seed=13)
    ffr = ana_root.getFitFractions(_coupling(ana_root, pr))
    ffd = ana_dat.getFitFractions(_coupling(ana_dat, pd))
    assert torch.allclose(ffr, ffd, rtol=1e-8, atol=1e-12), (ffr, ffd)


def test_fit_fraction_matrix_matches_api_and_total(make_analysis):
    """getFitFractionMatrix: 对角/非对角口径自洽 + Σ_i I_i + Σ_{i<j} I_ij = ∫|ΣA|²。

    同时在 eff2 上核对（该 config 的 phsp 与 phsp_truth 指向同一文件，因此
    驻留 phsp 振幅就是 truth 样本的振幅，|ΣA|² 可以直接用 getPhspTensor 重建）。
    """
    ana = make_analysis("eff2")
    params = _params(ana, seed=17)
    cpl = _coupling(ana, params)

    M = ana.getFitFractionMatrix(cpl)
    n = M.shape[0]
    assert M.dtype == torch.float64 and n == ana.getNVector()

    # 对称 + ctpwa 口径（非相干分母）与 getFitFractions 一致
    assert torch.allclose(M, M.T, rtol=0, atol=1e-9)
    ff = ana.getFitFractions(cpl)[:, 0]
    assert torch.allclose(M.diagonal() / M.diagonal().sum(), ff, rtol=1e-9, atol=0)

    # 相干分母: W = Σ_i M_ii + Σ_{i<j} M_ij（非对角已是 2Re，不能再全矩阵求和）
    W = float(M.diagonal().sum())
    for i in range(n):
        for j in range(i + 1, n):
            W += float(M[i][j])

    nsl = ana.getSLVectors().tolist()
    A = ana.getPhspTensor().reshape(-1, sum(nsl))
    v = ana.getExtendedVector(cpl).reshape(-1)
    S = A * v
    off = [0]
    for k in nsl:
        off.append(off[-1] + k)
    W_ref = float((abs(sum(S[:, off[i]:off[i + 1]].sum(1)
                         for i in range(len(nsl)))) ** 2).sum())
    assert abs(W - W_ref) / W_ref < 1e-10, (W, W_ref)

    # tf-pwa 口径: Σ_i FF_i + Σ_{i<j} FF_ij = 1（表闭合）
    ff_ij = sum(float(M[i][j]) for i in range(n) for j in range(i + 1, n)) / W
    assert abs((float(M.diagonal().sum()) / W) + ff_ij - 1.0) < 1e-12


def test_fit_fraction_matrix_empty_without_truth(make_analysis):
    """缺 phsp_truth → 返回空 [0,0]（与 getFitFractions 的早退一致）。"""
    ana = make_analysis("simple")
    cpl = _coupling(ana, _params(ana))
    M = ana.getFitFractionMatrix(cpl)
    assert tuple(M.shape) == (0, 0)


def test_fit_fraction_matrix_phsp_sample_matches_truth(make_analysis):
    """sample="phsp" 与 "truth"：eff2 两样本同文件 → 矩阵必须一致。

    真实分析里两者差的就是效率（P_ii = ε_i·T_ii）；这里锁住同源时的一致性，
    防止 sample 参数接错样本或权重。
    """
    ana = make_analysis("eff2")
    cpl = _coupling(ana, _params(ana, seed=23))
    M_t = ana.getFitFractionMatrix(cpl)                     # 默认 truth
    M_p = ana.getFitFractionMatrix(cpl, sample="phsp")
    assert tuple(M_p.shape) == tuple(M_t.shape)
    assert torch.allclose(M_t, M_p, rtol=1e-10, atol=1e-9)


def test_write_result_interference_equals_matrix(tmp_path, make_analysis):
    """writeResult 的 interference_rec == getFitFractionMatrix("phsp") / W_rec。

    ROOT 里存的是"重建侧（带效率 MC）的归一化表"，对角是 F_i^rec、非对角是
    I_ij^rec，分母 = W_rec = ∫ε|ΣA|²。文档 doc/branching_fraction_recipe.md 的
    换算就建立在这个恒等式上。
    """
    uproot = pytest.importorskip("uproot")
    ana = make_analysis("eff2")
    params = _params(ana, seed=29)
    cpl = _coupling(ana, params)

    M = ana.getFitFractionMatrix(cpl, sample="phsp")
    n = M.shape[0]
    W = float(M.diagonal().sum())
    for i in range(n):
        for j in range(i + 1, n):
            W += float(M[i][j])

    out = tmp_path / "wr.root"
    ana.writeResult(params.detach(), str(out), 0)
    with uproot.open(out) as f:
        flat = [float(x) for x in f["interference_rec"]._members["fElements"]]
    root_mat = torch.tensor(flat, dtype=torch.float64).reshape(n, n)

    assert torch.allclose(root_mat, M / W, rtol=1e-10, atol=1e-12), (root_mat, M / W)
    # 表闭合：Σ diag + Σ_{i<j} offdiag = 1（注意别用 M.sum()：对称矩阵的
    # 2Re 在两个三角各存一份，全矩阵求和会重复计入干涉项）
    closed = float(root_mat.diagonal().sum())
    for i in range(n):
        for j in range(i + 1, n):
            closed += float(root_mat[i][j])
    assert abs(closed - 1.0) < 1e-9, closed


def test_efficiency_equals_matrix_ratio(make_analysis):
    """getEfficiency 与矩阵 API 的互检：eff_i ≡ (P_ii/N_p)/(T_ii/N_t)。

    这条恒等式把"效率"和两个矩阵钉在一起（也就是 doc/branching_fraction_recipe.md
    里的 r_i）。注意 r_i = ε_i^abs/ε_all —— cut 后的样本分布 ∝ εdΦ/∫ε，
    样本内平均天然带 1/ε_all；eff2 两样本同文件（ε≡1）时 r_i 恰好是 ε_i。
    """
    ana = make_analysis("eff2")
    cpl = _coupling(ana, _params(ana, seed=31))
    eff = ana.getEfficiency(cpl)[:, 0]
    T = ana.getFitFractionMatrix(cpl)
    P = ana.getFitFractionMatrix(cpl, sample="phsp")
    # eff2：phsp 与 phsp_truth 同文件 → N_p == N_t
    ratio = P.diagonal() / T.diagonal()
    assert torch.allclose(eff, ratio, rtol=1e-10, atol=1e-12), (eff, ratio)
