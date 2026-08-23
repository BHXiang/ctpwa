import torch
import numpy as np
import time
import os
import ctpwa

# ============================================================
# 初始化分析对象
# ============================================================
int_time1 = int(time.time())
ana = ctpwa.analysis()
int_time2 = int(time.time())
print(f"振幅初始化耗时: {int_time2 - int_time1} 秒")

# 参数信息
conjugate_pairs = ana.getConstraintsIndex()
params_names = ana.getParamNames()          # 前 n_coupling_free 个耦合名 + 后 n_res_free 个共振态名
n_coupling_free = ana.getNVector()          # 自由耦合复数参数数

# 共振态参数信息
free_res_info = ana.getFreeResParams()      # [3, n_res] float64 CPU
n_res_free = free_res_info.shape[1]
HAS_FREE_RES = n_res_free > 0

n_params_total = 2 * n_coupling_free + n_res_free
print(f"耦合参数数量: {n_coupling_free}")
print(f"共振态参数数量: {n_res_free}")

# ============================================================
# 生成初始参数
# ============================================================
def generate_initial_params(
    n_coupling_free, free_res_info,
    seed=42, device="cuda"
):
    """生成统一参数向量 [real_coupling | imag_coupling | theta]

    Layout: [real_0,...,real_{n_free-1}, imag_0,...,imag_{n_free-1}, theta_0,...,theta_{n_res-1}]
    real_0=1.0, imag_0=0.0 为固定参考振幅
    """
    n_res = free_res_info.shape[1]
    n_total = 2 * n_coupling_free + n_res
    params = torch.zeros(n_total, dtype=torch.float64, device=device)

    torch.manual_seed(seed)

    # 耦合实部
    params[0] = 1.0  # 固定参考
    for idx in range(1, n_coupling_free):
        amplitude = torch.rand(1, device=device).item() * 0.5
        phase = torch.rand(1, device=device).item() * 2 * torch.pi
        params[idx] = amplitude * np.cos(phase)

    # 耦合虚部
    params[n_coupling_free] = 0.0  # 固定参考
    for idx in range(1, n_coupling_free):
        amplitude = torch.rand(1, device=device).item() * 0.5
        phase = torch.rand(1, device=device).item() * 2 * torch.pi
        params[n_coupling_free + idx] = amplitude * np.sin(phase)

    # 共振态参数
    if n_res > 0:
        init_vals = free_res_info[0].to(device=device, dtype=torch.float64)
        if seed > 42:
            torch.manual_seed(seed)
            lower = free_res_info[1].to(device=device, dtype=torch.float64)
            upper = free_res_info[2].to(device=device, dtype=torch.float64)
            noise = (torch.rand(n_res, device=device, dtype=torch.float64) - 0.5) * 0.1 * (upper - lower)
            init_vals = torch.clamp(init_vals + noise,
                                    lower + 1e-7 * (upper - lower),
                                    upper - 1e-7 * (upper - lower))
        params[2 * n_coupling_free:] = init_vals

    print(f"生成初始参数 (seed={seed}): n_coupling={n_coupling_free}, n_res={n_res}")
    return params


# ============================================================
# 构建自由耦合参数 -> 振幅下标的映射
# ============================================================
# ============================================================
# 优化器
# ============================================================
class UnifiedPWAOptimizer:
    def __init__(self, ana, free_res_info, params_names):
        self.analysis = ana
        self.params_names = params_names
        self.device = "cuda"
        self.best_nll = float("inf")
        self.best_params = None
        self.all_results = []

        self.n_coupling_free = ana.getNVector()
        self.n_res_free = free_res_info.shape[1]
        self.n_params = 2 * self.n_coupling_free + self.n_res_free
        self.has_free_res = self.n_res_free > 0
        # 耦合幅度上界（防 LBFGS 放飞，实测随机初值放开共振态参数时
        # 耦合会无界增长到 1e14 → 振幅溢出）+ 投影梯度（防边界伪收敛）。
        # ⚠️ 默认 10000 而不能更小: 不同波的振幅归一化差异极大
        # （ONE 模型/弱归一化波如 PHSP 需要 |v|~300-1000，实测本模型
        # 最佳解 |A| 达 297；v_max=50 会把它们掐死在墙上，模型形状被
        # 强制扭曲 → 拟合直接失败/正 NLL）。FIT_VMAX / FIT_PROJECT 可覆盖。
        self.v_max = float(os.environ.get("FIT_VMAX", "10000.0"))
        self.project_grad = os.environ.get("FIT_PROJECT", "1") == "1"
        # 统一 Hessian 缓存: 同参数点只在第一次真正计算一步 getHessian，
        # 后续（正定性判定/参数误差/分支比误差）直接复用。
        self._hess_cache = None  # (params.clone(), hessian_full)

        # 共振态参数bounds (GPU)
        if self.has_free_res:
            self._lower = free_res_info[1].to(dtype=torch.float64, device=self.device)
            self._upper = free_res_info[2].to(dtype=torch.float64, device=self.device)

    # --------------------------------------------------------
    def compute_loss_and_grad(self, params):
        """计算 NLL 和梯度。params: float64, [n_params]"""
        nc = self.n_coupling_free
        # 固定第一个耦合参数 (1+0j) + 耦合 clamp
        with torch.no_grad():
            params.data[0] = 1.0
            params.data[nc] = 0.0
            params.data[1:nc].clamp_(-self.v_max, self.v_max)
            params.data[nc + 1:2 * nc].clamp_(-self.v_max, self.v_max)

        # 共振态参数有界约束: clamp
        if self.has_free_res:
            with torch.no_grad():
                start = 2 * nc
                params.data[start:] = torch.clamp(
                    params.data[start:], self._lower, self._upper
                )

        nll = self.analysis.getNLL(params)
        grad = torch.autograd.grad(nll, params, retain_graph=False)[0]

        # 固定参数的梯度清零
        with torch.no_grad():
            grad[0] = 0.0
            grad[nc] = 0.0
            # 投影梯度: clamp 边界处指向边界外的梯度置零。
            # 否则 LBFGS 在边界处"参数不动、梯度非零" → tolerance_change
            # 伪收敛（实测停在正 NLL 的垃圾点）。FIT_PROJECT=0 可关闭。
            if self.project_grad:
                g_c = grad[1:nc]
                c = params[1:nc]
                g_c[(c <= -self.v_max) & (g_c < 0)] = 0.0
                g_c[(c >= self.v_max) & (g_c > 0)] = 0.0
                g_i = grad[nc + 1:2 * nc]
                ci = params[nc + 1:2 * nc]
                g_i[(ci <= -self.v_max) & (g_i < 0)] = 0.0
                g_i[(ci >= self.v_max) & (g_i > 0)] = 0.0
                if self.has_free_res:
                    res_start = 2 * nc
                    g_r = grad[res_start:]
                    phys = params[res_start:]
                    g_r[(phys <= self._lower) & (g_r < 0)] = 0.0
                    g_r[(phys >= self._upper) & (g_r > 0)] = 0.0

        return nll, grad

    # --------------------------------------------------------
    def optimize_single_run(
        self,
        initial_params,
        run_id=0,
        max_iter=500,
        lr=1.0,
        tolerance_grad=1e-8,
        tolerance_change=1e-10,
        history_size=100,
    ):
        """单次优化"""
        params = initial_params.clone().detach().requires_grad_(True)
        optimizer = torch.optim.LBFGS(
            [params],
            lr=lr,
            max_iter=max_iter,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            history_size=history_size,
            line_search_fn="strong_wolfe",
        )

        nll_history = []

        def closure():
            optimizer.zero_grad()
            nll, grad = self.compute_loss_and_grad(params)
            params.grad = grad
            nll_history.append(nll.item())
            return nll

        start_time = time.time()
        optimizer.step(closure)
        end_time = time.time()

        # 最终clamp一次确保共振态参数在界内 + 耦合在 ±v_max 内
        with torch.no_grad():
            params.data[0] = 1.0
            params.data[self.n_coupling_free] = 0.0
            params.data[1:self.n_coupling_free].clamp_(-self.v_max, self.v_max)
            params.data[self.n_coupling_free + 1:2 * self.n_coupling_free].clamp_(
                -self.v_max, self.v_max)
            if self.has_free_res:
                start = 2 * self.n_coupling_free
                params.data[start:] = torch.clamp(params.data[start:], self._lower, self._upper)

        final_nll = nll_history[-1] if nll_history else float("inf")
        final_params = params.clone().detach()

        # Hessian
        hessian_start = time.time()
        hessian_full = self.analysis.getHessian(final_params)
        # print("hessian矩阵：", hessian_full)
        hessian_time = time.time() - hessian_start
        # 播种缓存: 同一最佳点后续要 Hessian（正定性/误差/分支比）直接复用
        self._hess_cache = (final_params.clone(), hessian_full)

        # 去除固定参数 (index 0 和 n_coupling_free)
        fixed_mask = torch.ones(self.n_params, dtype=torch.bool, device=self.device)
        fixed_mask[0] = False
        fixed_mask[self.n_coupling_free] = False
        hessian = hessian_full[fixed_mask][:, fixed_mask]

        # 特征值分析
        try:
            eigenvalues = torch.linalg.eigvalsh(hessian)
            # print("Hessian特征值：", eigenvalues)
            is_pos_def = bool(torch.all(eigenvalues > 0).item())
            min_eig = eigenvalues[0].item()
            max_eig = eigenvalues[-1].item()
            cond_num = max_eig / min_eig if min_eig > 0 else float("inf")
        except Exception:
            is_pos_def = False
            min_eig = max_eig = cond_num = float("nan")

        # 参数误差
        coupling_real_errors = None
        coupling_imag_errors = None
        res_errors = None
        if is_pos_def:
            try:
                covariance = torch.linalg.inv(hessian)
                std_dev = torch.sqrt(torch.diag(covariance))

                # 耦合参数误差: 前 2*(n_coupling_free-1) 个元素
                n_c_var = self.n_coupling_free - 1  # 扣除固定的
                coupling_real_errors = torch.zeros(
                    self.n_coupling_free, dtype=torch.float32, device=self.device
                )
                coupling_imag_errors = torch.zeros(
                    self.n_coupling_free, dtype=torch.float32, device=self.device
                )

                # std_dev 前 2*n_c_var 个元素: 实部误差和虚部误差交替
                for i in range(n_c_var):
                    coupling_real_errors[i + 1] = std_dev[2 * i].float()
                    coupling_imag_errors[i + 1] = std_dev[2 * i + 1].float()

                # 共振态参数误差
                if self.has_free_res:
                    res_start = 2 * n_c_var
                    res_errors = std_dev[res_start:].float()
            except Exception as e:
                print(f"计算参数误差时出错: {e}")

        result = {
            "run_id": run_id,
            "final_params": final_params,
            "final_nll": final_nll,
            "nll_history": nll_history,
            "time": end_time - start_time,
            "hessian_time": hessian_time,
            "iterations": len(nll_history),
            "initial_params": initial_params.clone().detach(),
            "hessian_full": hessian_full,
            "hessian": hessian,
            "is_positive_definite": is_pos_def,
            "min_eigenvalue": min_eig,
            "max_eigenvalue": max_eig,
            "condition_number": cond_num,
            "coupling_real_errors": coupling_real_errors,
            "coupling_imag_errors": coupling_imag_errors,
            "res_errors": res_errors,
        }

        if final_nll < self.best_nll:
            self.best_nll = final_nll
            self.best_params = final_params.clone()
            self.best_result = result

        return result

    # --------------------------------------------------------
    def _project_params_(self, p):
        """把参数向量投影回可行域（固定参考 + 耦合 ±v_max + 共振态 bounds）"""
        with torch.no_grad():
            p.data[0] = 1.0
            p.data[self.n_coupling_free] = 0.0
            p.data[1:self.n_coupling_free].clamp_(-self.v_max, self.v_max)
            p.data[self.n_coupling_free + 1:2 * self.n_coupling_free].clamp_(
                -self.v_max, self.v_max)
            if self.has_free_res:
                res_start = 2 * self.n_coupling_free
                p.data[res_start:].clamp_(self._lower, self._upper)

    # --------------------------------------------------------
    def polish_damped_newton(self, params_phys, max_steps=40, tol=1e-6, lam0=1.0,
                             verbose=True):
        """LBFGS 之后的精确 Hessian 抛光（damped Newton / Levenberg-Marquardt 式）。

        params_phys: 完整参数 [Re_v, Im_v, θ_phys]（例: fit.py 的物理空间约定）。
        在自由参数子空间（mask 掉固定参考方向）解 (H + λI)·d = -g：
          - H 不正定时 λ 抬到 -λ_min + δ，保证方向可下降；
          - 每步后投影回可行域；目标不下降 → 增大 λ 重试（信任域语义）；
          - 收敛后 H 正定则再做一次纯 Newton 步收尾。
        实测: LBFGS（宽容差）停在 NLL=-673 的伪平坦点，抛光可到 -1175；
        对发散垃圾点也能救回（+903 → -1402）。
        返回 (params, nll, is_pos_def)。
        """
        dev = self.device
        mask = torch.ones(self.n_params, dtype=torch.bool, device=dev)
        mask[0] = False
        mask[self.n_coupling_free] = False
        n_free = int(mask.sum().item())

        best = params_phys.clone().detach()
        self._project_params_(best)
        nll_best = self.analysis.getNLL(best).item()
        lam = lam0

        for step in range(max_steps):
            H_full = self.analysis.getHessian(best)
            H = H_full[mask][:, mask]
            eig = torch.linalg.eigvalsh(H)
            eig_min = eig[0].item()

            p = best.clone().requires_grad_(True)
            nll = self.analysis.getNLL(p)
            g = torch.autograd.grad(nll, p)[0]
            g_m = g[mask]

            if eig_min > 1e-8:
                # 已正定：纯 Newton 一步收尾
                d_m = torch.linalg.solve(H, -g_m)
                cand = best.clone()
                cand[mask] = best[mask] + d_m
                self._project_params_(cand)
                nll_cand = self.analysis.getNLL(cand).item()
                if nll_cand < nll_best - tol:
                    best, nll_best = cand, nll_cand
                    if verbose:
                        print(f"[polish] step{step}: Newton accept, NLL={nll_best:.6f}")
                break

            lam = max(lam, -eig_min + 1e-6)
            I = torch.eye(n_free, dtype=torch.float64, device=dev)
            try:
                d_m = torch.linalg.solve(H + lam * I, -g_m)
            except Exception:
                d_m = torch.linalg.lstsq(H + lam * I, -g_m).solution
            cand = best.clone()
            cand[mask] = best[mask] + d_m
            self._project_params_(cand)
            nll_cand = self.analysis.getNLL(cand).item()
            if nll_cand < nll_best - tol:
                best, nll_best = cand, nll_cand
                lam = max(lam * 0.3, 1e-6)
                if verbose:
                    print(f"[polish] step{step}: accept λ={lam:.2e}, NLL={nll_best:.6f}")
            else:
                lam *= 10.0
                if verbose:
                    print(f"[polish] step{step}: reject, λ={lam:.2e}")
                if lam > 1e10:
                    break

        H_final = self.analysis.getHessian(best)
        H_f = H_final[mask][:, mask]
        eig_f = torch.linalg.eigvalsh(H_f)
        pd = bool((eig_f[0] > 1e-8).item())
        # 播种缓存: 抛光终点的 Hessian 供误差/分支比复用
        self._hess_cache = (best.clone(), H_final)
        if verbose:
            print(f"[polish] done: NLL={nll_best:.6f}, PD={pd}, "
                  f"min_eig={eig_f[0].item():.3e}, steps={step + 1}")
        return best, nll_best, pd

    # --------------------------------------------------------
    def _get_hessian_cached(self, params):
        """统一 Hessian（带缓存）: 参数与上次完全相同时直接复用，否则计算。
        fit.py 在多处（正定性/参数误差/分支比误差）会在同一最佳点上要 Hessian,
        只算一次即可。"""
        if (self._hess_cache is not None
                and self._hess_cache[0].shape == params.shape
                and torch.equal(self._hess_cache[0], params)):
            return self._hess_cache[1]
        h = self.analysis.getHessian(params)
        self._hess_cache = (params.clone(), h)
        return h

    # --------------------------------------------------------
    def compute_param_errors(self, params_phys):
        """在给定参数点用精确 Hessian 求参数误差（H 正定时有效）。
        返回 (coupling_real_errors, coupling_imag_errors, res_errors)。
        """
        hessian_full = self._get_hessian_cached(params_phys)
        fixed_mask = torch.ones(self.n_params, dtype=torch.bool, device=self.device)
        fixed_mask[0] = False
        fixed_mask[self.n_coupling_free] = False
        hessian = hessian_full[fixed_mask][:, fixed_mask]
        eig = torch.linalg.eigvalsh(hessian)
        if eig[0].item() <= 1e-8:
            return None, None, None
        try:
            covariance = torch.linalg.inv(hessian)
            std_dev = torch.sqrt(torch.diag(covariance))
            n_c_var = self.n_coupling_free - 1
            coupling_real_errors = torch.zeros(
                self.n_coupling_free, dtype=torch.float32, device=self.device)
            coupling_imag_errors = torch.zeros(
                self.n_coupling_free, dtype=torch.float32, device=self.device)
            for i in range(n_c_var):
                coupling_real_errors[i + 1] = std_dev[2 * i].float()
                coupling_imag_errors[i + 1] = std_dev[2 * i + 1].float()
            res_errors = None
            if self.has_free_res:
                res_start = 2 * n_c_var
                res_errors = std_dev[res_start:].float()
            return coupling_real_errors, coupling_imag_errors, res_errors
        except Exception as e:
            print(f"计算参数误差时出错: {e}")
            return None, None, None

    # --------------------------------------------------------
    def extract_coupling_complex(self, params):
        """从统一参数中提取复数耦合向量 (complex64, n_coupling_free)"""
        real = params[:self.n_coupling_free].float()
        imag = params[self.n_coupling_free:2 * self.n_coupling_free].float()
        return torch.complex(real, imag)

    def extract_theta_phys(self, params):
        """从统一参数中提取共振态物理参数"""
        if not self.has_free_res:
            return None
        return params[2 * self.n_coupling_free:]

    # --------------------------------------------------------
    def compute_fit_fractions(self, params=None):
        """拟合分数 (fit fractions): FF_i = ∫|A_i|² / Σ_j ∫|A_j|²。
        只用 phsp_truth（无效率 MC）→ 与效率/归一化无关，跨实验可比。
        误差传播用拟合同源的统一 Hessian（缓存命中直接复用）。"""
        if params is None:
            if self.best_params is None:
                print("没有优化结果!")
                return None, None
            params = self.best_params

        coupling = self.extract_coupling_complex(params)
        # 传拟合同源的全量 Hessian → FF 误差的正定性判定与拟合完全一致
        # （旧版内部用独立的 computeCouplingHessian, 近平坦方向
        #  min_eig~1e-5 时正定判定翻脸 → 误差被跳过变全 0）。
        hessian_full = self._get_hessian_cached(params)
        # getFitFractions 在配置无 phsp_truth 时天然返回空张量 [0,2]
        # （早期拟合不带 mctruth）→ 这里自然跳过, 不抛异常、不触碰相关 kernel
        ff_result = self.analysis.getFitFractions(coupling, hessian_full)
        if ff_result is None or ff_result.numel() == 0:
            print("跳过拟合分数: 配置没有 phsp_truth (无效率相空间 MC), "
                  "需要时再加入并重跑")
            return None, None
        return ff_result[:, 0], ff_result[:, 1]

    # --------------------------------------------------------
    def save_parameters(self, params, coupling_real_err, coupling_imag_err,
                        res_errors, run_id, filename_base):
        """保存所有参数到文件"""
        try:
            coupling = self.extract_coupling_complex(params)
            params_np = coupling.cpu().numpy()                         # [n_coupling_free]
            real_err_np = (coupling_real_err.cpu().numpy()
                           if coupling_real_err is not None
                           else np.zeros(self.n_coupling_free))
            imag_err_np = (coupling_imag_err.cpu().numpy()
                           if coupling_imag_err is not None
                           else np.zeros(self.n_coupling_free))

            if self.has_free_res:
                theta = self.extract_theta_phys(params)
                theta_np = theta.cpu().numpy()
                lower_np = self._lower.cpu().numpy()
                upper_np = self._upper.cpu().numpy()
                res_err_np = res_errors.cpu().numpy() if res_errors is not None else None

            txt_filename = f"{filename_base}.txt"
            if run_id == 0:
                with open(txt_filename, "w") as f:
                    f.write("# PWA Unified Parameters - All Runs\n")
                    f.write(f"# n_coupling_free: {self.n_coupling_free}\n")
                    f.write(f"# n_res_free: {self.n_res_free}\n")
                    f.write(f"# File generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("#" * 120 + "\n")
                    f.write("# Index  ParameterName                      ")
                    f.write("RealPart        RealError        ")
                    f.write("ImagPart        ImagError        ")
                    f.write("Magnitude       Phase(rad)      Phase(deg)\n")
                    f.write("#" * 120 + "\n")

            with open(txt_filename, "a") as f:
                f.write(f"# RUN: run_{run_id}\n")
                f.write("#" * 120 + "\n")
                for fi in range(self.n_coupling_free):
                    name = self.params_names[fi]
                    value = params_np[fi]
                    re_err = real_err_np[fi]
                    im_err = imag_err_np[fi]
                    magnitude = np.abs(value)
                    phase_rad = np.angle(value)
                    phase_deg = np.degrees(phase_rad)
                    f.write(
                        f"{fi:4d}  {name:50s}  "
                        f"{value.real:12.8f} ± {re_err:12.8f}  "
                        f"{value.imag:12.8f} ± {im_err:12.8f}  "
                        f"{magnitude:12.8f}  {phase_rad:12.8f}  {phase_deg:12.8f}\n"
                    )

                if self.has_free_res:
                    for i in range(self.n_res_free):
                        idx = self.n_coupling_free + i
                        name = self.params_names[idx]
                        err_str = f"± {res_err_np[i]:12.8f}" if res_err_np is not None else "             "
                        f.write(
                            f"{idx:4d}  {name:50s}  "
                            f"{theta_np[i]:12.8f}  {err_str}  "
                            f"bounds=[{lower_np[i]:.6g}, {upper_np[i]:.6g}]\n"
                        )
                f.write("#" * 120 + "\n")

            if run_id == 0:
                print(f"参数文件已创建: {txt_filename}")
            return True
        except Exception as e:
            print(f"保存参数失败: {e}")
            return False

    # --------------------------------------------------------
    def save_nll_history(self, nll_history, run_id, filename_base):
        try:
            txt_filename = f"{filename_base}.txt"
            if run_id == 0:
                with open(txt_filename, "w") as f:
                    f.write("# NLL History - All Runs\n")
                    f.write(f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("#" * 60 + "\n")
                    f.write("# Iteration  NLL\n")
                    f.write("#" * 60 + "\n")

            with open(txt_filename, "a") as f:
                f.write(f"# RUN: run_{run_id}\n")
                f.write("#" * 60 + "\n")
                for j, nll_val in enumerate(nll_history):
                    f.write(f"{j:8d}  {nll_val:15.8f}\n")
                f.write("#" * 60 + "\n")
            return True
        except Exception as e:
            print(f"保存NLL历史失败: {e}")
            return False

    # --------------------------------------------------------
    def save_weight_file(self, params, filename, waves=None):
        """保存权重文件。先reCalcAmp再writeResult。
        waves: 可选分波下标子集（如 [6,7]）, 只画 |Σ_{i∈S}A_i·v_i|² 的分布,
        空=全部。FIT_WAVES="6,7" 环境变量也可指定。
        FIT_EVENT_DATA=1 时 TTree 额外含末态四动量(任意分布按需现算);
        逐事件干涉 (interf_<i>_<j>) 请用 write_interf_result(pairs) 按需导出。"""
        try:
            if waves is None:
                w = os.environ.get("FIT_WAVES", "").strip()
                waves = [int(x) for x in w.split(",")] if w else []
            if self.has_free_res:
                theta = self.extract_theta_phys(params)
                self.analysis.reCalcAmp(theta)
            flag = 1 if os.environ.get("FIT_EVENT_DATA", "0") == "1" else 0
            self.analysis.writeResult(params, filename, flag, waves)
            if waves:
                print(f"权重文件已保存: {filename} (waves 子集: {waves}, "
                      f"直方图为 |Σ_{waves}A_i·v_i|²)")
            else:
                print(f"权重文件已保存: {filename}")
            return True
        except Exception as e:
            print(f"保存权重文件失败 {filename}: {e}")
            return False

    # --------------------------------------------------------
    def run_multiple_optimizations(self, num_runs=10, warm_start=None, **kwargs):
        results = []
        output_dir = "results"
        os.makedirs(output_dir, exist_ok=True)

        params_filename = os.path.join(output_dir, "parameters.txt")
        nll_filename = os.path.join(output_dir, "nll_history.txt")
        checkpoint = os.path.join(output_dir, "best_params.pt")

        for i in range(num_runs):
            print(f"\n{'='*80}")
            print(f"开始第 {i}/{num_runs-1} 次优化")
            print(f"{'='*80}")

            seed = 42 if i == 0 else 42 + i
            initial_params = generate_initial_params(
                self.n_coupling_free, free_res_info,
                seed=seed, device=self.device,
            )
            # warm start: run 0 的耦合取自收敛解（共振态参数保持 PDG 初值）。
            # 实测: 随机初值 + 放开共振态参数时 LBFGS 沿平坦方向放飞
            # （NLL 正值/撞边界）；从已收敛耦合出发则稳定收敛。
            # FIT_WARM=1 时自动用 results/best_params.pt 作为 warm start。
            if i == 0 and warm_start is not None:
                w = warm_start.to(self.device)
                initial_params[:2 * self.n_coupling_free] = w[:2 * self.n_coupling_free]
                print(f"warm start: 耦合来自收敛解 (NLL 优于随机初值的放飞解)")

            try:
                result = self.optimize_single_run(initial_params, run_id=i, **kwargs)
                results.append(result)
                self.all_results.append(result)

                print(f"第 {i} 次优化完成!")
                print(f"  NLL = {result['final_nll']:.6f}")
                print(f"  正定性 = {result['is_positive_definite']}")
                print(f"  耗时 = {result['time']:.2f}s, Hessian = {result['hessian_time']:.2f}s")
                print(f"  迭代次数 = {result['iterations']}")

                self.save_parameters(
                    result["final_params"],
                    result["coupling_real_errors"],
                    result["coupling_imag_errors"],
                    result["res_errors"],
                    i, params_filename.replace(".txt", ""),
                )

                self.save_nll_history(result["nll_history"], i,
                                      nll_filename.replace(".txt", ""))
                if result["final_nll"] <= self.best_nll:
                    torch.save(result["final_params"].cpu(), checkpoint)
            except Exception as e:
                print(f"第 {i} 次优化失败: {e}")
                import traceback
                traceback.print_exc()
                continue

        return results

    # --------------------------------------------------------
    def print_optimized_parameters(self, params=None, coupling_real_err=None,
                                   coupling_imag_err=None, res_errors=None,
                                   run_id=None):
        if params is None:
            if self.best_params is None:
                print("没有优化结果!")
                return
            params = self.best_params
            run_info = "最佳"
        else:
            run_info = f"第 {run_id} 次运行"

        coupling = self.extract_coupling_complex(params)
        params_np = coupling.cpu().numpy()                         # [n_coupling_free]
        real_err_np = (coupling_real_err.cpu().numpy()
                       if coupling_real_err is not None
                       else np.zeros(self.n_coupling_free))
        imag_err_np = (coupling_imag_err.cpu().numpy()
                       if coupling_imag_err is not None
                       else np.zeros(self.n_coupling_free))

        print(f"\n{'='*80}")
        print(f"{run_info}优化结果:")
        print(f"{'='*80}")
        print(f"固定参数: {self.params_names[0]} = 1.000000 + 0.000000i")

        for fi in range(1, self.n_coupling_free):
            name = self.params_names[fi]
            value = params_np[fi]
            re_err = real_err_np[fi]
            im_err = imag_err_np[fi]
            magnitude = np.abs(value)
            phase = np.angle(value)
            x, y = value.real, value.imag
            dx, dy = re_err, im_err
            mag_err = np.sqrt((x**2 * dx**2 + y**2 * dy**2) / (x**2 + y**2)) if magnitude > 0 else 0.0
            phase_err = np.sqrt((y**2 * dx**2 + x**2 * dy**2) / (x**2 + y**2)**2) if magnitude > 0 else 0.0
            print(
                f"{fi:3d}: {name:50s} = "
                f"({value.real:10.6f} ± {re_err:10.6f}) + "
                f"({value.imag:10.6f} ± {im_err:10.6f})i  "
                f"(|A|={magnitude:.6f} ± {mag_err:.6f}, "
                f"φ={np.degrees(phase):.2f}° ± {np.degrees(phase_err):.2f}°)"
            )

        # 共振态参数
        if self.has_free_res:
            theta = self.extract_theta_phys(params)
            print()
            theta_np = theta.cpu().numpy()
            lower_np = self._lower.cpu().numpy()
            upper_np = self._upper.cpu().numpy()
            res_err_np = res_errors.cpu().numpy() if res_errors is not None else None
            for j in range(self.n_res_free):
                idx = self.n_coupling_free + j
                name = self.params_names[idx]
                err_str = f" ± {res_err_np[j]:.6f}" if res_err_np is not None else ""
                print(f"{idx:3d}: {name:50s} = {theta_np[j]:12.8f}{err_str}"
                      f"  (bounds=[{lower_np[j]:.6g}, {upper_np[j]:.6g}])")

    # --------------------------------------------------------
    def save_all_results_summary(self, fit_values=None, fit_errors=None,
                                 fit_attempted=False):
        if not self.all_results:
            print("没有结果!")
            return

        sorted_results = sorted(self.all_results, key=lambda x: x["final_nll"])
        summary_file = "results/optimization_summary.txt"
        with open(summary_file, "w") as f:
            f.write("PWA优化结果\n")
            f.write("=" * 100 + "\n")
            f.write(f"总运行次数: {len(self.all_results)}\n")
            f.write(f"耦合参数数量: {self.n_coupling_free}\n")
            f.write(f"自由共振态参数: {self.n_res_free}\n")
            f.write(f"总参数维度: {self.n_params}\n")
            f.write(f"最佳NLL: {self.best_nll:.6f}\n")
            f.write(f"参数文件: parameters.txt\n")
            f.write(f"NLL历史: nll_history.txt\n\n")

            f.write("=" * 100 + "\n")
            f.write("运行结果 (按NLL排序):\n")
            f.write("=" * 100 + "\n")
            f.write(f"{'排名':<4} {'运行ID':<6} {'NLL':<12} {'迭代':<8} "
                    f"{'耗时':<10} {'Hessian耗时':<12} {'正定':<6}\n")
            f.write("-" * 100 + "\n")

            for rank, res in enumerate(sorted_results):
                f.write(f"{rank+1:<4} {res['run_id']:<6} {res['final_nll']:<12.6f} "
                        f"{res['iterations']:<8} {res['time']:<10.2f} "
                        f"{res['hessian_time']:<12.2f} "
                        f"{str(res['is_positive_definite']):<6}\n")

            if self.best_result.get("is_positive_definite", False):
                f.write("=" * 100 + "\n")
                f.write("最佳拟合分数 (fit fractions, 无效率/MC无关):\n")
                f.write("=" * 100 + "\n")
                try:
                    # 传入主程序已算好的结果，避免重复跑 truth 积分;
                    # fit_attempted=True 时主程序已处理(含 phsp_truth 缺失的跳过), 不再重试
                    if fit_values is None and not fit_attempted:
                        fit_values, fit_errors = self.compute_fit_fractions(self.best_params)
                    if fit_values is not None:
                        for i in range(len(fit_values)):
                            f.write(f"{i:2d}: {fit_values[i]:.6e} ± {fit_errors[i]:.6e}\n")
                except Exception as e:
                    f.write(f"计算拟合分数失败: {e}\n")

            if self.best_params is not None and self.has_free_res:
                f.write("=" * 100 + "\n")
                f.write("最佳共振态参数:\n")
                f.write("=" * 100 + "\n")
                theta = self.extract_theta_phys(self.best_params)
                theta_np = theta.cpu().numpy()
                lower_np = self._lower.cpu().numpy()
                upper_np = self._upper.cpu().numpy()
                f.write(f"{'Index':<6} {'Name':<30} {'Value':<16} {'Lower':<16} {'Upper':<16}\n")
                for i in range(self.n_res_free):
                    name = self.params_names[self.n_coupling_free + i]
                    f.write(f"{i:<6} {name:<30} {theta_np[i]:<16.8f} "
                            f"{lower_np[i]:<16.8f} {upper_np[i]:<16.8f}\n")

        print(f"优化结果摘要已保存到: {summary_file}")


# ============================================================
# 主程序
# ============================================================
optimizer = UnifiedPWAOptimizer(
    ana, free_res_info, params_names
)

# warm start: FIT_WARM=1 时用上次拟合的 results/best_params.pt
# （阶段式拟合: 先固定共振态参数拟合耦合, 再放开共振态参数 warm start,
#  可避免随机初值 + 放开共振态参数时的 LBFGS 放飞）
warm = None
if os.environ.get("FIT_WARM", "0") == "1":
    cp = os.path.join("results", "best_params.pt")
    if os.path.exists(cp):
        warm = torch.load(cp, weights_only=True)
        print(f"FIT_WARM: 从 {cp} 载入耦合作为 run 0 初值")

results = optimizer.run_multiple_optimizations(
    num_runs=1,
    max_iter=10000,
    lr=0.9,
    tolerance_grad=1e-10,
    tolerance_change=1e-10,
    history_size=500,
    warm_start=warm,
)

# 分析结果
print(f"\n{'='*80}")
print("所有优化结果总结:")
print(f"{'='*80}")

sorted_results = sorted(optimizer.all_results, key=lambda x: x["final_nll"])
for i, res in enumerate(sorted_results):
    print(f"运行 {res['run_id']:2d}: NLL = {res['final_nll']:12.6f}, "
          f"迭代 = {res['iterations']:3d}, "
          f"耗时 = {res['time']:6.2f}s, Hessian = {res['hessian_time']:6.2f}s, "
          f"正定 = {res['is_positive_definite']}")

print(f"\n{'='*80}")
print("最佳结果:")
print(f"{'='*80}")

best_res = sorted_results[0]
print(f"最佳NLL: {best_res['final_nll']:.6f} (来自第 {best_res['run_id']} 次运行)")
# print(f"Hessian正定性: {best_res['is_positive_definite']}")
# print(f"Hessian条件数: {best_res['condition_number']:.2e}")

# 精确 Hessian 抛光（FIT_POLISH=0 关闭）:
# LBFGS 常在宽容差/平坦方向提前停住（正定性=False）；
# damped Newton 用精确 Hessian 能继续大幅下降 NLL 并逼近正定极小值。
if os.environ.get("FIT_POLISH", "1") == "1":
    try:
        p2, nll2, pd2 = optimizer.polish_damped_newton(best_res["final_params"])
        if nll2 < best_res["final_nll"]:
            print(f"抛光: NLL {best_res['final_nll']:.6f} → {nll2:.6f} "
                  f"(Δ={nll2 - best_res['final_nll']:.3f}), 正定={pd2}")
            best_res["final_params"] = p2.clone()
            best_res["final_nll"] = nll2
            best_res["is_positive_definite"] = pd2
            if pd2:
                (best_res["coupling_real_errors"],
                 best_res["coupling_imag_errors"],
                 best_res["res_errors"]) = optimizer.compute_param_errors(p2)
            else:
                best_res["coupling_real_errors"] = None
                best_res["coupling_imag_errors"] = None
                best_res["res_errors"] = None
            torch.save(p2.cpu(), os.path.join("results", "best_params.pt"))
    except Exception as e:
        print(f"抛光失败: {e}")

# 打印最佳参数
if best_res["is_positive_definite"] and best_res["coupling_real_errors"] is not None:
    optimizer.print_optimized_parameters(
        best_res["final_params"],
        best_res["coupling_real_errors"],
        best_res["coupling_imag_errors"],
        best_res["res_errors"],
        best_res["run_id"],
    )
else:
    print(f"\n⚠️ Hessian矩阵不正定，无法提供参数误差估计")
    optimizer.print_optimized_parameters(best_res["final_params"])
print(f"{'='*80}")

# 保存最佳权重文件
best_weight_file = "results/weight_best.root"
optimizer.save_weight_file(best_res["final_params"], best_weight_file)

# 拟合分数（主输出: 无效率、与 MC 无关、跨实验可比; 只算一次,
# 主程序打印 + 摘要文件共用, 避免 truth 积分重复计算）
# 配置无 phsp_truth 时 getFitFractions 返回空张量 → compute_fit_fractions 自然跳过
ff_values = ff_errors = None
if best_res["is_positive_definite"]:
    try:
        ff_values, ff_errors = optimizer.compute_fit_fractions(
            best_res["final_params"]
        )
        if ff_values is not None:
            print(f"\n{'='*80}")
            print("最佳结果的拟合分数 (fit fractions, Σ=1, 无效率/MC无关):")
            print(f"{'='*80}")
            for i in range(len(ff_values)):
                print(f"{i:2d}: {ff_values[i]:.6f} ± {ff_errors[i]:.6f}")
    except Exception as e:
        print(f"计算拟合分数失败: {e}")

# 保存摘要
optimizer.save_all_results_summary(ff_values, ff_errors, fit_attempted=True)
