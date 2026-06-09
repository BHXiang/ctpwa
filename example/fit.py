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

        # 共振态参数bounds (GPU)
        if self.has_free_res:
            self._lower = free_res_info[1].to(dtype=torch.float64, device=self.device)
            self._upper = free_res_info[2].to(dtype=torch.float64, device=self.device)

    # --------------------------------------------------------
    def compute_loss_and_grad(self, params):
        """计算 NLL 和梯度。params: float64, [n_params]"""
        # 固定第一个耦合参数 (1+0j)
        with torch.no_grad():
            params.data[0] = 1.0
            params.data[self.n_coupling_free] = 0.0

        # 共振态参数有界约束: clamp
        if self.has_free_res:
            with torch.no_grad():
                start = 2 * self.n_coupling_free
                params.data[start:] = torch.clamp(
                    params.data[start:], self._lower, self._upper
                )

        nll = self.analysis.getNLL(params)
        grad = torch.autograd.grad(nll, params, retain_graph=False)[0]

        # 固定参数的梯度清零
        with torch.no_grad():
            grad[0] = 0.0
            grad[self.n_coupling_free] = 0.0

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

        # 最终clamp一次确保共振态参数在界内
        with torch.no_grad():
            params.data[0] = 1.0
            params.data[self.n_coupling_free] = 0.0
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
    def compute_branching_fractions(self, params=None):
        if params is None:
            if self.best_params is None:
                print("没有优化结果!")
                return None, None
            params = self.best_params

        coupling = self.extract_coupling_complex(params)
        bf_result = self.analysis.getBranchFractions(coupling)
        return bf_result[:, 0], bf_result[:, 1]

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
    def save_weight_file(self, params, filename):
        """保存权重文件。先reCalcAmp再writeResult"""
        try:
            if self.has_free_res:
                theta = self.extract_theta_phys(params)
                self.analysis.reCalcAmp(theta)
            self.analysis.writeResult(params, filename, 0)
            print(f"权重文件已保存: {filename}")
            return True
        except Exception as e:
            print(f"保存权重文件失败 {filename}: {e}")
            return False

    # --------------------------------------------------------
    def run_multiple_optimizations(self, num_runs=10, **kwargs):
        results = []
        output_dir = "results"
        os.makedirs(output_dir, exist_ok=True)

        params_filename = os.path.join(output_dir, "parameters.txt")
        nll_filename = os.path.join(output_dir, "nll_history.txt")

        for i in range(num_runs):
            print(f"\n{'='*80}")
            print(f"开始第 {i}/{num_runs-1} 次优化")
            print(f"{'='*80}")

            seed = 42 if i == 0 else 42 + i
            initial_params = generate_initial_params(
                self.n_coupling_free, free_res_info,
                seed=seed, device=self.device,
            )

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
    def save_all_results_summary(self):
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
                f.write("最佳分支比:\n")
                f.write("=" * 100 + "\n")
                try:
                    bf, bf_err = self.compute_branching_fractions(self.best_params)
                    if bf is not None:
                        for i in range(len(bf)):
                            f.write(f"{i:2d}: {bf[i]:.6e} ± {bf_err[i]:.6e}\n")
                except Exception as e:
                    f.write(f"计算分支比失败: {e}\n")

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

results = optimizer.run_multiple_optimizations(
    num_runs=1,
    max_iter=10000,
    lr=0.9,
    tolerance_grad=1e-10,
    tolerance_change=1e-10,
    history_size=500,
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

# 分支比
if best_res["is_positive_definite"]:
    try:
        bf_values, bf_errors = optimizer.compute_branching_fractions(
            best_res["final_params"]
        )
        if bf_values is not None:
            print(f"\n{'='*80}")
            print("最佳结果的分支比:")
            print(f"{'='*80}")
            for i in range(len(bf_values)):
                print(f"{i:2d}: {bf_values[i]:.6e} ± {bf_errors[i]:.6e}")
    except Exception as e:
        print(f"计算分支比失败: {e}")

# 保存摘要
optimizer.save_all_results_summary()
