import torch
import numpy as np
import time
import os
import argparse
import re
import ctpwa

# 初始化分析对象
int_time1 = int(time.time())
ana = ctpwa.analysis()
int_time2 = int(time.time())
print(f"振幅初始化耗时: {int_time2 - int_time1} 秒")

# 获取共轭对信息和振幅名称
conjugate_pairs = ana.getConstraintsIndex()
constraintValues = ana.getConstraintsValues()
amplitude_names = ana.getAmplitudeNames()
n_gls_total = ana.getNVector()

print(f"总振幅数量: {n_gls_total}")
print(f"可变参数数量: {n_gls_total - 1}")

# 显示共轭对信息
print("共轭对信息:")
for idx1, idx2 in conjugate_pairs:
    print(f"  {amplitude_names[idx1]} ({idx1}) <-> {amplitude_names[idx2]} ({idx2})")

n_polar = ana.getNPolarizations()
total_gls = n_gls_total + len(conjugate_pairs)


def generate_initial_params(
    n_gls_total, seed=44, device="cuda", initial_params_file=None, amplitude_names=None
):
    """生成初始参数 - 支持从文件加载或随机生成"""
    # 如果提供了初始参数文件，尝试加载
    if initial_params_file:
        params = ParameterLoader.load_from_file(
            initial_params_file, amplitude_names, device
        )
        if params is not None:
            print(f"使用文件中的初始参数: {initial_params_file}")
            return params
        else:
            print(f"无法从文件加载参数，将使用随机初始参数")

    # 随机生成初始参数
    torch.manual_seed(seed)

    # 创建初始参数 - 使用complex64（单精度复数）
    initial_params = torch.zeros(n_gls_total, dtype=torch.complex64, device=device)

    # 第一个参数固定为幅度1相位0 (1+0j)
    initial_params[0] = torch.complex(
        torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)
    )

    # 为其他参数生成随机值（幅度和相位）
    for idx in range(1, n_gls_total):
        # 幅度在 [0.1, 2.0] 范围内随机
        amplitude = torch.rand(1, device=device) * 0.5

        # 相位在 [0, 2π] 范围内随机
        phase = torch.rand(1, device=device) * 2 * torch.pi

        # 将幅度和相位转换为复数
        real_part = amplitude * torch.cos(phase)
        imag_part = amplitude * torch.sin(phase)

        initial_params[idx] = torch.complex(real_part, imag_part).squeeze()

    print(f"随机初始参数 (seed={seed})")
    return initial_params


class SimplePWAOptimizer:
    def __init__(self, ana, conjugate_pairs, amplitude_names):
        self.analysis = ana
        self.conjugate_pairs = conjugate_pairs
        self.amplitude_names = amplitude_names
        self.device = "cuda"
        self.best_nll = float("inf")
        self.best_params = None
        self.all_results = []  # 存储所有结果
        self.n_fixed = 1  # 固定参数数量
        self.n_variable = n_gls_total - self.n_fixed  # 可变参数数量
        self.n_real_variable = 2 * self.n_variable  # 可变实数参数数量 (实部+虚部)

    def compute_loss_and_grad(self, params):
        """计算损失和梯度 - 第一个参数固定为1+0j"""
        # 确保第一个参数保持为1+0j
        with torch.no_grad():
            params.data[0] = torch.complex(
                torch.tensor(1.0, device=self.device),
                torch.tensor(0.0, device=self.device),
            )

        # 计算NLL，C++会自动处理共轭
        nll = self.analysis.getNLL(params)

        # 计算梯度
        grad = torch.autograd.grad(nll, params, retain_graph=False)[0]

        # print("Params: ", params)
        # print("NLL: ", nll)
        # print("Gradiant: ", grad)

        # 确保第一个参数的梯度为0（因为我们固定了它）
        with torch.no_grad():
            grad[0] = torch.complex(
                torch.tensor(0.0, device=self.device),
                torch.tensor(0.0, device=self.device),
            )

        # # 添加L2正则化项（仅对可变参数）
        # lambda_reg = 1e-4  # 正则化强度，可以根据需要调整
        # reg_loss = lambda_reg * torch.sum(torch.abs(params[1:]) ** 2)
        # reg_grad = torch.zeros_like(grad)
        # reg_grad[1:] = 2 * lambda_reg * params[1:].conj()  # 复数参数的梯度

        # nll = nll + reg_loss
        # grad = grad + reg_grad

        return nll, grad

    def compute_branching_fractions(self, params=None):
        """使用C++端计算分支比，返回 (中心值, 误差)"""
        if params is None:
            if self.best_params is None:
                print("没有优化结果!")
                return None, None
            params = self.best_params

        bf_result = self.analysis.getBranchFractions(params)
        # 第一列是中心值，第二列是误差
        bf_values = bf_result[:, 0]
        bf_errors = bf_result[:, 1]
        return bf_values, bf_errors

    def optimize_single_run(
        self,
        initial_params,
        run_id=0,
        max_iter=500,
        lr=1.0,
        tolerance_grad=1e-5,
        tolerance_change=1e-7,
        history_size=100,
    ):
        """单次优化运行 - 第一个参数固定"""
        params = initial_params.clone().detach().requires_grad_(True)

        # 使用LBFGS优化器
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
            # 手动设置梯度
            params.grad = grad
            nll_history.append(nll.item())
            return nll

        start_time = time.time()
        optimizer.step(closure)
        end_time = time.time()

        final_nll = nll_history[-1] if nll_history else float("inf")
        final_params = params.clone().detach()

        # 计算Hessian矩阵 (使用C++端)
        hessian_start = time.time()
        hessian = self.analysis.getHessian(final_params)
        hessian_time = time.time() - hessian_start

        # 如果Hessian包含固定参数，去除之
        if hessian.shape[0] == 2 * n_gls_total:
            hessian = hessian[2:, 2:]

        # 计算特征值，判断正定性
        try:
            eigenvalues = torch.linalg.eigvalsh(hessian)
            is_pos_def = bool(torch.all(eigenvalues > 0).item())
        except torch._C._LinAlgError:
            print(f"警告: Hessian矩阵病态，无法计算特征值，视为不正定")
            eigenvalues = torch.zeros(hessian.shape[0], device=self.device)
            is_pos_def = False
        min_eig = eigenvalues[0].item()
        max_eig = eigenvalues[-1].item()
        cond_num = max_eig / min_eig if min_eig > 0 else float("inf")

        # 计算参数误差
        real_errors = None
        imag_errors = None
        if is_pos_def:
            try:
                covariance = torch.linalg.inv(hessian)
                std_dev = torch.sqrt(torch.diag(covariance))

                real_errors = torch.zeros(
                    n_gls_total, dtype=torch.float32, device=self.device
                )
                imag_errors = torch.zeros(
                    n_gls_total, dtype=torch.float32, device=self.device
                )
                for i in range(self.n_variable):
                    real_errors[i + 1] = std_dev[2 * i]
                    imag_errors[i + 1] = std_dev[2 * i + 1]
            except Exception as e:
                print(f"计算参数误差时出错: {e}")

        # 保存当前运行的结果
        result = {
            "run_id": run_id,
            "final_params": final_params,
            "final_nll": final_nll,
            "nll_history": nll_history,
            "time": end_time - start_time,
            "hessian_time": hessian_time,
            "iterations": len(nll_history),
            "initial_params": initial_params.clone().detach(),
            "hessian": hessian,
            "is_positive_definite": is_pos_def,
            "min_eigenvalue": min_eig,
            "max_eigenvalue": max_eig,
            "condition_number": cond_num,
            "real_errors": real_errors,
            "imag_errors": imag_errors,
        }

        # 更新最佳结果
        if final_nll < self.best_nll:
            self.best_nll = final_nll
            self.best_params = final_params.clone()
            self.best_result = result

        return result

    def save_parameters(
        self, params, real_errors, imag_errors, run_id, filename_base, save_txt=True
    ):
        """保存参数（包含误差）到.txt文件"""
        try:
            # 将参数和误差转换为numpy数组
            params_np = params.cpu().numpy()
            real_errors_np = (
                real_errors.cpu().numpy()
                if real_errors is not None
                else np.zeros_like(params_np.real)
            )
            imag_errors_np = (
                imag_errors.cpu().numpy()
                if imag_errors is not None
                else np.zeros_like(params_np.imag)
            )

            # 保存为.txt文件
            if save_txt:
                txt_filename = f"{filename_base}.txt"

                # 如果是第一次运行，创建文件并写入表头
                if run_id == 0:
                    with open(txt_filename, "w") as f:
                        # 写入文件头
                        f.write("# PWA Amplitude Parameters - All Runs\n")
                        f.write(f"# Total amplitudes: {len(params_np)}\n")
                        f.write(
                            f"# Fixed parameter: {self.amplitude_names[0]} = 1+0j\n"
                        )
                        f.write(
                            f"# File generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                        )
                        f.write("#" * 120 + "\n")
                        f.write("# Index  AmplitudeName                      ")
                        f.write("RealPart        RealError        ")
                        f.write("ImagPart        ImagError        ")
                        f.write("Magnitude       Phase(rad)      Phase(deg)\n")
                        f.write("#" * 120 + "\n")

                # 追加写入当前运行的参数
                with open(txt_filename, "a") as f:
                    # 写入每个参数
                    f.write(f"# RUN: run_{run_id}\n")
                    f.write("#" * 120 + "\n")  # 添加分隔线
                    for i, (name, value, re_err, im_err) in enumerate(
                        zip(
                            self.amplitude_names,
                            params_np,
                            real_errors_np,
                            imag_errors_np,
                        )
                    ):
                        magnitude = np.abs(value)
                        phase_rad = np.angle(value)
                        phase_deg = np.degrees(phase_rad)

                        # 格式化输出，包含误差
                        f.write(
                            f"{i:4d}  {name:50s}  "
                            f"{value.real:12.8f} ± {re_err:12.8f}  "
                            f"{value.imag:12.8f} ± {im_err:12.8f}  "
                            f"{magnitude:12.8f}  {phase_rad:12.8f}  {phase_deg:12.8f}\n"
                        )
                    f.write("#" * 120 + "\n")  # 添加分隔线

                if run_id == 0:
                    print(f"✅ 创建参数文件: {txt_filename}")

            return True
        except Exception as e:
            print(f"❌ 保存参数失败: {e}")
            return False

    def save_nll_history(self, nll_history, run_id, filename_base):
        """保存NLL历史到txt文件"""
        try:
            txt_filename = f"{filename_base}.txt"

            # 如果是第一次运行，创建文件并写入表头
            if run_id == 0:
                with open(txt_filename, "w") as f:
                    f.write("# NLL History - All Runs\n")
                    f.write(
                        f"# File generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    )
                    f.write("#" * 60 + "\n")
                    f.write("# Iteration  NLL\n")
                    f.write("#" * 60 + "\n")

            # 追加写入当前运行的NLL历史
            with open(txt_filename, "a") as f:
                f.write(f"# RUN: run_{run_id}\n")
                f.write("#" * 60 + "\n")  # 添加分隔线
                for iter_idx, nll_val in enumerate(nll_history):
                    f.write(f"{iter_idx:8d}  {nll_val:15.8f}\n")
                f.write("#" * 60 + "\n")  # 添加分隔线

            return True
        except Exception as e:
            print(f"❌ 保存NLL历史失败: {e}")
            return False

    def save_weight_file(self, params, filename):
        """保存权重文件"""
        try:
            self.analysis.writeResult(params, filename, 0)
            print(f"✅ 权重文件已保存: {filename}")
            return True
        except Exception as e:
            print(f"❌ 保存权重文件失败 {filename}: {e}")
            return False

    def run_multiple_optimizations(self, num_runs=10, **kwargs):
        """运行多次优化"""
        results = []

        # 创建输出目录
        output_dir = "results"
        os.makedirs(output_dir, exist_ok=True)

        # 定义统一的输出文件名
        params_filename = os.path.join(output_dir, "parameters.txt")
        nll_filename = os.path.join(output_dir, "nll_history.txt")

        for i in range(num_runs):
            print(f"\n{'='*80}")
            print(f"开始第 {i}/{num_runs-1} 次优化")
            print(f"{'='*80}")

            # 生成初始参数
            if i == 0:
                seed = 42  # 固定种子以便可重复
            else:
                seed = 42 + i

            initial_params = generate_initial_params(
                n_gls_total,
                seed=seed,
                device=self.device,
                amplitude_names=self.amplitude_names,
            )

            # 运行优化
            try:
                result = self.optimize_single_run(initial_params, run_id=i, **kwargs)
                results.append(result)
                self.all_results.append(result)

                print(f"第 {i} 次优化完成!")
                print(f"最终NLL: {result['final_nll']:.6f}")
                print(f"正定性: {result['is_positive_definite']}")
                print(f"优化耗时: {result['time']:.2f} 秒")
                print(f"海森耗时: {result['hessian_time']:.2f} 秒")
                print(f"迭代次数: {result['iterations']}")

                # 保存参数（含误差）
                self.save_parameters(
                    result["final_params"],
                    result["real_errors"],
                    result["imag_errors"],
                    i,
                    params_filename.replace(".txt", ""),
                    save_txt=True,
                )

                # 保存NLL历史
                self.save_nll_history(
                    result["nll_history"], i, nll_filename.replace(".txt", "")
                )
            except Exception as e:
                print(f"第 {i} 次优化失败: {e}")
                continue

        return results

    def print_optimized_parameters(
        self, params=None, real_errors=None, imag_errors=None, run_id=None
    ):
        """打印优化后的参数（包含误差）"""
        if params is None:
            if self.best_params is None:
                print("没有优化结果!")
                return
            params = self.best_params
            run_info = "最佳"
        else:
            run_info = f"第 {run_id} 次运行"

        print(f"\n{'='*80}")
        print(f"{run_info}优化后的参数值:")
        print(f"{'='*80}")
        print(
            f"固定参数: {self.amplitude_names[0]} = 1.000000 ± 0.000000 + 0.000000 ± 0.000000i\n"
        )

        params_np = params.cpu().numpy()
        real_errors_np = (
            real_errors.cpu().numpy()
            if real_errors is not None
            else np.zeros_like(params_np.real)
        )
        imag_errors_np = (
            imag_errors.cpu().numpy()
            if imag_errors is not None
            else np.zeros_like(params_np.imag)
        )

        for i, (name, value, re_err, im_err) in enumerate(
            zip(amplitude_names, params_np, real_errors_np, imag_errors_np)
        ):
            if i == 0:
                continue  # 跳过固定参数
            magnitude = np.abs(value)
            phase = np.angle(value)

            # 计算幅度和相位的误差（通过误差传播）
            x, y = value.real, value.imag
            dx, dy = re_err, im_err

            # 幅度误差: σ_r = sqrt((x²σ_x² + y²σ_y²)/(x²+y²))
            if magnitude > 0:
                mag_err = np.sqrt((x**2 * dx**2 + y**2 * dy**2) / (x**2 + y**2))
            else:
                mag_err = 0.0

            # 相位误差: σ_φ = sqrt((y²σ_x² + x²σ_y²)/(x²+y²)²)
            if magnitude > 0:
                phase_err = np.sqrt((y**2 * dx**2 + x**2 * dy**2) / (x**2 + y**2) ** 2)
                phase_err_deg = np.degrees(phase_err)
            else:
                phase_err = 0.0
                phase_err_deg = 0.0

            print(
                f"{i:3d}: {name:50s} = "
                f"({value.real:10.6f} ± {re_err:10.6f}) + "
                f"({value.imag:10.6f} ± {im_err:10.6f})i  "
                f"(|A|={magnitude:.6f} ± {mag_err:.6f}, "
                f"φ={np.degrees(phase):<.2f}° ± {phase_err_deg:.2f}°)"
            )

    def save_all_results_summary(self):
        """保存所有结果的摘要"""
        if not self.all_results:
            print("没有结果!")
            return

        # 按NLL排序
        sorted_results = sorted(self.all_results, key=lambda x: x["final_nll"])

        summary_file = "results/optimization_summary.txt"
        with open(summary_file, "w") as f:
            f.write("PWA优化结果\n")
            f.write("=" * 100 + "\n")
            f.write(f"总运行次数: {len(self.all_results)}\n")
            f.write(f"总振幅数: {total_gls}\n")
            f.write(f"可变参数: {self.n_variable}\n")
            f.write(f"最佳NLL: {self.best_nll:.6f}\n")
            f.write(f"所有参数结果: parameters.txt\n")
            f.write(f"所有NLL结果: nll_history.txt\n\n")

            f.write("=" * 100 + "\n")
            f.write("运行结果 (按NLL排序):\n")
            f.write("=" * 100 + "\n")
            f.write(
                f"{'排名':<4} {'运行ID':<6} {'NLL':<12} {'迭代次数':<8} "
                f"{'耗时(秒)':<10} {'Hessian耗时':<12} {'正定':<6}\n"
            )
            f.write("-" * 100 + "\n")

            for rank, res in enumerate(sorted_results):
                f.write(
                    f"{rank+1:<4} {res['run_id']:<6} {res['final_nll']:<12.6f} "
                    f"{res['iterations']:<8} {res['time']:<10.2f} "
                    f"{res['hessian_time']:<12.2f} {str(res['is_positive_definite']):<6}\n"
                )

            # 写入最佳结果的分支比
            if self.best_result.get("is_positive_definite", False):
                f.write("=" * 100 + "\n")
                f.write("最佳结果的分支比:\n")
                f.write("=" * 100 + "\n")
                try:
                    bf, bf_err = self.compute_branching_fractions(self.best_params)
                    if bf is not None:
                        for i in range(len(bf)):
                            f.write(f"{i:2d}: {bf[i]:.6e} ± {bf_err[i]:.6e}\n")
                except Exception as e:
                    f.write(f"计算分支比失败: {e}\n")
        print(f"✅ 优化结果摘要已保存到: {summary_file}")


# 创建优化器实例
optimizer = SimplePWAOptimizer(ana, conjugate_pairs, amplitude_names)

# 运行多次优化
results = optimizer.run_multiple_optimizations(
    num_runs=10,
    max_iter=10000,
    lr=0.5,
    tolerance_grad=1e-5,
    tolerance_change=1e-7,
    history_size=500,
)

# 分析结果
print(f"\n{'='*80}")
print("所有优化结果总结:")
print(f"{'='*80}")

# 按NLL排序
sorted_results = sorted(optimizer.all_results, key=lambda x: x["final_nll"])

for i, res in enumerate(sorted_results):
    print(
        f"运行 {res['run_id']:2d}: NLL = {res['final_nll']:12.6f}, "
        f"迭代次数 = {res['iterations']:3d}, "
        f"耗时 = {res['time']:6.2f}s, 海森耗时 = {res['hessian_time']:6.2f}s, "
        f"正定 = {res['is_positive_definite']}"
    )

print(f"\n{'='*80}")
print("最佳结果:")
print(f"{'='*80}")

best_res = sorted_results[0]
print(f"最佳NLL: {best_res['final_nll']:.6f} (来自第 {best_res['run_id']} 次运行)")
print(f"Hessian正定性: {best_res['is_positive_definite']}")
# print(f"Hessian最小特征值: {best_res['min_eigenvalue']:.2e}")
# print(f"Hessian最大特征值: {best_res['max_eigenvalue']:.2e}")
# print(f"Hessian条件数: {best_res['condition_number']:.2e}")
# print(f"Hessian计算耗时: {best_res['hessian_time']:.2f} 秒")

# 打印最佳参数（含误差）
if best_res["is_positive_definite"] and best_res["real_errors"] is not None:
    optimizer.print_optimized_parameters(
        best_res["final_params"],
        best_res["real_errors"],
        best_res["imag_errors"],
        best_res["run_id"],
    )
else:
    print(f"\n⚠️ Hessian矩阵不正定，无法提供参数误差估计")
    optimizer.print_optimized_parameters(best_res["final_params"])

"""
# 计算并打印分支比
if best_res["is_positive_definite"]:
    try:
        bf_values, bf_errors = optimizer.compute_branching_fractions(best_res["final_params"])
        if bf_values is not None:
            print(f"\n{'='*80}")
            print("最佳结果的分支比:")
            print(f"{'='*80}")
            for i in range(len(bf_values)):
                print(f"{i:2d}: {bf_values[i]:.6e} ± {bf_errors[i]:.6e}")
    except Exception as e:
        print(f"计算分支比失败: {e}")
"""

# 只保存最佳权重文件
best_weight_file = "results/weight_best.root"
optimizer.save_weight_file(best_res["final_params"], best_weight_file)

# 保存所有结果摘要
optimizer.save_all_results_summary()
