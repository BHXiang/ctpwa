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
print(f"共轭对数量: {len(conjugate_pairs)}")
print(f"可变参数数量: {n_gls_total - 1}")

# 显示共轭对信息
print("共轭对信息:")
for idx1, idx2 in conjugate_pairs:
    print(f"  {amplitude_names[idx1]} ({idx1}) <-> {amplitude_names[idx2]} ({idx2})")

# 获取数据张量
phsp = ana.getPhspTensor()
data = ana.getDataTensor()
bkg = ana.getBkgTensor()
bkg_wt = ana.getBkgWeightsTensor()

n_polar = ana.getNPolarizations()
total_gls = n_gls_total + len(conjugate_pairs)
Ndata = int(len(data) / n_polar / total_gls)
Nbkg = int(len(bkg) / n_polar / total_gls)


def generate_initial_params(
    n_gls_total, seed=42, device="cuda", initial_params_file=None, amplitude_names=None
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
        self.all_matrices = {}  # 存储所有矩阵结果
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

        # 确保第一个参数的梯度为0（因为我们固定了它）
        with torch.no_grad():
            grad[0] = torch.complex(
                torch.tensor(0.0, device=self.device),
                torch.tensor(0.0, device=self.device),
            )

        return nll, grad

    def compute_hessian(self, params):
        """计算Hessian矩阵 - 第一个参数固定为1+0j"""
        real_origin_vector_without_fixed = torch.view_as_real(params).flatten()

        # 使用 torch.autograd.functional.hessian
        # 将复数参数转换为实数参数，但排除第一个固定参数
        def real_loss_fn(real_params):
            # 第一个参数固定为1+0j，所以从第二个参数开始
            # 构建完整的参数向量，第一个参数固定为1+0j
            complex_vector = torch.view_as_complex(
                torch.cat(
                    [
                        torch.tensor([[1.0, 0.0]], device=phsp.device),
                        real_params.view(-1, 2),
                    ],
                    dim=0,
                )
            )
            vector = torch.zeros(
                n_gls_total + len(conjugate_pairs), dtype=torch.complex64, device="cuda"
            )
            # 填充实部和虚部
            for i in range(len(complex_vector)):
                # print(f"Index {i}, Value: {origin_vector[i]}")
                vector[i] = complex_vector[i]

            # 拓展vector以包含共轭对
            for i in range(n_gls_total - len(conjugate_pairs), n_gls_total):
                j = conjugate_pairs[i - (n_gls_total - len(conjugate_pairs))][1]
                # print(f"Index {i} is conjugate of index {j}")
                vector[j] = -1 * vector[i]

            # 重新计算损失
            phsp_factor_new = torch.sum(
                torch.abs(phsp.view(total_gls, -1).t() @ vector) ** 2
            ) / (len(phsp) / total_gls / n_polar)

            nll_new = -torch.sum(
                torch.log(
                    torch.sum(
                        torch.abs(
                            torch.sum(
                                data.view(total_gls, Ndata, n_polar)
                                * vector.unsqueeze(1).unsqueeze(2),
                                dim=0,
                            )
                        )
                        ** 2,
                        dim=1,
                    )
                    / phsp_factor_new
                )
            )
            if len(bkg_wt) == Nbkg:
                nll_bkg_new = -torch.sum(
                    bkg_wt
                    * torch.log(
                        torch.sum(
                            torch.abs(
                                torch.sum(
                                    bkg.view(total_gls, Nbkg, n_polar)
                                    * vector.unsqueeze(1).unsqueeze(2),
                                    dim=0,
                                )
                            )
                            ** 2,
                            dim=1,
                        )
                        / phsp_factor_new
                    )
                )
            else:
                nll_bkg_new = -torch.sum(
                    torch.log(
                        torch.sum(
                            torch.abs(
                                torch.sum(
                                    bkg.view(total_gls, Nbkg, n_polar)
                                    * vector.unsqueeze(1).unsqueeze(2),
                                    dim=0,
                                )
                            )
                            ** 2,
                            dim=1,
                        )
                        / phsp_factor_new
                    )
                )

            return nll_new - nll_bkg_new

        hessian = torch.autograd.functional.hessian(
            real_loss_fn, real_origin_vector_without_fixed
        )
        return hessian

    def check_positive_definite(self, matrix, tol=1e-8):
        """检查矩阵是否正定"""
        try:
            # 计算特征值
            eigenvalues = torch.linalg.eigvalsh(matrix)  # 使用eigvalsh处理实对称矩阵
            # 检查所有特征值是否大于0
            is_positive_definite = torch.all(eigenvalues > tol)
            min_eigenvalue = torch.min(eigenvalues).item()
            max_eigenvalue = torch.max(eigenvalues).item()
            condition_number = (
                max_eigenvalue / min_eigenvalue if min_eigenvalue > 0 else float("inf")
            )

            return (
                is_positive_definite.item(),
                min_eigenvalue,
                max_eigenvalue,
                condition_number,
            )
        except Exception as e:
            print(f"检查正定性时出错: {e}")
            return False, 0, 0, float("inf")

    def compute_covariance_matrix(self, hessian):
        """从Hessian矩阵计算协方差矩阵"""
        try:
            # 协方差矩阵是Hessian矩阵的逆
            covariance = torch.linalg.inv(hessian)
            return covariance
        except Exception as e:
            print(f"计算协方差矩阵时出错: {e}")
            return None

    def compute_correlation_matrix(self, covariance):
        """从协方差矩阵计算相关系数矩阵"""
        try:
            # 获取标准差（协方差矩阵对角线的平方根）
            std_dev = torch.sqrt(torch.diag(covariance))

            # 计算相关系数矩阵
            correlation = covariance / torch.outer(std_dev, std_dev)

            # 将对角线设为1（防止数值误差）
            n = correlation.shape[0]
            correlation[range(n), range(n)] = 1.0

            return correlation
        except Exception as e:
            print(f"计算相关系数矩阵时出错: {e}")
            return None

    def compute_parameter_errors(self, covariance, params):
        """计算参数误差"""
        try:
            # 获取标准差
            std_dev = torch.sqrt(torch.diag(covariance))

            # 将实数参数误差映射回复数参数
            # 实数参数顺序：[实部1, 虚部1, 实部2, 虚部2, ...]
            n_complex = self.n_variable
            real_errors = torch.zeros(
                n_complex, dtype=torch.float32, device=self.device
            )
            imag_errors = torch.zeros(
                n_complex, dtype=torch.float32, device=self.device
            )

            for i in range(n_complex):
                real_errors[i] = std_dev[2 * i]
                imag_errors[i] = std_dev[2 * i + 1]

            # 第一个参数（固定参数）的误差为0
            full_real_errors = torch.zeros(
                n_gls_total, dtype=torch.float32, device=self.device
            )
            full_imag_errors = torch.zeros(
                n_gls_total, dtype=torch.float32, device=self.device
            )
            full_real_errors[1:] = real_errors
            full_imag_errors[1:] = imag_errors

            return full_real_errors, full_imag_errors
        except Exception as e:
            print(f"计算参数误差时出错: {e}")
            return None, None

    def compute_branching_fractions(self, params=None):
        """计算并打印分支比率"""
        print("计算分支比...")
        total_gls = n_gls_total + len(conjugate_pairs)
        if params is None:
            if self.best_params is None:
                print("没有优化结果!")
                return
            params = self.best_params

        slvector = ana.getSLVectors()
        truth = ana.getTruthTensor()

        vector = torch.zeros(
            n_gls_total + len(conjugate_pairs),
            dtype=torch.complex64,
            device=phsp.device,
        )
        # 填充实部和虚部
        for i in range(len(params)):
            # print(f"Index {i}, Value: {origin_vector[i]}")
            vector[i] = params[i]

        # 拓展vector以包含共轭对
        for i in range(n_gls_total - len(conjugate_pairs), n_gls_total):
            j = conjugate_pairs[i - (n_gls_total - len(conjugate_pairs))][1]
            # print(f"Index {i} is conjugate of index {j}")
            vector[j] = -1 * vector[i]

        def compute_scattering_matrix(total_amp, slvector):
            """
            向量化版本计算实数散射矩阵

            Args:
                total_amp: 形状为 (total_gls, Nphsp*n_polar) 的复数张量
                slvector: 形状为 (num_particles,) 的张量

            Returns:
                real_matrix: 形状为 (num_particles, num_particles) 的实数对称矩阵
            """
            num_particles = slvector.size(0)
            total_gls = total_amp.size(0)

            # 1. 构建分波到粒子的映射矩阵
            wave_to_particle = torch.zeros(
                num_particles, total_gls, dtype=torch.complex64, device=total_amp.device
            )

            start_idx = 0
            for i in range(num_particles):
                n_waves = int(slvector[i])
                end_idx = start_idx + n_waves
                wave_to_particle[i, start_idx:end_idx] = 1.0 + 0j
                start_idx = end_idx

            # 2. 将振幅按粒子分组
            # (num_particles, total_gls) × (total_gls, N) → (num_particles, N)
            particle_amps = wave_to_particle @ total_amp

            # 3. 计算复数散射矩阵 S = A A^†
            # (num_particles, N) × (N, num_particles) → (num_particles, num_particles)
            complex_matrix = particle_amps @ particle_amps.conj().t()

            # 4. 转换为实数矩阵: 2 * Re(S)
            # 注意：由于 S_ij = A_i A_j^†，且 S_ji = S_ij^*，所以 S_ij + S_ij^* = 2 * Re(S_ij)
            real_matrix = 2 * complex_matrix.real

            return real_matrix

        def compute_branching_fraction(real_params):
            # 复数参数转换
            complex_vector = torch.view_as_complex(real_params.view(-1, 2))
            Ntruth = truth.shape[0] // total_gls // n_polar

            truth_amp = truth.view(
                total_gls, Ntruth * n_polar
            ) * complex_vector.unsqueeze(1)
            # phsp_amp = phsp.view(total_gls, Nphsp*n_polar)*complex_vector.unsqueeze(1)

            inter_truth = compute_scattering_matrix(truth_amp, slvector)

            ############### 计算branching fraction ###############
            eff_tot = torch.sum(
                torch.abs(phsp.view(total_gls, -1).t() @ complex_vector) ** 2
            ) / torch.sum(
                torch.abs(truth.view(total_gls, -1).t() @ complex_vector) ** 2
            )
            bf = (
                inter_truth.diag()
                * (Ndata - Nbkg)
                / eff_tot
                / torch.sum(
                    torch.abs(truth.view(total_gls, -1).t() @ complex_vector) ** 2
                )
            )
            return bf

        real_vector = torch.view_as_real(vector).flatten()
        bf_values = compute_branching_fraction(real_vector)
        jacobian = torch.autograd.functional.jacobian(
            compute_branching_fraction, real_vector
        )

        def real_loss_fn(real_params):
            # 第一个参数固定为1+0j，所以从第二个参数开始
            # 构建完整的参数向量，第一个参数固定为1+0j
            # complex_vector = torch.view_as_complex(torch.cat([torch.tensor([[1.0, 0.0]], device=phsp.device), real_params.view(-1, 2)], dim=0))
            complex_vector = torch.view_as_complex(real_params.view(-1, 2))

            # 重新计算损失
            phsp_factor_new = torch.sum(
                torch.abs(phsp.view(total_gls, -1).t() @ complex_vector) ** 2
            ) / (len(phsp) / total_gls / n_polar)

            nll_new = -torch.sum(
                torch.log(
                    torch.sum(
                        torch.abs(
                            torch.sum(
                                data.view(total_gls, Ndata, n_polar)
                                * complex_vector.unsqueeze(1).unsqueeze(2),
                                dim=0,
                            )
                        )
                        ** 2,
                        dim=1,
                    )
                    / phsp_factor_new
                )
            )
            if len(bkg_wt) == Nbkg:
                nll_bkg_new = -torch.sum(
                    bkg_wt
                    * torch.log(
                        torch.sum(
                            torch.abs(
                                torch.sum(
                                    bkg.view(total_gls, Nbkg, n_polar)
                                    * complex_vector.unsqueeze(1).unsqueeze(2),
                                    dim=0,
                                )
                            )
                            ** 2,
                            dim=1,
                        )
                        / phsp_factor_new
                    )
                )
            else:
                nll_bkg_new = -torch.sum(
                    torch.log(
                        torch.sum(
                            torch.abs(
                                torch.sum(
                                    bkg.view(total_gls, Nbkg, n_polar)
                                    * complex_vector.unsqueeze(1).unsqueeze(2),
                                    dim=0,
                                )
                            )
                            ** 2,
                            dim=1,
                        )
                        / phsp_factor_new
                    )
                )

            return nll_new - nll_bkg_new

        hessian = torch.autograd.functional.hessian(real_loss_fn, real_vector)

        bf_errors = torch.sqrt(
            torch.diag(jacobian @ torch.linalg.pinv(hessian) @ jacobian.t())
        )

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

        # 计算Hessian矩阵
        # print(f"计算Hessian矩阵...")
        hessian_start = time.time()
        hessian = self.compute_hessian(final_params[1:])
        hessian_time = time.time() - hessian_start
        # print(f"Hessian矩阵计算耗时: {hessian_time:.2f} 秒")

        # 检查Hessian矩阵正定性
        is_pos_def, min_eig, max_eig, cond_num = self.check_positive_definite(hessian)

        # 计算协方差矩阵和相关系数矩阵
        covariance = None
        correlation = None
        real_errors = None
        imag_errors = None

        if is_pos_def:
            # print(f"Hessian矩阵正定，计算协方差矩阵...")
            covariance = self.compute_covariance_matrix(hessian)
            if covariance is not None:
                correlation = self.compute_correlation_matrix(covariance)
                real_errors, imag_errors = self.compute_parameter_errors(
                    covariance, final_params
                )

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
            "covariance": covariance,
            "correlation": correlation,
            "real_errors": real_errors,
            "imag_errors": imag_errors,
        }

        # 更新最佳结果
        if final_nll < self.best_nll:
            self.best_nll = final_nll
            self.best_params = final_params.clone()
            self.best_result = result  # 保存完整的最佳结果

        return result

    def save_matrices(self, result, cov_file_path, corr_file_path, is_first_run=False):
        """将协方差矩阵和相关系数矩阵追加保存到文件"""
        run_id = result["run_id"]

        # 保存协方差矩阵
        if result["covariance"] is not None:
            self.save_matrix_to_file(
                result["covariance"],
                cov_file_path,
                f"协方差矩阵 - 第 {run_id} 次运行",
                run_id,
                is_first_run,
                "covariance_matrix.txt",
            )

        # 保存相关系数矩阵
        if result["correlation"] is not None:
            self.save_matrix_to_file(
                result["correlation"],
                corr_file_path,
                f"相关系数矩阵 - 第 {run_id} 次运行",
                run_id,
                is_first_run,
                "correlation_matrix.txt",
            )

    def save_matrix_to_file(
        self, matrix, filename, matrix_name, run_id, is_first_run, base_filename
    ):
        """保存矩阵到文件（追加模式）"""
        try:
            mode = "w" if is_first_run else "a"

            with open(filename, mode) as f:
                if is_first_run:
                    # 写入文件头
                    # f.write("# PWA 矩阵数据\n")
                    f.write("#" * 80 + "\n")
                    f.write(f"# 文件: {base_filename}\n")
                    f.write(f"# 总运行次数: 待定\n")
                    f.write(f"# 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    # f.write(f"# 可变参数数量: {self.n_variable}\n")
                    f.write(f"# 实数参数数量: {self.n_real_variable}\n")
                    f.write("#" * 80 + "\n\n")

                # 写入当前运行的矩阵
                f.write(f"{matrix_name}\n")
                f.write("=" * 60 + "\n")
                f.write(f"运行ID: run_{run_id}\n")
                f.write(f"矩阵维度: {matrix.shape[0]} × {matrix.shape[1]}\n")

                # 写入Hessian矩阵信息（如果可用）
                # if hasattr(self, "best_result") and self.best_result is not None:
                #     if run_id == self.best_result.get("run_id", -1):
                f.write(
                    f"Hessian矩阵正定性: {'是' if self.best_result.get('is_positive_definite', False) else '否'}\n"
                )

                f.write("=" * 60 + "\n\n")

                # 写入矩阵数据
                matrix_np = matrix.cpu().numpy()
                for i in range(matrix_np.shape[0]):
                    for j in range(matrix_np.shape[1]):
                        f.write(f"{matrix_np[i, j]:12.6e} ")
                    f.write("\n")

                f.write("\n" + "#" * 80 + "\n\n")

            if is_first_run:
                print(f"✅ 创建矩阵文件: {filename}")
            # else:
            #     print(f"✅ 追加第 {run_id} 次运行矩阵数据到: {filename}")

            return True
        except Exception as e:
            print(f"❌ 保存矩阵到文件 {filename} 失败: {e}")
            return False

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
            print(f"❌ 保存带误差的参数失败: {e}")
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
        cov_filename = os.path.join(output_dir, "covariance_matrix.txt")
        corr_filename = os.path.join(output_dir, "correlation_matrix.txt")

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
            result = self.optimize_single_run(initial_params, run_id=i, **kwargs)
            results.append(result)
            self.all_results.append(result)

            print(f"第 {i} 次优化完成!")
            print(f"最终NLL: {result['final_nll']:.6f}")
            print(f"正定性: {result['is_positive_definite']}")
            print(f"优化耗时: {result['time']:.2f} 秒")
            print(f"海森耗时: {result['hessian_time']:.2f} 秒")
            print(f"迭代次数: {result['iterations']}")

            # 保存带误差的参数
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

            # 保存协方差和相关系数矩阵到统一文件
            if result["is_positive_definite"]:
                is_first_run = i == 0
                self.save_matrices(result, cov_filename, corr_filename, is_first_run)

        return results

    def print_optimized_parameters_with_errors(
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
            # f.write(f"固定参数: {self.amplitude_names[0]} = 1+0j\n")
            f.write(f"可变参数: {self.n_variable}\n")
            f.write(f"最佳NLL: {self.best_nll:.6f}\n")
            f.write(f"所有参数结果: parameters.txt\n")
            f.write(f"所有NLL结果: nll_history.txt\n")
            f.write(f"协方差矩阵: covariance_matrix.txt\n")
            f.write(f"相关系数矩阵: correlation_matrix.txt\n\n")

            f.write("=" * 100 + "\n")
            f.write("运行结果 (按NLL排序):\n")
            f.write("=" * 100 + "\n")
            f.write(
                f"{'排名':<4} {'运行ID':<6} {'NLL':<12} {'迭代次数':<8} "
                f"{'耗时(秒)':<10} {'Hessian耗时':<12} {'正定':<6} \n"
            )
            f.write("-" * 100 + "\n")

            for rank, res in enumerate(sorted_results):
                f.write(
                    f"{rank+1:<4} {res['run_id']:<6} {res['final_nll']:<12.6f} "
                    f"{res['iterations']:<8} {res['time']:<10.2f} "
                    f"{res['hessian_time']:<12.2f} {str(res['is_positive_definite']):<6} \n"
                )

            # 写入最佳结果的分支比
            # 先判断是否正定
            if self.best_result.get("is_positive_definite", False):
                Ninit = 2.24e6  # 初始事件数
                f.write("=" * 100 + "\n")
                f.write("=" * 100 + "\n")
                f.write("最佳结果的分支比:\n")
                f.write("=" * 100 + "\n")
                bf, bf_err = self.compute_branching_fractions(self.best_params)
                for i in range(len(bf)):
                    f.write(f"{i:2d}: {bf[i]/Ninit:.6e} ± {bf_err[i]/Ninit:.6e}\n")
        print(f"✅ 优化结果摘要已保存到: {summary_file}")


# 创建优化器实例
optimizer = SimplePWAOptimizer(ana, conjugate_pairs, amplitude_names)

# 运行多次优化
results = optimizer.run_multiple_optimizations(
    num_runs=10,
    max_iter=1000,
    lr=0.9,
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
        f"正定 = {res['is_positive_definite']}, "
        # f"最小特征值 = {res['min_eigenvalue']:.2e}"
    )

print(f"\n{'='*80}")
print("最佳结果:")
print(f"{'='*80}")

best_res = sorted_results[0]
print(f"最佳NLL: {best_res['final_nll']:.6f} (来自第 {best_res['run_id']} 次运行)")
print(f"正定性: {best_res['is_positive_definite']}")

# 打印最佳参数（包含误差）
if best_res["is_positive_definite"] and best_res["real_errors"] is not None:
    optimizer.print_optimized_parameters_with_errors(
        best_res["final_params"],
        best_res["real_errors"],
        best_res["imag_errors"],
        best_res["run_id"],
    )
else:
    print(f"\n⚠️ Hessian矩阵不正定，无法提供参数误差估计")
    optimizer.print_optimized_parameters_with_errors(best_res["final_params"])

# 只保存最佳权重文件
best_weight_file = "results/weight_best.root"
optimizer.save_weight_file(best_res["final_params"], best_weight_file)

# 保存所有结果摘要
optimizer.save_all_results_summary()
