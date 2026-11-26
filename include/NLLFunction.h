// include/NLLFunction.h — torch autograd 包装（P4：从 Analysis.h 里拆出的第二个类）
//
// 之前 Analysis.h 里其实有**两个**类：`analysis`（编排）和 `NLLFunction`
// （autograd::Function 的 forward/backward 实现）。后者与前者没有耦合，
// 只是被 getNLL 调用。单独成头之后 Analysis.h 少 ~770 行。
#pragma once

class NLLFunction : public torch::autograd::Function<NLLFunction>
{
public:
    // 多 GPU 前向传播：输入统一 params = [real(v_0..n-1), imag(v_0..n-1), θ_0..P-1] float64
    static torch::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        torch::Tensor params_tensor,           // 统一参数 [2*nFreeVector + nFreeTheta] float64
        Parameters* params_mgr,                // 参数管理器（含约束 & 维度信息 & vspace flag）
        std::vector<ctComplex*>& d_all_amplitudes_list,
        AmpCalc* amp_calc,                     // 共振态管理器
        ctComplex* d_phsp_matrix_,
        cuDoubleComplex* d_phsp_matrix_double_,   // 非空 = phsp 流式模式（phsp_sum 走 double 矩阵）
        const std::vector<std::vector<int>>& events_list,
        const std::vector<std::vector<int>>& events_offsets_list,
        const std::vector<std::vector<int>>& amp_offsets_list,
        std::vector<double*>& d_data_weights_list,
        std::vector<double*>& d_phsp_weights_list,
        std::vector<double*>& d_bkg_weights_list,
        double bkg_integral_,
        int n_amplitudes_,
        int n_polar_)
    {
        TORCH_CHECK(params_mgr && params_mgr->initialized(), "Parameters not initialized");
        bool tprof = getenv("CTPWA_PROF") != nullptr;
        auto tF0 = std::chrono::high_resolution_clock::now();
        // phsp 梯度段的持久化缓冲（懒分配，避免每次 forward 的 cublasCreate/cudaMalloc）
        static std::vector<cublasHandle_t> s_phsp_handle;   // phsp CGEMV（θ 段 S 也用）
        static std::vector<ctComplex*> s_d_S_bufs;          // θ 段 S = A^T·v
        static std::vector<int> s_dS_alloc_sz;
        static std::vector<ctComplex*> s_d_P_part;          // free-θ: 每 GPU 局部 d_P_vec
        static std::vector<int> s_d_P_part_sz;
        static ctComplex* s_d_P_peer = nullptr;             // free-θ: 主 GPU 规约临时
        static int s_d_P_peer_sz = 0;

        // 0. 拆分 params → vector + theta
        torch::Tensor vector, theta;
        if (params_mgr->hasCouplingMatrix()) {
            const auto& cm = params_mgr->couplingMatrix();
            int ncf = cm.n_free;
            int na  = cm.n_amps;
            int nt  = params_mgr->nFreeTheta();

            // params_tensor: [Re_p, Im_p, θ] (2*ncf+nt) → extend → [Re_v, Im_v, θ] (2*na+nt)
            auto ext = torch::empty({2 * na + nt},
                torch::TensorOptions().dtype(torch::kFloat64).device(params_tensor.device()));
            params_mgr->extendCouplingParams(
                params_tensor.data_ptr<double>(), ext.data_ptr<double>(), ncf, nt);

            // Save original params for backward gradient transform
            ctx->saved_data["original_params"] = params_tensor;

            vector = torch::complex(
                ext.slice(0, 0, na).to(TORCH_FLOAT),
                ext.slice(0, na, 2 * na).to(TORCH_FLOAT));
            theta = (nt > 0) ? ext.slice(0, 2 * na, 2 * na + nt) : torch::Tensor();
        } else {
            std::tie(vector, theta) = params_mgr->splitParams(params_tensor);
        }

        int num_gpus = d_all_amplitudes_list.size();
        TORCH_CHECK(num_gpus > 0, "No GPUs provided");
        TORCH_CHECK(ctpwa_is_cuda_like(vector), "params 必须位于当前后端的设备上（CPU 构建用 CPU 张量）");

        // 1. 将 vector 扩展向量
        torch::Tensor extended_vector = params_mgr->extendVector(vector, vector.device());
        int extended_n_gls = extended_vector.numel();
        // torch 在 CPU 张量上返回 -1（没有设备索引）。这里统一归一化到 0：
        // 该值随后被当作设备号使用（ensure_device / cudaMemcpy / peer copy）。
        const int primary_dev = ctpwa_is_cuda_like(vector) ? vector.get_device() : 0;

        // 2. 将extended_vector分发到每个GPU（实际拷贝）
        std::vector<torch::Tensor> extended_vec_per_gpu;
        for (int i = 0; i < num_gpus; ++i) {
            extended_vec_per_gpu.push_back(
                extended_vector.to(ctpwa_dev(i)));
        }

#if defined(CTPWA_BACKEND_CPU)
        if (getenv("CTPWA_CPU_DEBUG")) {
            auto hv = extended_vector.to(torch::kCPU).contiguous();
            const auto* zp = hv.data_ptr<c10::complex<double>>();
            const int n = std::min<int>(6, (int)hv.numel());
            std::printf("[CPU-DBG] extended_vector n=%d:", (int)hv.numel());
            for (int i = 0; i < n; ++i)
                std::printf(" (%.6g,%.6g)", zp[i].real(), zp[i].imag());
            std::printf("\n");
        }
#endif
        auto tF1 = std::chrono::high_resolution_clock::now();  // extend+分发完成
        auto t2_reamp = tF1, t6_nll = tF1, t7_rg = tF1;
        auto tF2 = tF1, tF3 = tF1, tF4 = tF1;

        // === 新增：Step 2.5: 更新振幅 & 预计算有效耦合 ===
        int n_free_res = 0;
        if (amp_calc) n_free_res = amp_calc->nFreeResParams();
        if (amp_calc && n_free_res > 0 && theta.numel() > 0) {
            // 重新计算 d_all_amplitudes（用新的共振态参数）
            auto t0_reamp = std::chrono::high_resolution_clock::now();
            amp_calc->reComputeAmps(d_all_amplitudes_list,
                reinterpret_cast<const double*>(theta.data_ptr()),
                n_amplitudes_, events_offsets_list, amp_offsets_list, n_polar_,
                primary_dev);
            cudaDeviceSynchronize();
            auto t1_reamp = std::chrono::high_resolution_clock::now();

            // 预计算有效耦合 T——已停用: d_T 全库无消费者（死代码; 每 forward 全事件
            // kernel + nEv×nPol×16B/卡 表纯浪费）。若未来 Hessian 需 T 中间量，
            // 请先实现消费者再恢复此调用。
            // for (int gpu = 0; gpu < num_gpus; ++gpu) { ... computeEffectiveCoupling ... }
            t2_reamp = std::chrono::high_resolution_clock::now();
            if (getenv("CTPWA_PROF")) printf("[PROF] reComputeAmps: %.2f ms, +T: %.2f ms\n",
                std::chrono::duration<double, std::milli>(t1_reamp - t0_reamp).count(),
                std::chrono::duration<double, std::milli>(t2_reamp - t1_reamp).count());

        }

        tF2 = std::chrono::high_resolution_clock::now();  // reAmp+T 完成
        // (free-θ 的 B̄ 重建已移除: d_P_vec 由下方 phsp_factor 融合扫描直接给出,
        //  免 7.2GB 加权副本; fixed-θ 仍用构造期预建 B̄ 的快速 matvec)

        // 3. 全局量: d_P_vec和phsp_factor（小矩阵M<100，用自定义核替代cuBLAS更快）
        //   fixed-θ（常驻或流式）: B̄ 构造期就绪 → computeQuadraticForm 快速 matvec
        //   free-θ:  B̄ 不再每 forward 重建（删 7.2GB 加权副本）;
        //            phsp_factor 与 d_P_vec 由一次融合扫描直接给出
        //            d_P_vec = Σ_ev,p (w/W)·A·conj(S) ≡ B̄·v̄
        bool free_theta = (amp_calc && n_free_res > 0 && theta.numel() > 0);
        ctpwa::ensure_device(primary_dev);
        ctComplex* d_P_vec;
        cudaMalloc(&d_P_vec, n_amplitudes_ * sizeof(ctComplex));
        double phsp_sum = 0.0;
        // phsp 归一化分母: Σ w（无权重 = 事件数）。与 d_phsp_matrix_ 的
        // 1/W_total 以及梯度 d_P_vec 的 w/W_total 必须用同一个 W_total。
        double W_total = 0.0;
        for (int gpu = 0; gpu < num_gpus; ++gpu) {
            int nP = events_list[gpu][0];
            if (nP == 0) continue;
            if (gpu < (int)d_phsp_weights_list.size() && d_phsp_weights_list[gpu] != nullptr) {
                ctpwa::ensure_device(gpu);
                thrust::device_ptr<double> dp(d_phsp_weights_list[gpu]);
                W_total += thrust::reduce(dp, dp + nP);
            } else {
                W_total += (double)nP;
            }
        }
        if (W_total <= 0.0) W_total = 1.0;
        // 流式 double 矩阵与 float32 矩阵同归一化：B̄ = Σ(w/W)|A|²，
        // Re(v^H B̄ v) 已经是 Σ(w/W)|S|²，不能再除 Σw。
        bool phsp_stream_normalized = false;
        if (d_phsp_matrix_double_ != nullptr) {
            // phsp 流式模式: 原始 phsp 振幅不驻留，phsp_sum = Re(v^H M_double v)（主 GPU）
            TORCH_CHECK(n_free_res == 0,
                "phsp 流式模式（free_phsp_amplitudes）要求无自由共振态参数");
            ctpwa::ensure_device(primary_dev);
            const ctComplex* d_v_primary = reinterpret_cast<const ctComplex*>(
                extended_vector.data_ptr());
            phsp_sum = computeDoublePhspSum(d_phsp_matrix_double_, d_v_primary,
                                            n_amplitudes_);
            phsp_stream_normalized = true;
            // fixed-θ（n_free_res==0 必成立）: 构造期预建 B̄ → matvec
            ctPhspReal* d_phsp_r, * d_phsp_i;
            cudaMalloc(&d_phsp_r, sizeof(ctPhspReal));
            cudaMalloc(&d_phsp_i, sizeof(ctPhspReal));
            {
                torch::Tensor extended_vector_conj = extended_vector.conj();
                const ctComplex* d_vec_conj = reinterpret_cast<const ctComplex*>(extended_vector_conj.data_ptr());
                computeQuadraticForm(d_phsp_matrix_, d_vec_conj, d_P_vec,
                    d_phsp_r, d_phsp_i, n_amplitudes_);
            }
            cudaFree(d_phsp_r);
            cudaFree(d_phsp_i);
        } else if (free_theta) {
            // free-θ 常驻: phsp_factor 与 d_P_vec 一次融合扫描（免 B̄ 重建）
            //（phsp_factor 语义与 computePhspMeanSum 一致: Σ w|S|²，下面除 W_total）
            ctpwa::ensure_device(primary_dev);
            if ((int)s_d_P_part.size() < num_gpus) {
                s_d_P_part.resize(num_gpus, nullptr);
                s_d_P_part_sz.assign(num_gpus, 0);
            }
            if (s_d_P_peer_sz < n_amplitudes_) {
                if (s_d_P_peer) cudaFree(s_d_P_peer);
                cudaMalloc(&s_d_P_peer, n_amplitudes_ * sizeof(ctComplex));
                s_d_P_peer_sz = n_amplitudes_;
            }
            for (int gpu = 0; gpu < num_gpus; ++gpu) {
                int nP = events_list[gpu][0];
                if (nP == 0) continue;
                ctpwa::ensure_device(gpu);
                if ((int)s_phsp_handle.size() < num_gpus) s_phsp_handle.resize(num_gpus, nullptr);
                if (!s_phsp_handle[gpu]) cublasCreate(&s_phsp_handle[gpu]);   // θ 段 S CGEMV 共用
                if (s_d_P_part_sz[gpu] < n_amplitudes_) {
                    if (s_d_P_part[gpu]) cudaFree(s_d_P_part[gpu]);
                    cudaMalloc(&s_d_P_part[gpu], n_amplitudes_ * sizeof(ctComplex));
                    s_d_P_part_sz[gpu] = n_amplitudes_;
                }
                const ctComplex* d_v_gpu = reinterpret_cast<const ctComplex*>(
                    extended_vec_per_gpu[gpu].data_ptr());
                const double* d_w = (gpu < (int)d_phsp_weights_list.size())
                    ? d_phsp_weights_list[gpu] : nullptr;
                bool fA = amp_calc && amp_calc->floatOutput();
                if (fA) {
                    const float2* d_ampF = reinterpret_cast<const float2*>(d_all_amplitudes_list[gpu]);
                    phsp_sum += computePhspMeanSumAndGradPF(d_ampF, d_v_gpu,
                        d_w, nP, n_polar_, n_amplitudes_, 1.0 / W_total, s_d_P_part[gpu]);
                } else {
                    phsp_sum += computePhspMeanSumAndGradP(d_all_amplitudes_list[gpu], d_v_gpu,
                        d_w, nP, n_polar_, n_amplitudes_, 1.0 / W_total, s_d_P_part[gpu]);
                }
            }
            // 跨 GPU 规约局部 d_P_vec → 主 GPU d_P_vec
            ctpwa::ensure_device(primary_dev);
            cudaMemset(d_P_vec, 0, n_amplitudes_ * sizeof(ctComplex));
            {
                ctComplex one = ctMake(1.0f, 0.0f);
                for (int gpu = 0; gpu < num_gpus; ++gpu) {
                    int nP = events_list[gpu][0];
                    if (nP == 0) continue;
                    if (gpu == primary_dev) {
                        axpyComplex(d_P_vec, s_d_P_part[gpu], one, n_amplitudes_);
                    } else {
                        cudaMemcpyPeer(s_d_P_peer, primary_dev, s_d_P_part[gpu], gpu,
                            n_amplitudes_ * sizeof(ctComplex));
                        ctpwa::ensure_device(primary_dev);
                        axpyComplex(d_P_vec, s_d_P_peer, one, n_amplitudes_);
                    }
                }
            }
        } else {
            // fixed-θ 常驻: B̄ 构造期预建（forward 不重建）→ matvec（快速路径）
            // W_total 的逐 GPU 权重归约把 current device 留在最后一张卡。
            // d_phsp_matrix_ 与 d_P_vec 都在主卡，必须先切回主卡再做二次型；
            // 否则副卡 kernel 会直接读主卡指针（compute-sanitizer 已定位）。
            ctpwa::ensure_device(primary_dev);
            ctPhspReal* d_phsp_r, * d_phsp_i;
            cudaMalloc(&d_phsp_r, sizeof(ctPhspReal));
            cudaMalloc(&d_phsp_i, sizeof(ctPhspReal));
            {
                torch::Tensor extended_vector_conj = extended_vector.conj();
                const ctComplex* d_vec_conj = reinterpret_cast<const ctComplex*>(extended_vector_conj.data_ptr());
                computeQuadraticForm(d_phsp_matrix_, d_vec_conj, d_P_vec,
                    d_phsp_r, d_phsp_i, n_amplitudes_);
            }
            cudaFree(d_phsp_r);
            cudaFree(d_phsp_i);
            // phsp_factor = Σ w f / Σ w：Σ w f 用 double 累加（float32 phsp 矩阵
            // 求和在 50 万事件上有 ~1e-4 相对系统偏差）
            for (int gpu = 0; gpu < num_gpus; ++gpu) {
                int nP = events_list[gpu][0];
                if (nP == 0) continue;
                ctpwa::ensure_device(gpu);
                const ctComplex* d_v_gpu = reinterpret_cast<const ctComplex*>(
                    extended_vec_per_gpu[gpu].data_ptr());
                const double* d_w = (gpu < (int)d_phsp_weights_list.size())
                    ? d_phsp_weights_list[gpu] : nullptr;
                bool fA = amp_calc && amp_calc->floatOutput();
                if (fA) {
                    const float2* d_ampF = reinterpret_cast<const float2*>(d_all_amplitudes_list[gpu]);
                    phsp_sum += computePhspMeanSumF(d_ampF, d_v_gpu,
                                                   nP, n_polar_, n_amplitudes_, d_w);
                } else {
                    phsp_sum += computePhspMeanSum(d_all_amplitudes_list[gpu], d_v_gpu,
                                                   nP, n_polar_, n_amplitudes_, d_w);
                }
            }
        }
        // 加权 phsp 均值: Σ w f / Σ w。流式 double 矩阵已在构造期乘过
        // sqrt(w/W_total)，所以 phsp_sum 已经是归一化值（denom=1）。
        double phsp_denom = phsp_stream_normalized ? 1.0 : W_total;
        double phsp_factor = (phsp_denom > 0.0) ? phsp_sum / phsp_denom : 0.0;
        ctpwa::ensure_device(primary_dev);
        tF3 = std::chrono::high_resolution_clock::now();  // phsp_factor/d_P_vec（free-θ 融合扫描）完成
        tF4 = tF3;  // （原 B̄ CGEMM 段已删；D2H 计入 tF3 段）

        // //输出 d_P_vec 检查
        // std::vector<ctComplex> h_P_vec(n_amplitudes_);
        // cudaMemcpy(h_P_vec.data(), d_P_vec, n_amplitudes_ * sizeof(ctComplex),
        //     cudaMemcpyDeviceToHost);
        // std::cout << "P_vec: ";
        // for (int i = 0; i < n_amplitudes_; ++i) {
        //     std::cout << "(" << h_P_vec[i].x << ", " << h_P_vec[i].y << ") ";
        // }
        // std::cout << std::endl;

        // 4. 多GPU计算data+bkg似然值和梯度（纯data/bkg，不含phsp项）
        double total_data_nll = 0.0;
        double total_bkg_nll = 0.0;
        int totalDataEvents = 0;
        int totalBkgEvents = 0;
        double totalDataWeight = 0.0;   // Σ data 权重（无权重时 = 事件数）
        double bkg_integral = bkg_integral_;

        // 预分配持久化buffer（首次分配，之后复用）
        static ctComplex* s_d_grad_global = nullptr;
        static ctComplex* s_d_grad_buf = nullptr;
        static std::vector<ctComplex*> s_d_grad_per_gpu;
        static std::vector<ctComplex*> s_d_w_bufs;       // per-GPU data w buffer
        static std::vector<ctComplex*> s_d_w_bkg_bufs;   // per-GPU bkg  w buffer
        static std::vector<int> s_w_buf_sizes;            // data
        static std::vector<int> s_w_bkg_buf_sizes;        // bkg
        static int s_alloc_n = 0;

        if (s_alloc_n < extended_n_gls || s_d_grad_per_gpu.size() < (size_t)num_gpus) {
            // 释放后必须置空 + sizes 归零，否则 resize 不重置已有元素，
            // 悬空指针会被复用（cudaMemcpy invalid argument 的根因）
            if (s_d_grad_global) { cudaFree(s_d_grad_global); s_d_grad_global = nullptr; }
            if (s_d_grad_buf) { cudaFree(s_d_grad_buf); s_d_grad_buf = nullptr; }
            for (auto& p : s_d_grad_per_gpu) if (p) { cudaFree(p); p = nullptr; }
            for (auto& p : s_d_w_bufs) if (p) { cudaFree(p); p = nullptr; }
            for (auto& p : s_d_w_bkg_bufs) if (p) { cudaFree(p); p = nullptr; }

            ctpwa::ensure_device(primary_dev);
            cudaMalloc(&s_d_grad_global, extended_n_gls * sizeof(ctComplex));
            cudaMalloc(&s_d_grad_buf, extended_n_gls * sizeof(ctComplex));
            s_d_grad_per_gpu.resize(num_gpus, nullptr);
            s_d_w_bufs.resize(num_gpus, nullptr);
            s_d_w_bkg_bufs.resize(num_gpus, nullptr);
            s_w_buf_sizes.assign(num_gpus, 0);
            s_w_bkg_buf_sizes.assign(num_gpus, 0);
            for (int g = 0; g < num_gpus; ++g) {
                ctpwa::ensure_device(g);
                cudaMalloc(&s_d_grad_per_gpu[g], extended_n_gls * sizeof(ctComplex));
            }
            s_alloc_n = extended_n_gls;
        }

        ctComplex* d_grad_global = s_d_grad_global;
        ctComplex* d_grad_buf = s_d_grad_buf;
        ctpwa::ensure_device(primary_dev);
        cudaMemset(d_grad_global, 0, extended_n_gls * sizeof(ctComplex));

        for (int gpu = 0; gpu < num_gpus; ++gpu) {
            ctpwa::ensure_device(gpu);
            const ctComplex* d_vec_gpu = reinterpret_cast<const ctComplex*>(
                extended_vec_per_gpu[gpu].data_ptr());
            ctComplex* d_grad = s_d_grad_per_gpu[gpu];

            // --- data ---
            int nData_gpu = events_list[gpu][1];
            if (nData_gpu > 0) {
                ctComplex* d_amp = d_all_amplitudes_list[gpu] + amp_offsets_list[gpu][1];
                const double* d_w_data = (gpu < (int)d_data_weights_list.size())
                    ? d_data_weights_list[gpu] : nullptr;

                // 分配/检查 w buffer（仅当有 theta 即做共振态拟合时）
                ctComplex* d_w_out = nullptr;
                if (amp_calc && n_free_res > 0 && theta.numel() > 0) {
                    int nTotal = nData_gpu * n_polar_;
                    if (s_w_buf_sizes[gpu] < nTotal) {
                        if (s_d_w_bufs[gpu]) cudaFree(s_d_w_bufs[gpu]);
                        cudaMalloc(&s_d_w_bufs[gpu], nTotal * sizeof(ctComplex));
                        s_w_buf_sizes[gpu] = nTotal;
                    }
                    d_w_out = s_d_w_bufs[gpu];
                }

                auto t0_nll = std::chrono::high_resolution_clock::now();
                bool fA = amp_calc && amp_calc->floatOutput();
                double nll;
                if (fA) {
                    // amp_offsets 为元素数偏移：float2 模式下从基址 reinterpret 后偏移
                    const float2* d_ampF = reinterpret_cast<const float2*>(d_all_amplitudes_list[gpu])
                        + amp_offsets_list[gpu][1];
                    // w 缓冲保持 double（共振梯度按 ctComplex 读）：F 输出 float2 临时 → 上转
                    float2* d_w_f = nullptr;
                    if (d_w_out) {
                        cudaMalloc(&d_w_f, (size_t)nData_gpu * n_polar_ * sizeof(float2));
                    }
                    nll = computeFactorNLLF(d_ampF, d_vec_gpu,
                        d_grad, nData_gpu, n_polar_, n_amplitudes_, d_w_data, d_w_f);
                    if (d_w_f) {
                        int nW = nData_gpu * n_polar_;
                        if (ctIsFloatComplex) {
                            // 原生 float .so：float2 与 ctComplex(cuComplex) 同布局，
                            // 直接 D2D 拷贝；再"上转 double"会按 16B/元素越界写。
                            cudaMemcpy(d_w_out, d_w_f,
                                (size_t)nW * sizeof(ctComplex), cudaMemcpyDeviceToDevice);
                        } else {
                            CTPWA_LAUNCH(castF2ToDouble2Kernel, (nW + 255) / 256, 256, 0,
                                d_w_f, reinterpret_cast<cuDoubleComplex*>(d_w_out), nW);
                        }
                        cudaFree(d_w_f);
                    }
                } else {
                    nll = computeFactorNLL(d_amp, d_vec_gpu,
                        d_grad, nData_gpu, n_polar_, n_amplitudes_, d_w_data, d_w_out);
                }

#if defined(CTPWA_BACKEND_CPU)
                if (getenv("CTPWA_CPU_DEBUG"))
                    std::printf("[CPU-DBG] gpu %d data nll=%.10f (nData=%d nPol=%d nAmp=%d)\n",
                                gpu, nll, nData_gpu, n_polar_, n_amplitudes_);
#endif
                total_data_nll += nll;
                totalDataEvents += nData_gpu;
                // data 总权重（phsp 归一化 N_eff 用；无权重时 = 事件数）
                if (d_w_data != nullptr) {
                    ctpwa::ensure_device(gpu);
                    thrust::device_ptr<const double> dp(d_w_data);
                    totalDataWeight += thrust::reduce(dp, dp + nData_gpu);
                }
                else {
                    totalDataWeight += nData_gpu;
                }
                auto t1_nll = std::chrono::high_resolution_clock::now();
                if (getenv("CTPWA_PROF")) printf("[PROF] computeFactorNLL(data): %.2f ms\n",
                    std::chrono::duration<double, std::milli>(t1_nll - t0_nll).count());
                // P2P累加到global (正号)
                if (gpu == primary_dev) {
                    axpyComplex(d_grad_global, d_grad, ctMake(1.0f, 0.0f), extended_n_gls);
                }
                else {
                    cudaMemcpyPeer(d_grad_buf, primary_dev, d_grad, gpu, extended_n_gls * sizeof(ctComplex));
                    ctpwa::ensure_device(primary_dev);
                    axpyComplex(d_grad_global, d_grad_buf, ctMake(1.0f, 0.0f), extended_n_gls);
                }
            }

            // --- bkg ---
            int nBkg_gpu = (events_list[gpu].size() > 2) ? events_list[gpu][2] : 0;
            if (nBkg_gpu > 0) {
                ctpwa::ensure_device(gpu);
                ctComplex* d_amp = d_all_amplitudes_list[gpu] + amp_offsets_list[gpu][2];
                const double* d_w = d_bkg_weights_list[gpu];

                // 分配/检查 bkg  w buffer
                ctComplex* d_w_bkg_out = nullptr;
                if (amp_calc && n_free_res > 0 && theta.numel() > 0) {
                    int nTotal = nBkg_gpu * n_polar_;
                    if (s_w_bkg_buf_sizes[gpu] < nTotal) {
                        if (s_d_w_bkg_bufs[gpu]) cudaFree(s_d_w_bkg_bufs[gpu]);
                        cudaMalloc(&s_d_w_bkg_bufs[gpu], nTotal * sizeof(ctComplex));
                        s_w_bkg_buf_sizes[gpu] = nTotal;
                    }
                    d_w_bkg_out = s_d_w_bkg_bufs[gpu];
                }

                auto t0_nllb = std::chrono::high_resolution_clock::now();
                bool fA = amp_calc && amp_calc->floatOutput();
                double nll;
                if (fA) {
                    const float2* d_ampF = reinterpret_cast<const float2*>(d_all_amplitudes_list[gpu])
                        + amp_offsets_list[gpu][2];
                    float2* d_w_f = nullptr;
                    if (d_w_bkg_out) {
                        cudaMalloc(&d_w_f, (size_t)nBkg_gpu * n_polar_ * sizeof(float2));
                    }
                    nll = computeFactorNLLF(d_ampF, d_vec_gpu,
                        d_grad, nBkg_gpu, n_polar_, n_amplitudes_, d_w, d_w_f);
                    if (d_w_f) {
                        int nW = nBkg_gpu * n_polar_;
                        if (ctIsFloatComplex) {
                            cudaMemcpy(d_w_bkg_out, d_w_f,
                                (size_t)nW * sizeof(ctComplex), cudaMemcpyDeviceToDevice);
                        } else {
                            CTPWA_LAUNCH(castF2ToDouble2Kernel, (nW + 255) / 256, 256, 0,
                                d_w_f, reinterpret_cast<cuDoubleComplex*>(d_w_bkg_out), nW);
                        }
                        cudaFree(d_w_f);
                    }
                } else {
                    nll = computeFactorNLL(d_amp, d_vec_gpu,
                        d_grad, nBkg_gpu, n_polar_, n_amplitudes_, d_w, d_w_bkg_out);
                }

                total_bkg_nll += nll;
                totalBkgEvents += nBkg_gpu;
                auto t1_nllb = std::chrono::high_resolution_clock::now();
                if (getenv("CTPWA_PROF")) printf("[PROF] computeFactorNLL(bkg): %.2f ms\n",
                    std::chrono::duration<double, std::milli>(t1_nllb - t0_nllb).count());
                if (gpu == primary_dev) {
                    axpyComplex(d_grad_global, d_grad, ctMake(-1.0f, 0.0f), extended_n_gls);
                }
                else {
                    cudaMemcpyPeer(d_grad_buf, primary_dev, d_grad, gpu, extended_n_gls * sizeof(ctComplex));
                    ctpwa::ensure_device(primary_dev);
                    axpyComplex(d_grad_global, d_grad_buf, ctMake(-1.0f, 0.0f), extended_n_gls);
                }
            }
        }

        t6_nll = std::chrono::high_resolution_clock::now();  // computeFactorNLL data+bkg 完成

        // === 新增：计算共振态参数梯度 ===
        torch::Tensor grad_theta;
        if (amp_calc && n_free_res > 0 && theta.numel() > 0) {
            ctpwa::ensure_device(primary_dev);
            // 持久化懒分配（n_free_res 固定，避免每次 forward 的 malloc/free）
            static double* s_d_grad_res = nullptr;
            static int s_grad_res_sz = 0;
            if (s_grad_res_sz < n_free_res) {
                if (s_d_grad_res) cudaFree(s_d_grad_res);
                ctpwa::ensure_device(primary_dev);
                cudaMalloc(&s_d_grad_res, n_free_res * sizeof(double));
                s_grad_res_sz = n_free_res;
            }
            double* d_grad_res = s_d_grad_res;
            ctpwa::ensure_device(primary_dev);
            cudaMemset(d_grad_res, 0, n_free_res * sizeof(double));

            // data 贡献 (sign=+1, d_T/d_momenta 需加 phsp 偏移)
            if (totalDataEvents > 0) {
                std::vector<int> n_data_events(num_gpus);
                std::vector<int> phsp_offsets(num_gpus);
                std::vector<ctComplex*> d_v_ptrs(num_gpus);
                for (int g = 0; g < num_gpus; ++g) {
                    n_data_events[g] = events_list[g][1];
                    phsp_offsets[g] = events_list[g][0];
                    d_v_ptrs[g] = const_cast<ctComplex*>(
                        reinterpret_cast<const ctComplex*>(extended_vec_per_gpu[g].data_ptr()));
                }
                auto t0_rg = std::chrono::high_resolution_clock::now();
                amp_calc->computeResonanceGradient(primary_dev, s_d_w_bufs, n_data_events, d_grad_res,
                    +1.0, phsp_offsets, d_v_ptrs);
                auto t1_rg = std::chrono::high_resolution_clock::now();
                if (getenv("CTPWA_PROF")) printf("[PROF] resonanceGradient(data): %.2f ms\n",
                    std::chrono::duration<double, std::milli>(t1_rg - t0_rg).count());
            }

            // bkg 贡献 (sign=-1, loss = data_nll - bkg_nll)
            if (totalBkgEvents > 0) {
                std::vector<int> n_bkg_events(num_gpus);
                std::vector<int> bkg_offsets(num_gpus);
                std::vector<ctComplex*> d_v_ptrs(num_gpus);
                for (int g = 0; g < num_gpus; ++g) {
                    n_bkg_events[g] = events_list[g][2];
                    bkg_offsets[g] = events_list[g][0] + events_list[g][1]; // phsp + data 偏移
                    d_v_ptrs[g] = const_cast<ctComplex*>(
                        reinterpret_cast<const ctComplex*>(extended_vec_per_gpu[g].data_ptr()));
                }
                auto t0_rgb = std::chrono::high_resolution_clock::now();
                amp_calc->computeResonanceGradient(primary_dev, s_d_w_bkg_bufs, n_bkg_events, d_grad_res,
                    -1.0, bkg_offsets, d_v_ptrs);
                auto t1_rgb = std::chrono::high_resolution_clock::now();
                if (getenv("CTPWA_PROF")) printf("[PROF] resonanceGradient(bkg): %.2f ms\n",
                    std::chrono::duration<double, std::milli>(t1_rgb - t0_rgb).count());
            }

            // phsp 贡献: ∂((N_data-W_bkg)*log(phsp))/∂θ
            {
                int total_phsp = 0;
                for (int g = 0; g < num_gpus; ++g) total_phsp += events_list[g][0];
                double effective_data = totalDataWeight - bkg_integral_;
                if (total_phsp > 0 && phsp_factor > 1e-30) {
                    // S 下面会乘 w/W_total（scalePhspSByWeightKernel），
                    // 因此 sign 不再除 total_phsp（W_total 已在 S 里）。
                    double phsp_sign = -effective_data / phsp_factor;

                    std::vector<ctComplex*> d_S_bufs(num_gpus, nullptr);
                    std::vector<int> n_phsp_evts(num_gpus);
                    if ((int)s_d_S_bufs.size() < num_gpus) {
                        s_d_S_bufs.resize(num_gpus, nullptr);
                        s_dS_alloc_sz.assign(num_gpus, 0);
                    }
                    for (int g = 0; g < num_gpus; ++g) {
                        ctpwa::ensure_device(g);
                        int nP = events_list[g][0];
                        n_phsp_evts[g] = nP;
                        if (nP == 0) continue;
                        int nTot = nP * n_polar_;
                        if (s_dS_alloc_sz[g] < nTot) {
                            if (s_d_S_bufs[g]) cudaFree(s_d_S_bufs[g]);
                            cudaMalloc(&s_d_S_bufs[g], nTot * sizeof(ctComplex));
                            s_dS_alloc_sz[g] = nTot;
                        }
                        d_S_bufs[g] = s_d_S_bufs[g];
                        const ctComplex* d_vg = reinterpret_cast<const ctComplex*>(
                            extended_vec_per_gpu[g].data_ptr());
                        bool fA = amp_calc && amp_calc->floatOutput();
                        if (fA) {
                            // float2 A 直读算 S（无 7GB double 临时；v double、S 输出 double）
                            const float2* d_ampF = reinterpret_cast<const float2*>(d_all_amplitudes_list[g]);
                            int gS = (nP + 255) / 256;
                            CTPWA_LAUNCH(computeSFromFloatAmpsKernel, gS, 256, 0, 
                                s_d_S_bufs[g], d_ampF, d_vg, nP, n_polar_, n_amplitudes_);
                        } else {
                            ctComplex* d_amp = d_all_amplitudes_list[g];
                            ctComplex a = ctMake(1.0f, 0.0f);
                            ctComplex b = ctMake(0.0f, 0.0f);
                            CUBLAS_CGEMV(s_phsp_handle[g], CUBLAS_OP_T, n_amplitudes_, nTot,
                                &a, d_amp, n_amplitudes_, d_vg, 1, &b, d_S_bufs[g], 1);
                        }
                        const double* d_w = (g < (int)d_phsp_weights_list.size())
                            ? d_phsp_weights_list[g] : nullptr;
                        int gS = (nP + 255) / 256;
                        CTPWA_LAUNCH(scalePhspSByWeightKernel, gS, 256, 0,
                            d_S_bufs[g], d_w, nP, n_polar_, 1.0 / W_total);
                    }

                    ctpwa::ensure_device(primary_dev);  // d_grad_res is on primary_dev
                    std::vector<ctComplex*> d_v_ptrs(num_gpus);
                    for (int g = 0; g < num_gpus; ++g)
                        d_v_ptrs[g] = const_cast<ctComplex*>(
                            reinterpret_cast<const ctComplex*>(extended_vec_per_gpu[g].data_ptr()));
                    auto t0_rgp = std::chrono::high_resolution_clock::now();
                    amp_calc->computeResonanceGradient(primary_dev, d_S_bufs, n_phsp_evts, d_grad_res,
                        phsp_sign, {}, d_v_ptrs);
                    auto t1_rgp = std::chrono::high_resolution_clock::now();
                    if (getenv("CTPWA_PROF")) printf("[PROF] resonanceGradient(phsp): %.2f ms\n",
                        std::chrono::duration<double, std::milli>(t1_rgp - t0_rgp).count());

                    ctpwa::ensure_device(primary_dev);
                }
            }
            t7_rg = std::chrono::high_resolution_clock::now();  // resonanceGradient 各段完成

            ctpwa::ensure_device(primary_dev);
            grad_theta = torch::empty({ n_free_res },
                torch::TensorOptions().dtype(torch::kFloat64).device(ctpwa_dev(primary_dev)));
            cudaMemcpy(grad_theta.data_ptr(), d_grad_res,
                n_free_res * sizeof(double), cudaMemcpyDeviceToDevice);
        }
        else {
            grad_theta = torch::empty({ 0 },
                torch::TensorOptions().dtype(torch::kFloat64).device(vector.device()));
        }

        // // 输出d_grad_global检查
        // std::vector<ctComplex> h_grad_global(extended_n_gls);
        // cudaMemcpy(h_grad_global.data(), d_grad_global, extended_n_gls * sizeof(ctComplex),
        //     cudaMemcpyDeviceToHost);
        // std::cout << "Global gradient: ";
        // for (int i = 0; i < extended_n_gls; ++i) {
        //     std::cout << "(" << h_grad_global[i].x << ", " << h_grad_global[i].y << ") ";
        // }
        // std::cout << std::endl;

        // // 输出d_vec_gpu检查
        // std::vector<ctComplex> h_vec_gpu(extended_n_gls);
        // cudaMemcpy(h_vec_gpu.data(), extended_vec_per_gpu[0].data_ptr(),
        //     extended_n_gls * sizeof(ctComplex), cudaMemcpyDeviceToHost);
        // std::cout << "Extended vector (GPU 0): ";
        // for (int i = 0; i < extended_n_gls; ++i) {
        //     std::cout << "(" << h_vec_gpu[i].x << ", " << h_vec_gpu[i].y << ") ";
        // }
        // std::cout << std::endl;

        // 5. 合并data和bkg的NLL与梯度（loss = data_nll - bkg_nll）
        if (phsp_factor <= 1e-30 || isnan(phsp_factor) || isinf(phsp_factor)) {
            // std::cerr << "WARNING: phsp_factor invalid (" << phsp_factor << "), clamping to 1e-30" << std::endl;
            phsp_factor = 1e-30;
        }
        if (getenv("CTPWA_PROF")) printf("PY_NLL data_nll=%.10f bkg_nll=%.10f phsp_factor=%.10f totalData=%d totalBkg=%d bkg_integral=%.6f\n",
            total_data_nll, total_bkg_nll, phsp_factor, totalDataEvents, totalBkgEvents, bkg_integral_);

        double loss = total_data_nll - total_bkg_nll + (totalDataWeight - bkg_integral_) * log(phsp_factor);
        // Constraints.gauss_constr: 罚项 Σ(x-μ)²/(2σ²)
        // （NaN 情况下由下方 loss_reset 逻辑一并处理）
        if (params_mgr->nGaussConstr() > 0 && n_free_res > 0 && theta.numel() > 0)
            loss += params_mgr->gaussPenalty(theta);
        bool loss_reset = false;
        if (isnan(loss) || isinf(loss)) {
            // std::cerr << "WARNING: loss is " << (isnan(loss) ? "NaN" : "Inf") << ", resetting to 1e30" << std::endl;
            loss = 1e30;
            cudaMemset(d_grad_global, 0, extended_n_gls * sizeof(ctComplex));
            // theta 梯度也要清零，否则 NaN 会传播到最终梯度
            if (n_free_res > 0)
                cudaMemset(grad_theta.data_ptr<double>(), 0, n_free_res * sizeof(double));
            loss_reset = true;
        }
        // loss 重置时跳过 phsp 梯度累加——phsp_factor 已 clamp 到 1e-30，
        // scale_phsp ~ 1e33 会把 d_P_vec 中的任何 NaN/Inf 重新注入梯度
        if (!loss_reset) {
            ctComplex scale_phsp = ctMake(
                static_cast<float>(totalDataWeight - bkg_integral_) / static_cast<float>(phsp_factor), 0.0f);
            axpyComplex(d_grad_global, d_P_vec, scale_phsp, extended_n_gls);
        }
        cudaFree(d_P_vec);

        // gauss_constr 罚项梯度 (x-μ)/σ² 累加到 grad_theta
        // （loss_reset 时 grad_theta 已清零，跳过以免注入 NaN）
        if (!loss_reset && params_mgr->nGaussConstr() > 0 && n_free_res > 0
            && grad_theta.numel() > 0) {
            params_mgr->gaussPenaltyGrad(theta, grad_theta);
        }

        // 6. 保存梯度和loss到ctx
        ctpwa::ensure_device(primary_dev);
        torch::Tensor global_extended_grad = torch::empty({ extended_n_gls },
            TORCH_COMPLEX).to(vector.device());
        cudaMemcpy(global_extended_grad.data_ptr(), d_grad_global,
            extended_n_gls * sizeof(ctComplex), cudaMemcpyDeviceToDevice);

        ctx->save_for_backward({ params_tensor });
        ctx->saved_data["global_extended_grad"] = global_extended_grad;
        ctx->saved_data["grad_theta"] = grad_theta;
        ctx->saved_data["params_ptr"] = reinterpret_cast<int64_t>(params_mgr);

        // std::cout << "Total data NLL: " << loss << std::endl;
        if (tprof) {
            auto tFend = std::chrono::high_resolution_clock::now();
            auto ms = [](auto a, auto b) {
                return std::chrono::duration<double, std::milli>(b - a).count(); };
            printf("[PROF] f.extend: %.2f | f.reAmp+T: %.2f | f.phspFactor+dPvec: %.2f | f.nll: %.2f | f.gradRes: %.2f | f.tail: %.2f\n",
                ms(tF0, tF1), ms(tF1, t2_reamp), ms(t2_reamp, tF3),
                ms(tF3, t6_nll), ms(t6_nll, t7_rg), ms(t7_rg, tFend));
        }
        // 恢复主设备再返回：forward 尾部各 per-GPU 段可能把 raw current device
        // 留在非主卡，torch 缓存又只随 torch 自身调用更新——保持 raw 与 torch
        // 主设备一致，避免后续 python/torch 的裸/张量操作落到错误物理卡。
        ctpwa::ensure_device(primary_dev);
        return torch::tensor(loss, torch::kDouble).to(vector.device());
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx,
        const torch::autograd::tensor_list& grad_outputs)
    {
        bool tprof = getenv("CTPWA_PROF") != nullptr;
        auto tB0 = std::chrono::high_resolution_clock::now();
        const auto saved = ctx->get_saved_variables();
        const auto& params_tensor = saved[0];
        const auto& global_extended_grad = ctx->saved_data["global_extended_grad"].toTensor();
        const auto& grad_theta = ctx->saved_data["grad_theta"].toTensor();
        auto* params_mgr = reinterpret_cast<Parameters*>(ctx->saved_data["params_ptr"].toInt());

        int nv = params_mgr->nFreeVector();
        int nt = params_mgr->nFreeTheta();

        torch::Tensor grad_params;

        if (params_mgr->hasCouplingMatrix()) {
            // Coupling matrix mode: transform ∂L/∂v → ∂L/∂p
            const auto& cm = params_mgr->couplingMatrix();
            int na = cm.n_amps;
            int ncf = cm.n_free;

            auto original_params = ctx->saved_data["original_params"].toTensor();
            const double* d_p = original_params.data_ptr<double>();

            // Current v values for gradient transform
            auto v_buf = torch::empty({na}, torch::TensorOptions()
                .dtype(TORCH_COMPLEX).device(global_extended_grad.device()));
            ctComplex* d_v = reinterpret_cast<ctComplex*>(v_buf.data_ptr());
            params_mgr->applyCouplingMatrix(d_p, d_v);

            // Gradient w.r.t coupling params [2*ncf] in [Re_p, Im_p] format
            auto grad_p = torch::empty({2 * ncf},
                torch::TensorOptions().dtype(torch::kFloat64).device(global_extended_grad.device()));
            double* d_grad_p = grad_p.data_ptr<double>();
            ctComplex* d_grad_v = reinterpret_cast<ctComplex*>(global_extended_grad.data_ptr());
            params_mgr->transformCouplingGradient(d_grad_v, d_v, d_p, d_grad_p);

            grad_params = (nt > 0 && grad_theta.numel() > 0)
                ? torch::cat({grad_p, grad_theta})
                : grad_p;
        } else {
            // Legacy mode: collapseVectorGrad + real/imag extraction
            torch::Tensor grad_vec = params_mgr->collapseVectorGrad(global_extended_grad, nv);
            torch::Tensor grad_real = (torch::real(grad_vec) * 2.0).to(torch::kFloat64);
            torch::Tensor grad_imag = (torch::imag(grad_vec) * 2.0).to(torch::kFloat64);

            if (nt > 0 && grad_theta.numel() > 0) {
                grad_params = torch::cat({grad_real, grad_imag, grad_theta});
            } else {
                grad_params = torch::cat({grad_real, grad_imag});
            }
        }

        auto tB1 = std::chrono::high_resolution_clock::now();
        if (tprof) printf("[PROF] backward: %.2f ms\n",
            std::chrono::duration<double, std::milli>(tB1 - tB0).count());
        return { grad_params * grad_outputs[0],
                torch::Tensor(),   // params_mgr
                torch::Tensor(), torch::Tensor(), torch::Tensor(),
                torch::Tensor(), torch::Tensor(), torch::Tensor(),
                torch::Tensor(), torch::Tensor(), torch::Tensor(),
                torch::Tensor(), torch::Tensor(), torch::Tensor(),
                torch::Tensor() };  // d_phsp_matrix_double_（流式 phsp 矩阵，裸指针无梯度）
    }
};
