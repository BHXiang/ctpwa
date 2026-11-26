// src/AnalysisMethods.cu — analysis 的剩余成员实现（P4：从 Analysis.h 内联定义搬出）
//
// 这些方法此时都已经是"装配"形态（填 ctx + 调自由函数），搬出来只是为了让头文件
// 只留声明。方法体一字未改。
#include <Analysis.h>

void analysis::release()
{

        int saved_dev = 0;
        if (cudaGetDevice(&saved_dev) != cudaSuccess || cudaFree(nullptr) != cudaSuccess) {
            (void)cudaGetLastError();
            d_all_amplitudes_.clear();
            initialized_ = false;
            return;
        }

        for (size_t g = 0; g < d_all_amplitudes_.size(); ++g) {
            cudaSetDevice(static_cast<int>(g));
            if (d_all_amplitudes_[g]) {
                cudaFree(d_all_amplitudes_[g]);
                d_all_amplitudes_[g] = nullptr;
            }
        }
        d_all_amplitudes_.clear();

        ctpwa::ensure_device(primary_dev_);
        if (d_phsp_matrix_) {
            cudaFree(d_phsp_matrix_);
            d_phsp_matrix_ = nullptr;
        }
        if (d_phsp_matrix_double_) {
            cudaFree(d_phsp_matrix_double_);
            d_phsp_matrix_double_ = nullptr;
        }
        auto free_weights = [](std::vector<double*>& weights) {
            for (size_t g = 0; g < weights.size(); ++g) {
                cudaSetDevice(static_cast<int>(g));
                if (weights[g]) cudaFree(weights[g]);
            }
            weights.clear();
        };
        free_weights(data_weights_);
        free_weights(phsp_weights_);
        free_weights(bkg_weights_);

        // 这些表也按 primary_dev_ 分配，先切到主卡再释放。
        params_.freeCouplingData();
        // 释放各 ResBlock、d_dF/static/JIT、slot 及 cas_list_ 的持久化资源。
        // AmpCalc::release 内部会保存并恢复 current device。
        amp_calc_.release();
        initialized_ = false;
        // 析构/释放不能成为新的 current-device 污染源：恢复进入前的设备。
        cudaSetDevice(saved_dev);
    
}

double analysis::computeFileBatchIntegrals(
    const std::string& file_key,
    const torch::Tensor& ev_center,
    const std::vector<torch::Tensor>& ev_perturbed_p,
    const std::vector<torch::Tensor>& ev_perturbed_m,
    double* out_partial,
    double* out_scattering,
    double* out_square,
    double* out_p_partial,
    double* out_p_scattering,
    double* out_m_partial,
    double* out_m_scattering,
    int npartials)
{

        // P4：实现已搬到 ctpwa::computeFileBatchIntegrals（include/BatchIntegrals.h）。
        // 状态显式化 + 两个回调（θ 读取、按显式 θ 算振幅）。
        ctpwa::BatchIntegralCtx ctx;
        ctx.data_files = &config_parser_.getDataFiles();
        ctx.data_order = &config_parser_.getDataOrder();
        ctx.particles = &particles_;
        ctx.events_offsets = &events_offsets_;
        ctx.amp_offsets = &amp_offsets_;
        ctx.n_slvectors = &nSLvectors_;
        ctx.n_amplitudes = n_amplitudes_;
        ctx.n_polar = n_polar_;
        ctx.float_amps = float_amps_;
        ctx.n_polar = n_polar_;
        ctx.n_gpus = n_gpus_;
        ctx.primary_dev = primary_dev_;
        ctx.float_amps = float_amps_;
        ctx.current_theta_host = [this]() { return currentThetaHost(); };
        ctx.calculate_amplitudes = [this](
            const std::vector<std::map<std::string, std::vector<LorentzVector>>>& Vp4,
            int active_gpus, const double* d_theta) {
            return calculateAmplitudes(Vp4, nullptr, active_gpus, d_theta);
        };
        return ctpwa::computeFileBatchIntegrals(
            ctx, file_key, ev_center, ev_perturbed_p, ev_perturbed_m,
            out_partial, out_scattering, out_square,
            out_p_partial, out_p_scattering, out_m_partial, out_m_scattering, npartials);
    
}

std::vector<torch::Tensor> analysis::computeMatrixErrors(
    torch::Tensor params, torch::Tensor hessian_in)
{

        // P4：计算在 ctpwa::computeMatrixErrors（include/MatrixErrors.h）。
        // 状态走 ctx 字段，算法走回调——本类只做装配。
        ctpwa::MatrixErrorCtx ctx;
        ctx.events = &events_;
        ctx.phsp_weights = &phsp_weights_;
        ctx.n_slvectors = &nSLvectors_;
        ctx.d_phsp_matrix = d_phsp_matrix_;
        ctx.d_phsp_matrix_double = d_phsp_matrix_double_;
        ctx.n_amplitudes = n_amplitudes_;
        ctx.n_theta = params_.nFreeTheta();
        ctx.primary_dev = primary_dev_;
        ctx.has_gen = config_parser_.getDataFiles().count("phsp_truth") > 0;
        ctx.n_vector = [this]() { return getNVector(); };
        ctx.free_params_to_amplitudes =
            [this](const torch::Tensor& cpl) { return freeParamsToAmplitudes(cpl); };
        ctx.hessian = [this](const torch::Tensor& p) { return getHessian(p); };
        ctx.set_theta = [this](const torch::Tensor& t) { theta_cur_ = t; };
        ctx.get_theta = [this]() { return theta_cur_; };
        ctx.batch_integrals_full =
            [this](const std::string& sample, const torch::Tensor& ev,
                   const std::vector<torch::Tensor>& ev_p,
                   const std::vector<torch::Tensor>& ev_m,
                   double* diag, double* off, double* sq,
                   double* pp, double* ps, double* mp, double* ms, int npart) {
                return computeFileBatchIntegrals(sample, ev, ev_p, ev_m,
                                                 diag, off, sq, pp, ps, mp, ms, npart);
            };
        return ctpwa::computeMatrixErrors(ctx, params, hessian_in);
    
}

void analysis::initializeMultiGPUs(std::vector<int> init_events)
{

        // 根据GPU个数分配evets给events_，每个元素都均匀分配
        if (n_gpus_ != 0)
        {
            events_.clear();                         // 清空旧数据
            events_.resize(n_gpus_);                 // 为每个 GPU 预留一行

            // 为每个 GPU 的每一列（事件类型）预先分配空间
            for (int gpu = 0; gpu < n_gpus_; ++gpu) {
                events_[gpu].resize(init_events.size());
            }

            // 对每一种事件类型进行分配
            for (size_t j = 0; j < init_events.size(); ++j) {
                int val = init_events[j];            // 该类事件的总数
                int base = val / n_gpus_;            // 每个 GPU 至少获得的数量
                int remainder = val % n_gpus_;       // 余数，前 remainder 个 GPU 各多得 1

                for (int gpu = 0; gpu < n_gpus_; ++gpu) {
                    events_[gpu][j] = base + (gpu < remainder ? 1 : 0);
                }
            }
        }
        else
        {
            std::cerr << "No GPUs available. Running on CPU." << std::endl;
        }
    
}

torch::Tensor analysis::getHessian(torch::Tensor params)
{

        // P4：实现已搬到 ctpwa::getHessian（include/FullHessian.h）。
        // 十二轮的"先搬底层"到这里收口：它曾经的 12 个依赖只剩 1 个还需要回调。
        ctpwa::FullHessianCtx ctx;
        ctx.d_all_amplitudes = &d_all_amplitudes_;
        ctx.events = &events_;
        ctx.events_offsets = &events_offsets_;
        ctx.amp_offsets = &amp_offsets_;
        ctx.data_weights = &data_weights_;
        ctx.bkg_weights = &bkg_weights_;
        ctx.phsp_weights = &phsp_weights_;
        ctx.d_phsp_matrix = d_phsp_matrix_;
        ctx.params = &params_;
        ctx.amp_calc = &amp_calc_;
        ctx.n_amplitudes = n_amplitudes_;
        ctx.n_polar = n_polar_;
        ctx.primary_dev = primary_dev_;
        ctx.float_amps = float_amps_;
        ctx.phsp_freed = phsp_freed_;
        ctx.data_total_weight = data_total_weight_;
        ctx.bkg_integral = bkg_integral_;
        ctx.free_params_to_amplitudes =
            [this](const torch::Tensor& v) { return freeParamsToAmplitudes(v); };
        return ctpwa::getHessian(ctx, params);
    
}

torch::Tensor analysis::getNLL(torch::Tensor params)
{

        TORCH_CHECK(initialized_, "analysis not initialized: invalid or missing config file");

        TORCH_CHECK(params.dtype() == torch::kFloat64, "params must be float64");
        TORCH_CHECK(ctpwa_on_primary(params, primary_dev_),
            "params 必须位于主设备 (primary=" + std::to_string(primary_dev_) + ")，"
            "当前在 " + params.device().str());

        // 记录当前 θ：真值/积分路径（getFitFractions/getEfficiency/矩阵）不读驻留
        // 振幅，必须显式带 θ；这里顺手记下，拟合循环里最后一次调用的 θ 就是最优点。
        if (params_.nFreeTheta() > 0) {
            const int n_cpl = getNVector();
            if (params.numel() == 2 * n_cpl + params_.nFreeTheta())
                theta_cur_ = params.slice(0, 2 * n_cpl, params.size(0)).detach().clone();
        }

        return NLLFunction::apply(params, &params_, d_all_amplitudes_, &amp_calc_,
            d_phsp_matrix_, d_phsp_matrix_double_, events_, events_offsets_, amp_offsets_,
            data_weights_, phsp_weights_, bkg_weights_, bkg_integral_, n_amplitudes_, n_polar_);
    
}

torch::Tensor analysis::computeCouplingHessian(torch::Tensor vector)
{

        // P4：实现已搬到 ctpwa::computeCouplingHessian（include/CouplingHessian.h）
        ctpwa::CouplingHessianCtx ctx;
        ctx.d_all_amplitudes = &d_all_amplitudes_;
        ctx.events = &events_;
        ctx.amp_offsets = &amp_offsets_;
        ctx.data_weights = &data_weights_;
        ctx.bkg_weights = &bkg_weights_;
        ctx.d_phsp_matrix = d_phsp_matrix_;
        ctx.params = &params_;
        ctx.n_amplitudes = n_amplitudes_;
        ctx.n_polar = n_polar_;
        ctx.float_amps = float_amps_;
        ctx.primary_dev = primary_dev_;
        ctx.data_total_weight = data_total_weight_;
        ctx.bkg_integral = bkg_integral_;
        ctx.free_params_to_amplitudes =
            [this](const torch::Tensor& v) { return freeParamsToAmplitudes(v); };
        return ctpwa::computeCouplingHessian(ctx, vector);
    
}

void analysis::reCalcAmp(torch::Tensor params)
{

        TORCH_CHECK(ctpwa_is_cuda_like(params), "params 必须位于当前后端的设备上（CPU 构建用 CPU 张量）");
        TORCH_CHECK(params.dtype() == torch::kFloat64, "params must be float64");
        TORCH_CHECK(params.dim() == 1, "params must be 1-dimensional");
        TORCH_CHECK(params.numel() == amp_calc_.nFreeResParams(),
            "params size mismatch: got ", params.numel(),
            ", expected ", amp_calc_.nFreeResParams());

        theta_cur_ = params.detach().clone();
        amp_calc_.reComputeAmps(d_all_amplitudes_,
            reinterpret_cast<const double*>(params.data_ptr()),
            n_amplitudes_, events_offsets_, amp_offsets_, n_polar_,
            params.device().index());
        // θ 变了就要把 phsp 的 Gram 一起更新，否则常驻 B̄ 还是旧 θ 的
        //（此前只有 getHessian 会更新，reCalcAmp 之后矩阵/份额会拿到过期值）。
        if (!phsp_freed_) rebuildPhspGram();
    
}

