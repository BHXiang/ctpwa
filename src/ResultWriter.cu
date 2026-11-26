// src/ResultWriter.cu — analysis 的结果写出（P4：从 Analysis.h 内联定义搬出）
//
// 为什么要单独成 TU：这两个方法合计 ~1836 行，是 god class 里最大的一块**写文件**
// 逻辑。第 8 轮把 class analysis 搬进头文件之后，"类内内联定义 → 类内声明 + 本 TU 定义"
// 就是纯搬家：不改一行方法体、不动任何接口，却让头文件小三分之一。
//
// 这也是 P4 之后所有方法迁出的通用手法（比"改成自由函数 + ctx"省事得多，
// 只在真要把方法变成模块接口时才需要那一步）。
#include <Analysis.h>

void analysis::writeResult(torch::Tensor params, const std::string& filename,
                           const int is_saved_weight,
                           const std::vector<int>& waves,
                           torch::Tensor hessian_in)
{
        ctpwa::ensure_device(primary_dev_);   // 入口恢复主设备
        TORCH_CHECK(ctpwa_is_cuda_like(params), "params 必须位于当前后端的设备上（CPU 构建用 CPU 张量）");

        // 流式 phsp 分支会临时改写 offsets；异常时也要还原。
        auto wr_ev0 = events_offsets_, wr_amp0 = amp_offsets_;
        auto wr_guard = makeScopeExit([this, wr_ev0, wr_amp0]() {
            events_offsets_ = wr_ev0;
            amp_offsets_ = wr_amp0;
            ctpwa::ensure_device(primary_dev_);
        });

        int npartials_all = (int)nSLvectors_.size();
        std::vector<int> wave_mask_host(npartials_all, 0);
        bool has_waves = !waves.empty();
        if (has_waves) {
            for (int w : waves) {
                TORCH_CHECK(w >= 0 && w < npartials_all,
                    "waves 下标越界: ", w, " (partials = ", npartials_all, ")");
                wave_mask_host[w] = 1;
            }
        }

        torch::Device dev = params.device();
        torch::Tensor extended_vector;

        if (params_.hasCouplingMatrix()) {
            TORCH_CHECK(params.dtype() == torch::kFloat64, "params must be float64 with coupling matrix");
            const auto& cm = params_.couplingMatrix();
            int ncf = cm.n_free;
            int na  = cm.n_amps;
            int nt  = static_cast<int>(params.size(0)) - 2 * ncf;

            // Extend: [Re_p, Im_p, θ] → [Re_v, Im_v, θ]
            auto ext = torch::empty({2 * na + nt},
                torch::TensorOptions().dtype(torch::kFloat64).device(dev));
            params_.extendCouplingParams(
                params.data_ptr<double>(), ext.data_ptr<double>(), ncf, nt);

            // Build complex vector from extended format
            extended_vector = torch::complex(
                ext.slice(0, 0, na).to(TORCH_FLOAT),
                ext.slice(0, na, 2 * na).to(TORCH_FLOAT));

            // Recompute amplitudes if theta params present
            if (nt > 0) {
                auto theta = params.slice(0, 2 * ncf, params.size(0));
                theta_cur_ = theta.detach().clone();
                amp_calc_.reComputeAmps(d_all_amplitudes_,
                    reinterpret_cast<const double*>(theta.data_ptr()),
                    n_amplitudes_, events_offsets_, amp_offsets_, n_polar_,
                    dev.index());
                // θ 变了：常驻 B̄ 同步刷新（否则再算矩阵还是旧 θ 的）
                if (!phsp_freed_) rebuildPhspGram();
            }
        } else {
            TORCH_CHECK(params.dtype() == TORCH_COMPLEX, "params must be complex128 in legacy mode");
            extended_vector = params_.extendVector(params, dev);
        }

        // 将extended_vector分配到多个GPU
        std::vector<torch::Tensor> extended_vec_per_gpu;
        for (int i = 0; i < d_all_amplitudes_.size(); ++i) {
            extended_vec_per_gpu.push_back(extended_vector.to(ctpwa_dev(i)));
        }

        const int target_dev = dev.index();

        int npartials = nSLvectors_.size();
        // 每个GPU在自己的设备上分配输出缓冲区
        std::vector<double*> d_final_result_vec;
        std::vector<double*> d_partial_result_vec;
        for (int i = 0; i < d_all_amplitudes_.size(); ++i) {
            cudaSetDevice(i);
            double* d_final_result;
            cudaMalloc(&d_final_result, events_[i][0] * sizeof(double));
            d_final_result_vec.push_back(d_final_result);
            double* d_partial_result;
            cudaMalloc(&d_partial_result, events_[i][0] * npartials * sizeof(double));
            // ⚠️ 必须清零: kernel 对 partial 用 += 累加（总权重是 = 赋值），
            // 不清零 = 垃圾内存 → weight_<i> 分支一直是错的（守恒校验发掘）
            cudaMemset(d_partial_result, 0,
                events_[i][0] * npartials * sizeof(double));
            d_partial_result_vec.push_back(d_partial_result);
        }
        cudaSetDevice(target_dev);

        // 分配nSLvectors_的设备内存
        // int* d_nSLvectors;
        // cudaMalloc(&d_nSLvectors, nSLvectors_.size() * sizeof(int));
        // cudaMemcpy(d_nSLvectors, nSLvectors_.data(), npartials * sizeof(int), cudaMemcpyHostToDevice);

        // 分配干涉矩阵和事件干涉项的设备内存
        // 干涉矩阵不再由 computeResults 累加（改由 computeFileBatchIntegrals 出
        // interference_rec/gen），这里传 nullptr 省掉每事例 n² 次原子操作。

        int N_phsp = 0;
        int N_data = 0;
        int N_bkg = 0;
        for (size_t gpu = 0; gpu < d_all_amplitudes_.size(); ++gpu) {
            cudaSetDevice(gpu);
            int phsp_events = events_[gpu][0];
            N_phsp += static_cast<int>(phsp_events);
            int data_events = events_[gpu][1];
            N_data += static_cast<int>(data_events);
            int bkg_events = (events_[gpu].size() > 2) ? events_[gpu][2] : 0;
            N_bkg += static_cast<int>(bkg_events);
        }

        double h_phsp_integral = 0.0;
        double* h_total_results = new double[N_phsp];
        double* h_partial_results = new double[N_phsp * npartials];
        // 中途抛异常（Hessian 维度/ROOT 写入失败等）时也不能漏掉这两块 host
        // 缓冲：大样本下 N_phsp×npartials×8B 可达 GB 级。
        auto h_guard = makeScopeExit([&]() {
            delete[] h_total_results;
            delete[] h_partial_results;
        });
        int ev_cumulative = 0;  // 多GPU累加偏移
        for (size_t gpu = 0; gpu < d_all_amplitudes_.size(); ++gpu) {
            cudaSetDevice(gpu);
            int* d_nSLvectors;
            cudaMalloc(&d_nSLvectors, nSLvectors_.size() * sizeof(int));
            cudaMemcpy(d_nSLvectors, nSLvectors_.data(), npartials * sizeof(int), cudaMemcpyHostToDevice);
            // 单个GPU integral
            double* d_total_integral_gpu;
            cudaMalloc(&d_total_integral_gpu, sizeof(double));
            cudaMemset(d_total_integral_gpu, 0, sizeof(double));
            // 单个GPU event interference（本函数不用, 传 nullptr; writeInterfResult 使用）
            // 子集选择（waves 非空时）: mask + 子集积分缓冲
            int* d_wave_mask_gpu = nullptr;
            double* d_selected_integral_gpu = nullptr;
            if (has_waves) {
                cudaMalloc(&d_wave_mask_gpu, npartials * sizeof(int));
                cudaMemcpy(d_wave_mask_gpu, wave_mask_host.data(),
                    npartials * sizeof(int), cudaMemcpyHostToDevice);
                cudaMalloc(&d_selected_integral_gpu, sizeof(double));
                cudaMemset(d_selected_integral_gpu, 0, sizeof(double));
            }

            /////////////////////////
            /////////////////////////
            if (!phsp_freed_) {
                // phsp 驻留模式: 逐批上转 + computeResults 累加。原实现整表
                // upcast(double 副本) + computeResults 内部整段临时, 大 phsp 下
                // 峰值 = 2×PA×nEv×16B(可达 7GB+), A100-40GB 实测 OOM →
                // 改按批（峰值 ~批内 2×PA×16B, 默认 100k 事件 ≈ 0.3GB）。
                int ne = events_[gpu][0];
                if (ne > 0) {
                    long long per_ev_bytes = (long long)n_polar_ * n_amplitudes_ * 8;
                    int batch_cap = 100000;
                    if (const char* e = getenv("CTPWA_WR_BATCH")) {
                        int v = atoi(e);
                        if (v > 0) batch_cap = v;   // 显式覆盖（回归测试/调优用）
                    } else if (per_ev_bytes > 0) {
                        long long cap = 200LL * 1024 * 1024 / per_ev_bytes;
                        if (cap < 2000) cap = 2000;
                        if (cap < batch_cap) batch_cap = (int)cap;
                    }
                    const ctComplex* d_amp_base = d_all_amplitudes_[gpu];
                    for (int c0 = 0; c0 < ne; c0 += batch_cap) {
                        int nch = std::min(batch_cap, ne - c0);
                        const ctComplex* d_amp_seg = d_amp_base;
                        ctComplex* d_amp_dbl = nullptr;
                        if (float_amps_ && !ctIsFloatComplex) {
                            // 注意: A 表为 float2(8B) 存储而 d_amp_base 是
                            // ctComplex*(16B 语义) —— 段偏移必须按 float2 算术,
                            // 否则 c0>0 的批指针翻倍错位（分批后 hfit 全 nan）
                            const float2* f2 =
                                reinterpret_cast<const float2*>(d_amp_base)
                                + (size_t)c0 * n_polar_ * n_amplitudes_;
                            d_amp_dbl = upcastAmpSegToDouble(
                                reinterpret_cast<const ctComplex*>(f2),
                                (size_t)nch * n_polar_ * n_amplitudes_);
                            d_amp_seg = d_amp_dbl;
                        } else {
                            d_amp_seg =
                                d_amp_base + (size_t)c0 * n_polar_ * n_amplitudes_;
                        }
                        double* d_ct = nullptr;
                        double* d_cp = nullptr;
                        if (cudaMalloc(&d_ct, (size_t)nch * sizeof(double))
                                != cudaSuccess
                            || cudaMalloc(&d_cp, (size_t)nch * npartials
                                                     * sizeof(double))
                                   != cudaSuccess) {
                            char buf[256];
                            snprintf(buf, sizeof(buf),
                                     "writeResult: 批缓冲分配失败 (nch=%d)",
                                     nch);
                            throw std::runtime_error(buf);
                        }
                        cudaMemset(d_cp, 0,
                                   (size_t)nch * npartials * sizeof(double));
                        computeResults(d_amp_seg,
                            reinterpret_cast<const ctComplex*>(
                                extended_vec_per_gpu[gpu].data_ptr()),
                            d_ct, d_total_integral_gpu, d_cp,
                            nullptr, nullptr,
                            d_wave_mask_gpu, d_selected_integral_gpu,
                            d_ct,
                            nullptr, 0,
                            d_nSLvectors, npartials, nch, n_amplitudes_,
                            n_polar_);
                        // 批结果拷回全尺寸缓冲偏移（total 段 = 赋值, partial
                        // 段 [p][ev] 主序）
                        cudaMemcpy(d_final_result_vec[gpu] + c0, d_ct,
                                   (size_t)nch * sizeof(double),
                                   cudaMemcpyDeviceToDevice);
                        for (int p = 0; p < npartials; ++p)
                            cudaMemcpy(d_partial_result_vec[gpu]
                                           + (size_t)p * ne + c0,
                                       d_cp + (size_t)p * nch,
                                       (size_t)nch * sizeof(double),
                                       cudaMemcpyDeviceToDevice);
                        cudaFree(d_ct);
                        cudaFree(d_cp);
                        if (d_amp_dbl) cudaFree(d_amp_dbl);
                    }
                }
            } else {
                // phsp 流式模式: 按批重算 phsp 振幅 → computeResults 逐批累加
                int ne = events_[gpu][0];
                if (ne > 0) {
                    const int PHSP_BATCH = 100000;
                    long long per_ev_bytes = (long long)n_polar_ * n_amplitudes_ * 8;
                    int batch_cap = PHSP_BATCH;
                    if (per_ev_bytes > 0) {
                        long long cap = 200LL * 1024 * 1024 / per_ev_bytes;
                        if (cap < 2000) cap = 2000;
                        if (cap < batch_cap) batch_cap = (int)cap;
                    }
                    auto saved_ev = events_offsets_, saved_amp = amp_offsets_;
                    for (int c0 = 0; c0 < ne; c0 += batch_cap) {
                        int nch = std::min(batch_cap, ne - c0);
                        std::vector<std::map<std::string, std::vector<LorentzVector>>> Vp4_chunk(n_gpus_);
                        for (int g = 0; g < n_gpus_; ++g) {
                            if (g == (int)gpu) {
                                for (const auto& [k, v] : Vp4_all_[gpu])
                                    Vp4_chunk[g][k].assign(v.begin() + c0, v.begin() + c0 + nch);
                                events_offsets_[g] = {0, nch};
                                amp_offsets_[g] = {0, nch * n_polar_ * n_amplitudes_};
                            } else {
                                for (const auto& [k, v] : Vp4_all_[g])
                                    Vp4_chunk[g][k].assign(v.begin(), v.begin() + 1);
                                events_offsets_[g] = {0, 1};
                                amp_offsets_[g] = {0, n_polar_ * n_amplitudes_};
                            }
                        }
                        std::vector<ctComplex*> d_chunk = calculateAmplitudes(Vp4_chunk);
                        cudaSetDevice(gpu);
                        double* d_ct;
                        cudaMalloc(&d_ct, nch * sizeof(double));
                        double* d_cp;
                        cudaMalloc(&d_cp, (size_t)nch * npartials * sizeof(double));
                        cudaMemset(d_cp, 0, (size_t)nch * npartials * sizeof(double));
                        {
                        const ctComplex* d_chunk_res = d_chunk[gpu];
                        ctComplex* d_chunk_dbl = nullptr;
                        if (float_amps_ && !ctIsFloatComplex) {
                            size_t nElem = (size_t)nch * n_polar_ * n_amplitudes_;
                            d_chunk_dbl = upcastAmpSegToDouble(d_chunk_res, nElem);
                            d_chunk_res = d_chunk_dbl;
                        }
                        computeResults(d_chunk_res,
                            reinterpret_cast<const ctComplex*>(extended_vec_per_gpu[gpu].data_ptr()),
                            d_ct, d_total_integral_gpu, d_cp,
                            nullptr, nullptr,
                            d_wave_mask_gpu, d_selected_integral_gpu,
                            d_ct,
                            nullptr, 0,
                            d_nSLvectors, npartials, nch, n_amplitudes_, n_polar_);
                        if (d_chunk_dbl) cudaFree(d_chunk_dbl);
                        }
                        // 拷贝进全尺寸结果缓冲（chunk 结果 → 偏移 c0）
                        cudaMemcpy(d_final_result_vec[gpu] + c0, d_ct,
                                   nch * sizeof(double), cudaMemcpyDeviceToDevice);
                        for (int p = 0; p < npartials; ++p)
                            cudaMemcpy(d_partial_result_vec[gpu] + (size_t)p * ne + c0,
                                       d_cp + (size_t)p * nch, nch * sizeof(double),
                                       cudaMemcpyDeviceToDevice);
                        cudaFree(d_ct);
                        cudaFree(d_cp);
                        for (int g = 0; g < n_gpus_; ++g)
                            if (d_chunk[g]) { cudaSetDevice(g); cudaFree(d_chunk[g]); }
                        cudaSetDevice(gpu);
                    }
                    events_offsets_ = saved_ev;
                    amp_offsets_ = saved_amp;
                }
            }

            // 将单个GPU的结果累加到全局结果
            double h_total_integral_gpu;
            if (has_waves) {
                cudaMemcpy(&h_total_integral_gpu, d_selected_integral_gpu,
                    sizeof(double), cudaMemcpyDeviceToHost);
            } else {
                cudaMemcpy(&h_total_integral_gpu, d_total_integral_gpu,
                    sizeof(double), cudaMemcpyDeviceToHost);
            }
            h_phsp_integral += h_total_integral_gpu;
            double* h_total_results_gpu = new double[events_[gpu][0]];
            cudaMemcpy(h_total_results_gpu, d_final_result_vec[gpu], events_[gpu][0] * sizeof(double), cudaMemcpyDeviceToHost);
            std::copy(h_total_results_gpu, h_total_results_gpu + events_[gpu][0], h_total_results + ev_cumulative);
            // 每 GPU 块是 [分量][事件]（p-major）→ 展开到全局 [分量][全事件] 布局
            for (int p = 0; p < npartials; ++p)
                cudaMemcpy(h_partial_results + (size_t)p * N_phsp + ev_cumulative,
                    d_partial_result_vec[gpu] + (size_t)p * events_[gpu][0],
                    events_[gpu][0] * sizeof(double), cudaMemcpyDeviceToHost);
            ev_cumulative += events_[gpu][0];
            cudaFree(d_total_integral_gpu);
            cudaFree(d_nSLvectors);
            if (d_wave_mask_gpu) cudaFree(d_wave_mask_gpu);
            if (d_selected_integral_gpu) cudaFree(d_selected_integral_gpu);
            delete[] h_total_results_gpu;

        }

        // bkg weights积分
        double h_bkg_integral = 0.0;
        // if (bkg_weights_ != nullptr && bkg_length > 0)
        if (!bkg_weights_.empty() && N_bkg > 0)
        {
            h_bkg_integral = bkg_integral_;
            // // thrust::device_ptr<double> d_ptr(bkg_weights_);
            // // std::cout << "Calculating background integral with " << events_[2] << " events..." << std::endl;
            // // h_bkg_integral = thrust::reduce(d_ptr, d_ptr + events_[2]);
            // // h_bkg_integral = thrust::reduce(d_ptr, d_ptr + bkg_length /
            // // n_polar_, 0.0, thrust::plus<double>());
            // for (size_t gpu = 0; gpu < d_all_amplitudes_.size(); ++gpu) {
            //     cudaSetDevice(gpu);
            //     if (events_[gpu][2] > 0) {
            //         // double* d_bkg_weights_gpu;
            //         // cudaMalloc(&d_bkg_weights_gpu, events_[gpu][2] * sizeof(double));
            //         // cudaMemcpy(d_bkg_weights_gpu, bkg_weights_[gpu], events_[gpu][2] * sizeof(double), cudaMemcpyDeviceToDevice);
            //         thrust::device_ptr<double> d_ptr_gpu(bkg_weights_[gpu]);
            //         h_bkg_integral += thrust::reduce(d_ptr_gpu, d_ptr_gpu + events_[gpu][2]);
            //         // cudaFree(d_bkg_weights_gpu);
            //     }
            // }
        }

        // int dataIntegral = data_length / n_polar_;
        // if (bkg_fix_ != nullptr && bkg_length > 0)
        // data 加权时用总权重（无权重时 = 事件数），与 NLL 的 N_eff 一致
        double dataIntegral = data_total_weight_;
        if (N_bkg > 0)
        {
            if (h_bkg_integral > 0)
            {
                dataIntegral -= h_bkg_integral;
            }
            else
            {
                dataIntegral -= N_bkg;
            }
        }
        // 空 phsp（N_phsp=0）时 h_phsp_integral=0；inf 会把所有 plot 直方图
        // scale 成 inf。没有 phsp 就没有可画的拟合分布，保持 1.0 即可。
        double normFactor = (h_phsp_integral > 0.0)
            ? static_cast<double>(dataIntegral) / h_phsp_integral : 1.0;

        TFile* rootFile = new TFile(filename.c_str(), "RECREATE");
        // 中途抛异常（Hessian 形状/维度、积分扫描失败等）时也必须关闭 ROOT
        // 文件，否则 TFile 缓冲区不落盘，后续再写同名文件会拿到半成品。
        auto root_guard = makeScopeExit([&]() {
            if (rootFile) { rootFile->Close(); delete rootFile; rootFile = nullptr; }
        });

        TTree* legend = new TTree("legends", "Amplitude Legends");
        legend->Branch("legend", &legends_);
        legend->Fill();
        legend->Write();
        delete legend;


        if (is_saved_weight == 1)
        {
            TTree* phspTree = new TTree("saved_weight", "fitting result weights");

            // 添加权重分支
            double total_weight;
            phspTree->Branch("totalweight", &total_weight);

            // 为每个部分波创建分支
            std::vector<double> partial_weights(npartials);
            for (int i = 0; i < npartials; ++i)
            {
                std::string branch_name = "weight_" + resonance_names_[i];
                phspTree->Branch(branch_name.c_str(), &partial_weights[i]);
            }

            // 末态粒子四动量分支（任意分布按需现算: M/θ/φ/cosβ/... 全部可由它导出）
            std::vector<std::string> mom_names;
            if (!Vp4_all_.empty())
                for (const auto& kv : Vp4_all_[0])
                    mom_names.push_back(kv.first);
            const int n_mom = (int)mom_names.size();
            std::vector<LorentzVector> mom_host((size_t)N_phsp * (size_t)std::max(1, n_mom));
            {
                int off = 0;
                for (size_t gpu = 0; gpu < Vp4_all_.size(); ++gpu) {
                    int ne = events_[gpu][0];
                    for (int m = 0; m < n_mom; ++m) {
                        const auto& vec = Vp4_all_[gpu].at(mom_names[m]);
                        for (int e = 0; e < ne; ++e)
                            mom_host[(size_t)(off + e) * n_mom + m] = vec[e];
                    }
                    off += ne;
                }
            }
            std::vector<double> mom_vals((size_t)std::max(1, n_mom) * 4, 0.0);
            for (int m = 0; m < n_mom; ++m)
            {
                const std::string& nm = mom_names[m];
                phspTree->Branch((nm + "_px").c_str(), &mom_vals[m * 4 + 0]);
                phspTree->Branch((nm + "_py").c_str(), &mom_vals[m * 4 + 1]);
                phspTree->Branch((nm + "_pz").c_str(), &mom_vals[m * 4 + 2]);
                phspTree->Branch((nm + "_E").c_str(),  &mom_vals[m * 4 + 3]);
            }

            // 填充 phsp tree
            // ⚠️ kernel 缓冲是 [分量][事件]（p-major: p*N+ev）, 读取必须同布局——
            // 旧代码按事件主序读 = 转置错位（weight_<i> 分支一直是错配的）
            for (int i = 0; i < N_phsp; ++i)
            {
                // 设置权重
                total_weight = h_total_results[i] / h_phsp_integral * static_cast<double>(dataIntegral);
                // std::cout << "Event " << i << ": Total Weight = " <<
                // total_weight << std::endl;
                for (int j = 0; j < npartials; ++j)
                {
                    partial_weights[j] = h_partial_results[(size_t)j * N_phsp + i] * normFactor;
                    // std::cout << "  Partial Weight " << j << " = " <<
                    // partial_weights[j] << std::endl;
                }
                for (int m = 0; m < n_mom; ++m)
                {
                    const LorentzVector& mv = mom_host[(size_t)i * n_mom + m];
                    mom_vals[m * 4 + 0] = mv.Px;
                    mom_vals[m * 4 + 1] = mv.Py;
                    mom_vals[m * 4 + 2] = mv.Pz;
                    mom_vals[m * 4 + 3] = mv.E;
                }

                phspTree->Fill();
            }

            phspTree->Write();
            delete phspTree;
        }

        auto plotconfig = config_parser_.getPlotConfigs();
        std::vector<MassHistConfig> masshist;
        std::vector<AngleHistConfig> anglehist;
        std::vector<DalitzHistConfig> dalitzhist;
        int mass_hist_count = 0;
        int angle_hist_count = 0;
        int dalitz_hist_count = 0;
        for (const auto& histConfig : plotconfig)
        {
            // 输出histConfig内容以进行调试
            // std::cout << "PlotConfig type: " << histConfig.type << std::endl;
            if (histConfig.type == "mass")
            {
                std::vector<std::string> particles = histConfig.particles[0];
                int bins = histConfig.bins[0];
                std::vector<double> range = histConfig.ranges[0];
                std::vector<std::string> display = histConfig.display;

                std::string hist_name =
                    "mass" + std::to_string(mass_hist_count++);
                for (const auto& p : particles)
                {
                    hist_name += "_" + p;
                }
                std::cout << "Creating mass histogram: " << hist_name << std::endl;
                masshist.emplace_back(hist_name, "", particles, bins, range, display);
            }
            else if (histConfig.type == "cosbeta")
            {
                // 处理角度直方图配置
                std::vector<std::vector<std::string>> particles = histConfig.particles;
                int bins = histConfig.bins[0];
                std::vector<double> range = histConfig.ranges[0];
                std::vector<std::string> display = histConfig.display;
                std::string hist_name = "cosbeta" + std::to_string(angle_hist_count++);
                for (const auto& pvec : particles)
                {
                    hist_name += "_";
                    for (const auto& p : pvec)
                    {
                        hist_name += p;
                    }
                }
                std::cout << "Creating angle histogram: " << hist_name << std::endl;
                anglehist.emplace_back(hist_name, "", particles, bins, range, display);
            }
            else if (histConfig.type == "dalitz")
            {
                std::vector<std::vector<std::string>> particles = histConfig.particles;
                std::vector<int> bins = histConfig.bins;
                std::vector<std::vector<double>> ranges = histConfig.ranges;
                std::vector<std::string> display = histConfig.display;
                std::string hist_name = "dalitz" + std::to_string(dalitz_hist_count++);
                for (const auto& pvec : particles)
                {
                    hist_name += "_";
                    for (const auto& p : pvec)
                    {
                        hist_name += p;
                    }
                }
                std::cout << "Creating dalitz histogram: " << hist_name << std::endl;
                dalitzhist.emplace_back(hist_name, "", particles, bins, ranges, display);
            }
        }

        for (const auto& histConfig : masshist)
        {
            TDirectory* histDir = rootFile->mkdir(histConfig.name.c_str());
            histDir->cd();

            TObjString xlabel_obj(histConfig.tex[0].c_str());
            TObjString ylabel_obj(histConfig.tex[1].c_str());
            xlabel_obj.Write("xlabel", TObject::kOverwrite);
            ylabel_obj.Write("ylabel", TObject::kOverwrite);
        }

        for (const auto& histConfig : anglehist)
        {
            TDirectory* histDir = rootFile->mkdir(histConfig.name.c_str());
            histDir->cd();

            TObjString xlabel_obj(histConfig.tex[0].c_str());
            TObjString ylabel_obj(histConfig.tex[1].c_str());
            xlabel_obj.Write("xlabel", TObject::kOverwrite);
            ylabel_obj.Write("ylabel", TObject::kOverwrite);
        }

        for (const auto& histConfig : dalitzhist)
        {
            TDirectory* histDir = rootFile->mkdir(histConfig.name.c_str());
            histDir->cd();

            TObjString xlabel_obj(histConfig.tex[0].c_str());
            TObjString ylabel_obj(histConfig.tex[1].c_str());
            xlabel_obj.Write("xlabel", TObject::kOverwrite);
            ylabel_obj.Write("ylabel", TObject::kOverwrite);
        }

        // 四动量index
        std::map<std::string, int> particleToIndex;
        for (int i = 0; i < particles_.size(); ++i)
        {
            particleToIndex[particles_[i].name] = i;
        }

        std::vector<LorentzVector*> device_momenta_list = convertToLorentzVectors(Vp4_all_, particleToIndex);
        int n_particles = particleToIndex.size();

        // 计算并保存data直方图
        // data_weights_ 为空/该 GPU 无权重时返回 nullptr，填充 kernel 按权重 1 处理。
        auto data_weights_for_gpu = [this](size_t gpu) -> double* {
            return gpu < data_weights_.size() ? data_weights_[gpu] : nullptr;
        };
        if (N_data > 0)
        {
            std::vector<TH1F*> masshist_data;
            for (const auto& histConfig : masshist)
            {
                TH1F* hist = new TH1F((histConfig.name + "_data").c_str(), histConfig.title.c_str(), histConfig.bins, histConfig.range[0], histConfig.range[1]);
                masshist_data.emplace_back(hist);
            }

            // 临时直方图向量，用于每个GPU
            std::vector<std::vector<TH1F*>> temp_hists(device_momenta_list.size());
            for (size_t i = 0; i < device_momenta_list.size(); ++i)
            {
                cudaSetDevice(i);
                // 为当前GPU创建临时直方图
                std::vector<TH1F*>& temp = temp_hists[i];
                for (const auto& histConfig : masshist)
                {
                    TH1F* hist = new TH1F((histConfig.name + "_temp").c_str(), histConfig.title.c_str(), histConfig.bins, histConfig.range[0], histConfig.range[1]);
                    temp.emplace_back(hist);
                }
                // 计算当前GPU的直方图
                CalculateMassHist(device_momenta_list[i] + events_offsets_[i][1] * n_particles, particleToIndex, masshist, data_weights_for_gpu(i), temp, events_[i][1], n_particles);
                // 累加到总直方图
                for (size_t j = 0; j < masshist.size(); ++j)
                {
                    masshist_data[j]->Add(temp[j]);
                }
                // 清理临时直方图
                for (auto* hist : temp) delete hist;
            }
            for (size_t i = 0; i < masshist.size(); ++i)
            {
                TDirectory* histDir = rootFile->GetDirectory(masshist[i].name.c_str());
                histDir->cd();
                masshist_data[i]->Write("hdata", TObject::kOverwrite);
                delete masshist_data[i];
            }

            std::vector<TH1F*> anglehist_data;
            for (const auto& histConfig : anglehist)
            {
                TH1F* hist = new TH1F((histConfig.name + "_data").c_str(), histConfig.title.c_str(), histConfig.bins, histConfig.range[0], histConfig.range[1]);
                anglehist_data.emplace_back(hist);
            }
            // 临时直方图向量，用于每个GPU
            std::vector<std::vector<TH1F*>> temp_angle_hists(device_momenta_list.size());
            for (size_t i = 0; i < device_momenta_list.size(); ++i)
            {
                cudaSetDevice(i);
                // 为当前GPU创建临时直方图
                std::vector<TH1F*>& temp = temp_angle_hists[i];
                for (const auto& histConfig : anglehist)
                {
                    TH1F* hist = new TH1F((histConfig.name + "_temp").c_str(), histConfig.title.c_str(), histConfig.bins, histConfig.range[0], histConfig.range[1]);
                    temp.emplace_back(hist);
                }
                // 计算当前GPU的直方图
                CalculateAngleHist(device_momenta_list[i] + events_offsets_[i][1] * n_particles, particleToIndex, anglehist, data_weights_for_gpu(i), temp, events_[i][1], n_particles);
                // 累加到总直方图
                for (size_t j = 0; j < anglehist.size(); ++j)
                {
                    anglehist_data[j]->Add(temp[j]);
                }
                // 清理临时直方图
                for (auto* hist : temp) delete hist;
            }
            for (size_t i = 0; i < anglehist.size(); ++i)
            {
                TDirectory* histDir = rootFile->GetDirectory(anglehist[i].name.c_str());
                histDir->cd();
                anglehist_data[i]->Write("hdata", TObject::kOverwrite);
                delete anglehist_data[i];
            }

            std::vector<TH2F*> dalitzhist_data;
            for (const auto& histConfig : dalitzhist)
            {
                TH2F* hist = new TH2F((histConfig.name + "_data").c_str(), histConfig.title.c_str(),
                    histConfig.bins[0], histConfig.range[0][0], histConfig.range[0][1],
                    histConfig.bins[1], histConfig.range[1][0], histConfig.range[1][1]);
                dalitzhist_data.emplace_back(hist);
            }
            // 临时直方图向量，用于每个GPU
            std::vector<std::vector<TH2F*>> temp_dalitz_hists(device_momenta_list.size());
            for (size_t i = 0; i < device_momenta_list.size(); ++i)
            {
                cudaSetDevice(i);
                // 为当前GPU创建临时直方图
                std::vector<TH2F*>& temp = temp_dalitz_hists[i];
                for (const auto& histConfig : dalitzhist)
                {
                    TH2F* hist = new TH2F((histConfig.name + "_temp").c_str(), histConfig.title.c_str(),
                        histConfig.bins[0], histConfig.range[0][0], histConfig.range[0][1],
                        histConfig.bins[1], histConfig.range[1][0], histConfig.range[1][1]);
                    temp.emplace_back(hist);
                }
                // 计算当前GPU的直方图
                CalculateDalitzHist(device_momenta_list[i] + events_offsets_[i][1] * n_particles, particleToIndex, dalitzhist, data_weights_for_gpu(i), temp, events_[i][1], n_particles);
                // 累加到总直方图
                for (size_t j = 0; j < dalitzhist.size(); ++j)
                {
                    dalitzhist_data[j]->Add(temp[j]);
                }
                // 清理临时直方图
                for (auto* hist : temp) delete hist;
            }
            for (size_t i = 0; i < dalitzhist.size(); ++i)
            {
                TDirectory* histDir = rootFile->GetDirectory(dalitzhist[i].name.c_str());
                histDir->cd();
                dalitzhist_data[i]->Write("hdata", TObject::kOverwrite);
                delete dalitzhist_data[i];
            }

            // if (device_momenta != nullptr)
            // cudaFree(device_momenta); // device_momenta not allocated
        }

        // 计算并保存拟合结果直方图
        if (N_phsp > 0)
        {
            std::vector<TH1F*> masshist_fit;
            for (const auto& histConfig : masshist)
            {
                TH1F* hist = new TH1F((histConfig.name + "_fit").c_str(),
                    histConfig.title.c_str(), histConfig.bins,
                    histConfig.range[0], histConfig.range[1]);
                masshist_fit.emplace_back(hist);
            }

            // 临时直方图向量，用于每个GPU
            std::vector<std::vector<TH1F*>> temp_mass_hists(device_momenta_list.size());
            for (size_t i = 0; i < device_momenta_list.size(); ++i)
            {
                cudaSetDevice(i);
                // 为当前GPU创建临时直方图
                std::vector<TH1F*>& temp = temp_mass_hists[i];
                for (const auto& histConfig : masshist)
                {
                    TH1F* hist = new TH1F((histConfig.name + "_temp").c_str(),
                        histConfig.title.c_str(), histConfig.bins,
                        histConfig.range[0], histConfig.range[1]);
                    temp.emplace_back(hist);
                }
                // 计算当前GPU的直方图（phsp偏移量为0）
                CalculateMassHist(device_momenta_list[i], particleToIndex, masshist,
                    d_final_result_vec[i], temp, events_[i][0], n_particles);
                // 累加到总直方图
                for (size_t j = 0; j < masshist.size(); ++j)
                {
                    masshist_fit[j]->Add(temp[j]);
                }
                // 清理临时直方图
                for (auto* hist : temp) delete hist;
            }
            for (size_t i = 0; i < masshist.size(); ++i)
            {
                TDirectory* histDir = rootFile->GetDirectory(masshist[i].name.c_str());
                histDir->cd();

                TH1F* hist = masshist_fit[i];
                hist->Scale(normFactor);
                hist->Write("hfit", TObject::kOverwrite);
                delete masshist_fit[i];
            }

            std::vector<TH1F*> anglehist_fit;
            for (const auto& histConfig : anglehist)
            {
                TH1F* hist = new TH1F((histConfig.name + "_fit").c_str(),
                    histConfig.title.c_str(), histConfig.bins,
                    histConfig.range[0], histConfig.range[1]);
                anglehist_fit.emplace_back(hist);
            }
            // 临时直方图向量，用于每个GPU
            std::vector<std::vector<TH1F*>> temp_angle_hists(device_momenta_list.size());
            for (size_t i = 0; i < device_momenta_list.size(); ++i)
            {
                cudaSetDevice(i);
                // 为当前GPU创建临时直方图
                std::vector<TH1F*>& temp = temp_angle_hists[i];
                for (const auto& histConfig : anglehist)
                {
                    TH1F* hist = new TH1F((histConfig.name + "_temp").c_str(),
                        histConfig.title.c_str(), histConfig.bins,
                        histConfig.range[0], histConfig.range[1]);
                    temp.emplace_back(hist);
                }
                // 计算当前GPU的直方图
                CalculateAngleHist(device_momenta_list[i], particleToIndex, anglehist,
                    d_final_result_vec[i], temp, events_[i][0], n_particles);
                // 累加到总直方图
                for (size_t j = 0; j < anglehist.size(); ++j)
                {
                    anglehist_fit[j]->Add(temp[j]);
                }
                // 清理临时直方图
                for (auto* hist : temp) delete hist;
            }
            for (size_t i = 0; i < anglehist.size(); ++i)
            {
                TDirectory* histDir = rootFile->GetDirectory(anglehist[i].name.c_str());
                histDir->cd();
                TH1F* hist = anglehist_fit[i];
                hist->Scale(normFactor);
                hist->Write("hfit", TObject::kOverwrite);
                delete anglehist_fit[i];
            }

            std::vector<TH2F*> dalitzhist_fit;
            for (const auto& histConfig : dalitzhist)
            {
                TH2F* hist = new TH2F((histConfig.name + "_fit").c_str(), histConfig.title.c_str(),
                    histConfig.bins[0], histConfig.range[0][0],
                    histConfig.range[0][1], histConfig.bins[1],
                    histConfig.range[1][0], histConfig.range[1][1]);
                dalitzhist_fit.emplace_back(hist);
            }
            // 临时直方图向量，用于每个GPU
            std::vector<std::vector<TH2F*>> temp_dalitz_hists(device_momenta_list.size());
            for (size_t i = 0; i < device_momenta_list.size(); ++i)
            {
                cudaSetDevice(i);
                // 为当前GPU创建临时直方图
                std::vector<TH2F*>& temp = temp_dalitz_hists[i];
                for (const auto& histConfig : dalitzhist)
                {
                    TH2F* hist = new TH2F((histConfig.name + "_temp").c_str(), histConfig.title.c_str(),
                        histConfig.bins[0], histConfig.range[0][0],
                        histConfig.range[0][1], histConfig.bins[1],
                        histConfig.range[1][0], histConfig.range[1][1]);
                    temp.emplace_back(hist);
                }
                // 计算当前GPU的直方图
                CalculateDalitzHist(device_momenta_list[i], particleToIndex, dalitzhist,
                    d_final_result_vec[i], temp, events_[i][0], n_particles);
                // 累加到总直方图
                for (size_t j = 0; j < dalitzhist.size(); ++j)
                {
                    dalitzhist_fit[j]->Add(temp[j]);
                }
                // 清理临时直方图
                for (auto* hist : temp) delete hist;
            }
            for (size_t i = 0; i < dalitzhist.size(); ++i)
            {
                TDirectory* histDir = rootFile->GetDirectory(dalitzhist[i].name.c_str());
                histDir->cd();
                TH2F* hist = dalitzhist_fit[i];
                hist->Scale(normFactor);
                hist->Write("hfit", TObject::kOverwrite);
                delete dalitzhist_fit[i];
            }
        }

        // if (N_bkg > 0 && !bkg_weights_.empty() && bkg_weights_[0] != nullptr)
        if (N_bkg > 0)
        {
            std::vector<TH1F*> masshist_bkg;
            for (const auto& histConfig : masshist)
            {
                TH1F* hist = new TH1F((histConfig.name + "_bkg").c_str(),
                    histConfig.title.c_str(), histConfig.bins,
                    histConfig.range[0], histConfig.range[1]);
                masshist_bkg.emplace_back(hist);
            }

            // 临时直方图向量，用于每个GPU
            std::vector<std::vector<TH1F*>> temp_mass_hists(device_momenta_list.size());
            // 计算权重偏移量
            int bkg_weight_offset = 0;
            for (size_t i = 0; i < device_momenta_list.size(); ++i)
            {
                if (events_[i][2] == 0) continue;  // 该GPU无bkg事件

                cudaSetDevice(i);
                // 为当前GPU创建临时直方图
                std::vector<TH1F*>& temp = temp_mass_hists[i];
                for (const auto& histConfig : masshist)
                {
                    TH1F* hist = new TH1F((histConfig.name + "_temp").c_str(),
                        histConfig.title.c_str(), histConfig.bins,
                        histConfig.range[0], histConfig.range[1]);
                    temp.emplace_back(hist);
                }
                CalculateMassHist(device_momenta_list[i] + events_offsets_[i][2] * n_particles,
                    particleToIndex, masshist,
                    bkg_weights_[i], temp, events_[i][2], n_particles);
                // 累加到总直方图
                for (size_t j = 0; j < masshist.size(); ++j)
                {
                    masshist_bkg[j]->Add(temp[j]);
                }
                // 清理临时直方图
                for (auto* hist : temp) delete hist;

                // 更新权重偏移量
                bkg_weight_offset += events_[i][2];
            }
            for (size_t i = 0; i < masshist.size(); ++i)
            {
                TDirectory* histDir = rootFile->GetDirectory(masshist[i].name.c_str());
                histDir->cd();
                masshist_bkg[i]->Write("hbkg", TObject::kOverwrite);
                delete masshist_bkg[i];
            }

            std::vector<TH1F*> anglehist_bkg;
            for (const auto& histConfig : anglehist)
            {
                TH1F* hist = new TH1F((histConfig.name + "_bkg").c_str(),
                    histConfig.title.c_str(), histConfig.bins,
                    histConfig.range[0], histConfig.range[1]);
                anglehist_bkg.emplace_back(hist);
            }
            // 临时直方图向量，用于每个GPU
            std::vector<std::vector<TH1F*>> temp_angle_hists(device_momenta_list.size());
            bkg_weight_offset = 0;
            for (size_t i = 0; i < device_momenta_list.size(); ++i)
            {
                if (events_[i][2] == 0) continue;

                cudaSetDevice(i);
                // 为当前GPU创建临时直方图
                std::vector<TH1F*>& temp = temp_angle_hists[i];
                for (const auto& histConfig : anglehist)
                {
                    TH1F* hist = new TH1F((histConfig.name + "_temp").c_str(),
                        histConfig.title.c_str(), histConfig.bins,
                        histConfig.range[0], histConfig.range[1]);
                    temp.emplace_back(hist);
                }
                // 计算当前GPU的直方图
                CalculateAngleHist(device_momenta_list[i] + events_offsets_[i][2] * n_particles,
                    particleToIndex, anglehist,
                    bkg_weights_[i], temp, events_[i][2], n_particles);
                // 累加到总直方图
                for (size_t j = 0; j < anglehist.size(); ++j)
                {
                    anglehist_bkg[j]->Add(temp[j]);
                }
                // 清理临时直方图
                for (auto* hist : temp) delete hist;

                bkg_weight_offset += events_[i][2];
            }
            for (size_t i = 0; i < anglehist.size(); ++i)
            {
                TDirectory* histDir = rootFile->GetDirectory(anglehist[i].name.c_str());
                histDir->cd();
                anglehist_bkg[i]->Write("hbkg", TObject::kOverwrite);
                delete anglehist_bkg[i];
            }

            std::vector<TH2F*> dalitzhist_bkg;
            for (const auto& histConfig : dalitzhist)
            {
                TH2F* hist = new TH2F((histConfig.name + "_bkg").c_str(), histConfig.title.c_str(),
                    histConfig.bins[0], histConfig.range[0][0],
                    histConfig.range[0][1], histConfig.bins[1],
                    histConfig.range[1][0], histConfig.range[1][1]);
                dalitzhist_bkg.emplace_back(hist);
            }
            // 临时直方图向量，用于每个GPU
            std::vector<std::vector<TH2F*>> temp_dalitz_hists(device_momenta_list.size());
            bkg_weight_offset = 0;
            for (size_t i = 0; i < device_momenta_list.size(); ++i)
            {
                if (events_[i][2] == 0) continue;

                cudaSetDevice(i);
                // 为当前GPU创建临时直方图
                std::vector<TH2F*>& temp = temp_dalitz_hists[i];
                for (const auto& histConfig : dalitzhist)
                {
                    TH2F* hist = new TH2F((histConfig.name + "_temp").c_str(), histConfig.title.c_str(),
                        histConfig.bins[0], histConfig.range[0][0],
                        histConfig.range[0][1], histConfig.bins[1],
                        histConfig.range[1][0], histConfig.range[1][1]);
                    temp.emplace_back(hist);
                }
                // 计算当前GPU的直方图
                // double* bkg_weight_ptr = nullptr;//bkg_weights_[i] + bkg_weight_offset;
                CalculateDalitzHist(device_momenta_list[i] + events_offsets_[i][2] * n_particles,
                    particleToIndex, dalitzhist,
                    bkg_weights_[i], temp, events_[i][2], n_particles);
                // 累加到总直方图
                for (size_t j = 0; j < dalitzhist.size(); ++j)
                {
                    dalitzhist_bkg[j]->Add(temp[j]);
                }
                // 清理临时直方图
                for (auto* hist : temp) delete hist;

                bkg_weight_offset += events_[i][2];
            }
            for (size_t i = 0; i < dalitzhist.size(); ++i)
            {
                TDirectory* histDir = rootFile->GetDirectory(dalitzhist[i].name.c_str());
                histDir->cd();
                dalitzhist_bkg[i]->Write("hbkg", TObject::kOverwrite);
                delete dalitzhist_bkg[i];
            }
        }

        // 计算并保存部分波直方图（多GPU版本）
        if (N_phsp > 0)
        {
            int npartials = nSLvectors_.size();
            for (int partial_idx = 0; partial_idx < npartials; ++partial_idx)
            {
                // 为当前部分波创建质量直方图
                std::vector<TH1F*> masshist_partial;
                for (const auto& histConfig : masshist)
                {
                    TH1F* hist = new TH1F((histConfig.name + "_partial").c_str(),
                        histConfig.title.c_str(), histConfig.bins,
                        histConfig.range[0], histConfig.range[1]);
                    masshist_partial.push_back(hist);
                }

                // 临时直方图向量，用于每个GPU
                std::vector<std::vector<TH1F*>> temp_mass_hists(device_momenta_list.size());
                for (size_t gpu = 0; gpu < device_momenta_list.size(); ++gpu)
                {
                    if (events_[gpu][0] == 0) continue;  // 该GPU无phsp事件

                    cudaSetDevice(gpu);
                    // 为当前GPU创建临时直方图
                    std::vector<TH1F*>& temp = temp_mass_hists[gpu];
                    for (const auto& histConfig : masshist)
                    {
                        TH1F* hist = new TH1F((histConfig.name + "_temp").c_str(),
                            histConfig.title.c_str(), histConfig.bins,
                            histConfig.range[0], histConfig.range[1]);
                        temp.emplace_back(hist);
                    }
                    // 计算当前GPU的部分波直方图
                    // 权重指针：d_partial_result_vec[gpu][partial_idx * events_[gpu][0]]
                    double* partial_weight_ptr = d_partial_result_vec[gpu] + partial_idx * events_[gpu][0];
                    CalculateMassHist(device_momenta_list[gpu], particleToIndex, masshist,
                        partial_weight_ptr, temp, events_[gpu][0], n_particles);
                    // 累加到总直方图
                    for (size_t j = 0; j < masshist.size(); ++j)
                    {
                        masshist_partial[j]->Add(temp[j]);
                    }
                    // 清理临时直方图
                    for (auto* hist : temp) delete hist;
                }

                for (size_t j = 0; j < masshist_partial.size(); ++j)
                {
                    TDirectory* histDir = rootFile->GetDirectory(masshist[j].name.c_str());
                    histDir->cd();

                    std::string partial_dir_name = "h_" + sanitizeROOTName (resonance_names_[partial_idx]);

                    TH1F* hist = masshist_partial[j];
                    hist->Scale(normFactor);
                    hist->Write(partial_dir_name.c_str(), TObject::kOverwrite);
                    delete hist;
                }

                // 角度直方图
                std::vector<TH1F*> anglehist_partial;
                for (const auto& histConfig : anglehist)
                {
                    TH1F* hist = new TH1F((histConfig.name + "_partial").c_str(),
                        histConfig.title.c_str(), histConfig.bins,
                        histConfig.range[0], histConfig.range[1]);
                    anglehist_partial.push_back(hist);
                }

                // 临时直方图向量，用于每个GPU
                std::vector<std::vector<TH1F*>> temp_angle_hists(device_momenta_list.size());
                for (size_t gpu = 0; gpu < device_momenta_list.size(); ++gpu)
                {
                    if (events_[gpu][0] == 0) continue;

                    cudaSetDevice(gpu);
                    std::vector<TH1F*>& temp = temp_angle_hists[gpu];
                    for (const auto& histConfig : anglehist)
                    {
                        TH1F* hist = new TH1F((histConfig.name + "_temp").c_str(),
                            histConfig.title.c_str(), histConfig.bins,
                            histConfig.range[0], histConfig.range[1]);
                        temp.emplace_back(hist);
                    }
                    double* partial_weight_ptr = d_partial_result_vec[gpu] + partial_idx * events_[gpu][0];
                    CalculateAngleHist(device_momenta_list[gpu], particleToIndex, anglehist,
                        partial_weight_ptr, temp, events_[gpu][0], n_particles);
                    for (size_t j = 0; j < anglehist.size(); ++j)
                    {
                        anglehist_partial[j]->Add(temp[j]);
                    }
                    for (auto* hist : temp) delete hist;
                }

                for (size_t j = 0; j < anglehist_partial.size(); ++j)
                {
                    TDirectory* histDir = rootFile->GetDirectory(anglehist[j].name.c_str());
                    histDir->cd();

                    std::string partial_dir_name = "h_" + sanitizeROOTName(resonance_names_[partial_idx]);

                    TH1F* hist = anglehist_partial[j];
                    hist->Scale(normFactor);
                    hist->Write(partial_dir_name.c_str(), TObject::kOverwrite);
                    delete anglehist_partial[j];
                }
            }
        }

        // =====================================================================
        // 统一观测直方图 (Plot 序列 form, type=="obs"): 1d/2d 自动区分
        //   数据 hdata: data_weights_（无权重时按 1）;  模型 hfit: phsp 权重
        //   d_final_result_vec (全模型或 waves 子集) × normFactor
        // =====================================================================
        {
            std::vector<PlotConfig> obshists;
            for (const auto& pc : plotconfig)
                if (pc.type == "obs") obshists.push_back(pc);
            if (!obshists.empty()) {
                // 防御: 过滤 bins/ranges 维度不足的项(正常解析不会发生; 兜底手写/旧 .so)
                std::vector<PlotConfig> valid;
                for (const auto& cfg : obshists) {
                    bool ok = (int)cfg.bins.size() >= (int)cfg.obs.size()
                           && (int)cfg.ranges.size() >= (int)cfg.obs.size();
                    for (int d = 0; ok && d < (int)cfg.obs.size(); ++d)
                        if (cfg.ranges[d].size() < 2) ok = false;
                    if (ok) valid.push_back(cfg);
                    else std::cerr << "Warning: 跳过格式不完整的 obs 项 '"
                                   << cfg.name << "' (bins/ranges 维度不足)" << std::endl;
                }
                obshists.swap(valid);
                // 顶层母粒子 index（Angle/CosAngle 的轴; 取第一个链的第一步母粒子）
                int motherIdx = -1;
                {
                    const auto& chains = config_parser_.getDecayChains();
                    if (!chains.empty() && !chains[0].decay_steps.empty()) {
                        auto it = particleToIndex.find(chains[0].decay_steps[0].mother);
                        if (it != particleToIndex.end()) motherIdx = it->second;
                    }
                }
                for (const auto& cfg : obshists) {
                    TDirectory* dir = rootFile->mkdir(cfg.name.c_str());
                    dir->cd();
                    if (cfg.display.size() >= 2) {
                        TObjString xl(cfg.display[0].c_str());
                        TObjString yl(cfg.display[1].c_str());
                        xl.Write("xlabel", TObject::kOverwrite);
                        yl.Write("ylabel", TObject::kOverwrite);
                    }
                }

                // ---- 数据直方图 ----
                std::vector<TH1F*> obs1d_data(obshists.size(), nullptr);
                std::vector<TH2F*> obs2d_data(obshists.size(), nullptr);
                for (size_t j = 0; j < obshists.size(); ++j) {
                    const auto& cfg = obshists[j];
                    if (cfg.obs.size() == 1)
                        obs1d_data[j] = new TH1F((cfg.name + "_data").c_str(), "",
                            cfg.bins[0], cfg.ranges[0][0], cfg.ranges[0][1]);
                    else
                        obs2d_data[j] = new TH2F((cfg.name + "_data").c_str(), "",
                            cfg.bins[0], cfg.ranges[0][0], cfg.ranges[0][1],
                            cfg.bins[1], cfg.ranges[1][0], cfg.ranges[1][1]);
                }
                for (size_t gpu = 0; gpu < device_momenta_list.size(); ++gpu) {
                    if (events_[gpu][1] == 0) continue;
                    cudaSetDevice(gpu);
                    std::vector<TH1F*> t1(obshists.size(), nullptr);
                    std::vector<TH2F*> t2(obshists.size(), nullptr);
                    for (size_t j = 0; j < obshists.size(); ++j) {
                        const auto& cfg = obshists[j];
                        if (cfg.obs.size() == 1)
                            t1[j] = new TH1F((cfg.name + "_t").c_str(), "",
                                cfg.bins[0], cfg.ranges[0][0], cfg.ranges[0][1]);
                        else
                            t2[j] = new TH2F((cfg.name + "_t").c_str(), "",
                                cfg.bins[0], cfg.ranges[0][0], cfg.ranges[0][1],
                                cfg.bins[1], cfg.ranges[1][0], cfg.ranges[1][1]);
                    }
                    CalculateObsHist(
                        device_momenta_list[gpu] + events_offsets_[gpu][1] * n_particles,
                        particleToIndex, obshists, data_weights_for_gpu(gpu), t1, t2,
                        events_[gpu][1], n_particles, motherIdx);
                    for (size_t j = 0; j < obshists.size(); ++j) {
                        if (obs1d_data[j]) obs1d_data[j]->Add(t1[j]);
                        if (obs2d_data[j]) obs2d_data[j]->Add(t2[j]);
                    }
                    for (auto* h : t1) if (h) delete h;
                    for (auto* h : t2) if (h) delete h;
                }
                for (size_t j = 0; j < obshists.size(); ++j) {
                    TDirectory* dir = rootFile->GetDirectory(obshists[j].name.c_str());
                    dir->cd();
                    if (obs1d_data[j]) {
                        obs1d_data[j]->Write("hdata", TObject::kOverwrite);
                        delete obs1d_data[j];
                    }
                    if (obs2d_data[j]) {
                        obs2d_data[j]->Write("hdata", TObject::kOverwrite);
                        delete obs2d_data[j];
                    }
                }

                // ---- 本底直方图 hbkg (与旧格式一致: bkg 事件 × bkg_weights_, 不 Scale) ----
                if (N_bkg > 0) {
                    std::vector<TH1F*> obs1d_bkg(obshists.size(), nullptr);
                    std::vector<TH2F*> obs2d_bkg(obshists.size(), nullptr);
                    for (size_t j = 0; j < obshists.size(); ++j) {
                        const auto& cfg = obshists[j];
                        if (cfg.obs.size() == 1)
                            obs1d_bkg[j] = new TH1F((cfg.name + "_bkg").c_str(), "",
                                cfg.bins[0], cfg.ranges[0][0], cfg.ranges[0][1]);
                        else
                            obs2d_bkg[j] = new TH2F((cfg.name + "_bkg").c_str(), "",
                                cfg.bins[0], cfg.ranges[0][0], cfg.ranges[0][1],
                                cfg.bins[1], cfg.ranges[1][0], cfg.ranges[1][1]);
                    }
                    for (size_t gpu = 0; gpu < device_momenta_list.size(); ++gpu) {
                        if (events_[gpu][2] == 0) continue;
                        cudaSetDevice(gpu);
                        std::vector<TH1F*> t1(obshists.size(), nullptr);
                        std::vector<TH2F*> t2(obshists.size(), nullptr);
                        for (size_t j = 0; j < obshists.size(); ++j) {
                            const auto& cfg = obshists[j];
                            if (cfg.obs.size() == 1)
                                t1[j] = new TH1F((cfg.name + "_t").c_str(), "",
                                    cfg.bins[0], cfg.ranges[0][0], cfg.ranges[0][1]);
                            else
                                t2[j] = new TH2F((cfg.name + "_t").c_str(), "",
                                    cfg.bins[0], cfg.ranges[0][0], cfg.ranges[0][1],
                                    cfg.bins[1], cfg.ranges[1][0], cfg.ranges[1][1]);
                        }
                        CalculateObsHist(
                            device_momenta_list[gpu] + events_offsets_[gpu][2] * n_particles,
                            particleToIndex, obshists, bkg_weights_[gpu], t1, t2,
                            events_[gpu][2], n_particles, motherIdx);
                        for (size_t j = 0; j < obshists.size(); ++j) {
                            if (obs1d_bkg[j]) obs1d_bkg[j]->Add(t1[j]);
                            if (obs2d_bkg[j]) obs2d_bkg[j]->Add(t2[j]);
                        }
                        for (auto* h : t1) if (h) delete h;
                        for (auto* h : t2) if (h) delete h;
                    }
                    for (size_t j = 0; j < obshists.size(); ++j) {
                        TDirectory* dir = rootFile->GetDirectory(obshists[j].name.c_str());
                        dir->cd();
                        if (obs1d_bkg[j]) {
                            obs1d_bkg[j]->Write("hbkg", TObject::kOverwrite);
                            delete obs1d_bkg[j];
                        }
                        if (obs2d_bkg[j]) {
                            obs2d_bkg[j]->Write("hbkg", TObject::kOverwrite);
                            delete obs2d_bkg[j];
                        }
                    }
                }

                // ---- 模型直方图 ----
                std::vector<TH1F*> obs1d_fit(obshists.size(), nullptr);
                std::vector<TH2F*> obs2d_fit(obshists.size(), nullptr);
                for (size_t j = 0; j < obshists.size(); ++j) {
                    const auto& cfg = obshists[j];
                    if (cfg.obs.size() == 1)
                        obs1d_fit[j] = new TH1F((cfg.name + "_fit").c_str(), "",
                            cfg.bins[0], cfg.ranges[0][0], cfg.ranges[0][1]);
                    else
                        obs2d_fit[j] = new TH2F((cfg.name + "_fit").c_str(), "",
                            cfg.bins[0], cfg.ranges[0][0], cfg.ranges[0][1],
                            cfg.bins[1], cfg.ranges[1][0], cfg.ranges[1][1]);
                }
                for (size_t gpu = 0; gpu < device_momenta_list.size(); ++gpu) {
                    if (events_[gpu][0] == 0) continue;
                    cudaSetDevice(gpu);
                    std::vector<TH1F*> t1(obshists.size(), nullptr);
                    std::vector<TH2F*> t2(obshists.size(), nullptr);
                    for (size_t j = 0; j < obshists.size(); ++j) {
                        const auto& cfg = obshists[j];
                        if (cfg.obs.size() == 1)
                            t1[j] = new TH1F((cfg.name + "_t").c_str(), "",
                                cfg.bins[0], cfg.ranges[0][0], cfg.ranges[0][1]);
                        else
                            t2[j] = new TH2F((cfg.name + "_t").c_str(), "",
                                cfg.bins[0], cfg.ranges[0][0], cfg.ranges[0][1],
                                cfg.bins[1], cfg.ranges[1][0], cfg.ranges[1][1]);
                    }
                    CalculateObsHist(device_momenta_list[gpu], particleToIndex,
                        obshists, d_final_result_vec[gpu], t1, t2,
                        events_[gpu][0], n_particles, motherIdx);
                    for (size_t j = 0; j < obshists.size(); ++j) {
                        if (obs1d_fit[j]) obs1d_fit[j]->Add(t1[j]);
                        if (obs2d_fit[j]) obs2d_fit[j]->Add(t2[j]);
                    }
                    for (auto* h : t1) if (h) delete h;
                    for (auto* h : t2) if (h) delete h;
                }
                for (size_t j = 0; j < obshists.size(); ++j) {
                    TDirectory* dir = rootFile->GetDirectory(obshists[j].name.c_str());
                    dir->cd();
                    if (obs1d_fit[j]) {
                        obs1d_fit[j]->Scale(normFactor);
                        obs1d_fit[j]->Write("hfit", TObject::kOverwrite);
                        delete obs1d_fit[j];
                    }
                    if (obs2d_fit[j]) {
                        obs2d_fit[j]->Scale(normFactor);
                        obs2d_fit[j]->Write("hfit", TObject::kOverwrite);
                        delete obs2d_fit[j];
                    }
                }

                // ---- 分波形状: 每个 obs 图 × 每波 |A_i|² 权重 → h_<波名> 子目录 ----
                // (与旧格式的 mass/angle 分波形状一致; 1d 用 TH1F, 2d 用 TH2F)
                for (size_t j = 0; j < obshists.size(); ++j) {
                    const auto& cfg = obshists[j];
                    const bool is2 = (cfg.obs.size() == 2);
                    std::vector<TH1F*> obs1d_part(npartials, nullptr);
                    std::vector<TH2F*> obs2d_part(npartials, nullptr);
                    for (int p = 0; p < npartials; ++p) {
                        std::string pn = cfg.name + "_p" + std::to_string(p);
                        if (is2)
                            obs2d_part[p] = new TH2F(pn.c_str(), "",
                                cfg.bins[0], cfg.ranges[0][0], cfg.ranges[0][1],
                                cfg.bins[1], cfg.ranges[1][0], cfg.ranges[1][1]);
                        else
                            obs1d_part[p] = new TH1F(pn.c_str(), "",
                                cfg.bins[0], cfg.ranges[0][0], cfg.ranges[0][1]);
                    }
                    for (size_t gpu = 0; gpu < device_momenta_list.size(); ++gpu) {
                        if (events_[gpu][0] == 0) continue;
                        cudaSetDevice(gpu);
                        for (int p = 0; p < npartials; ++p) {
                            std::vector<PlotConfig> one = {cfg};
                            std::vector<TH1F*> t1(is2 ? 0 : 1, nullptr);
                            std::vector<TH2F*> t2(is2 ? 1 : 0, nullptr);
                            std::string tn = cfg.name + "_t" + std::to_string(p);
                            if (is2)
                                t2[0] = new TH2F(tn.c_str(), "",
                                    cfg.bins[0], cfg.ranges[0][0], cfg.ranges[0][1],
                                    cfg.bins[1], cfg.ranges[1][0], cfg.ranges[1][1]);
                            else
                                t1[0] = new TH1F(tn.c_str(), "",
                                    cfg.bins[0], cfg.ranges[0][0], cfg.ranges[0][1]);
                            double* partial_weight_ptr =
                                d_partial_result_vec[gpu] + (size_t)p * events_[gpu][0];
                            CalculateObsHist(device_momenta_list[gpu], particleToIndex,
                                one, partial_weight_ptr, t1, t2,
                                events_[gpu][0], n_particles, motherIdx);
                            if (is2) obs2d_part[p]->Add(t2[0]);
                            else obs1d_part[p]->Add(t1[0]);
                            if (!t1.empty() && t1[0]) delete t1[0];
                            if (!t2.empty() && t2[0]) delete t2[0];
                        }
                    }
                    for (int p = 0; p < npartials; ++p) {
                        std::string partial_dir_name =
                            "h_" + sanitizeROOTName(resonance_names_[p]);
                        TDirectory* dir = rootFile->GetDirectory(cfg.name.c_str());
                        dir->cd();
                        if (is2) {
                            if (!obs2d_part[p]) continue;
                            obs2d_part[p]->Scale(normFactor);
                            obs2d_part[p]->Write(partial_dir_name.c_str(),
                                TObject::kOverwrite);
                            delete obs2d_part[p];
                        } else {
                            if (!obs1d_part[p]) continue;
                            obs1d_part[p]->Scale(normFactor);
                            obs1d_part[p]->Write(partial_dir_name.c_str(),
                                TObject::kOverwrite);
                            delete obs1d_part[p];
                        }
                    }
                }
            }
        }

        // 直方图都写完之后，先释放逐事例设备缓冲，再跑矩阵/误差的全样本扫描。
        // 大统计量下这些缓冲非常占显存（每卡）：
        //   d_partial_result_vec = N_phsp × npartials × 8 B   （15M×20 → 2.4 GB）
        //   d_final_result_vec   = N_phsp × 8 B
        //   device_momenta_list  = N_phsp × n_final × sizeof(LorentzVector)
        // 不先放掉，后面每张矩阵/每个误差方向都要再分配批 workspace → 多卡 OOM。
        for (size_t i = 0; i < d_final_result_vec.size(); ++i)
            if (d_final_result_vec[i]) {
                cudaSetDevice((int)i);
                cudaFree(d_final_result_vec[i]);
                d_final_result_vec[i] = nullptr;
            }
        for (size_t i = 0; i < d_partial_result_vec.size(); ++i)
            if (d_partial_result_vec[i]) {
                cudaSetDevice((int)i);
                cudaFree(d_partial_result_vec[i]);
                d_partial_result_vec[i] = nullptr;
            }
        for (size_t i = 0; i < device_momenta_list.size(); ++i)
            if (device_momenta_list[i]) {
                cudaSetDevice((int)i);
                cudaFree(device_momenta_list[i]);
                device_momenta_list[i] = nullptr;
            }
        cudaSetDevice(target_dev);


        // =====================================================================
        // 两张干涉矩阵（各自除以本样本的相干总额 W，因此元素是 O(1) 的份额）:
        //   interference_rec : P_ij / W_rec   （重建侧，带效率 MC）
        //   interference_gen : T_ij / W_gen   （产生侧 truth / 无效率 MC）
        // 定义（权重口径与拟合 NLL 一致，phsp_weights / phsp_truth_weights 生效）:
        //   M_ii = Σ_e w·|A_i|² ,  M_ij = Σ_e w·2Re(A_iA_j*)  (i≠j, 对称)
        //   W    = Σ_i M_ii + Σ_{i<j} M_ij  = Σ_e w·|Σ_i A_i|²
        // 于是（与 doc/branching_fraction_recipe.md 一致）:
        //   文献拟合分数 F_i = interference_gen[i][i]，干涉项 I_ij = interference_gen[i][j]
        //   非相干份额   FF_i = diag / Σ diag           （本库 getFitFractions 的口径）
        //   ε̄ = W_rec / W_gen（两样本同源产生时）;  ε_i = (P_ii/N_rec)/(T_ii/N_gen)
        //   —— 需要"原始积分"时把矩阵乘回对应的 W（integrals 树里有 W_rec/W_gen）。
        // integrals 树给出 W_rec, W_gen, N_rec, N_gen（Σw）, npartials。
        // 矩阵行列顺序 = 分波顺序（与 getFitFractions / writeResult 的 h_<n> 一致）。
        // CTPWA_WR_NO_INTEGRALS=1 可跳过这两趟全样本扫描（不写矩阵与 integrals）。
        // =====================================================================
        const auto& wr_data_files = config_parser_.getDataFiles();
        const bool want_raw = (std::getenv("CTPWA_WR_NO_INTEGRALS") == nullptr);
        double N_rec_w = 0.0, N_gen_w = 0.0, W_rec_raw = 0.0, W_gen_raw = 0.0;
        std::vector<double> P_diag, P_off, T_diag, T_off;
        auto gramTotal = [npartials](const std::vector<double>& diag,
                                     const std::vector<double>& off) {
            double w = 0.0;
            for (int i = 0; i < npartials; ++i) w += diag[i];
            for (int i = 0; i < npartials; ++i)
                for (int j = i + 1; j < npartials; ++j) w += off[i * npartials + j];
            return w;
        };
        auto runIntegrals = [&](const std::string& key, std::vector<double>& diag,
                                std::vector<double>& off) -> double {
            diag.assign(npartials, 0.0);
            off.assign((size_t)npartials * npartials, 0.0);
            auto t0 = std::chrono::high_resolution_clock::now();
            double wsum = computeFileBatchIntegrals(key, extended_vector,
                std::vector<torch::Tensor>{}, std::vector<torch::Tensor>{},
                diag.data(), off.data(), nullptr,
                nullptr, nullptr, nullptr, nullptr, npartials);
            auto t1 = std::chrono::high_resolution_clock::now();
            if (std::getenv("CTPWA_VERBOSE"))
                std::printf("%s 积分矩阵: %.1f ms\n", key.c_str(),
                    std::chrono::duration<double, std::milli>(t1 - t0).count());
            return wsum;
        };
        if (want_raw && wr_data_files.count("phsp")) {
            {
                // 重建侧：常驻 B̄ 收缩（不读文件）；Mg 只有 n×n，取完 diag/off 立即析构
                double Wg = 0.0;
                torch::Tensor Mg = phspMatrixFromGram(extended_vector, &Wg, &N_rec_w);
                P_diag.assign(npartials, 0.0);
                P_off.assign((size_t)npartials * npartials, 0.0);
                for (int i = 0; i < npartials; ++i) {
                    P_diag[i] = Mg[i][i].item<double>();
                    for (int j = i + 1; j < npartials; ++j)
                        P_off[(size_t)i * npartials + j] = Mg[i][j].item<double>();
                }
            }

            W_rec_raw = gramTotal(P_diag, P_off);
        }
        bool have_truth = false;
        if (want_raw && wr_data_files.count("phsp_truth")) {
            N_gen_w = runIntegrals("phsp_truth", T_diag, T_off);
            W_gen_raw = gramTotal(T_diag, T_off);
            have_truth = true;
        }

        // 创建 ROOT 文件

        // 对称矩阵写出（对角用 partial，非对角用 2Re；两三角都填）
        auto writeSymmetric = [&](const char* name, const std::vector<double>& diag,
                                  const std::vector<double>& off, double scale) {
            if ((int)diag.size() != npartials) return;
            rootFile->cd();
            TMatrixD m(npartials, npartials);
            for (int i = 0; i < npartials; ++i) {
                m[i][i] = diag[i] / scale;
                for (int j = i + 1; j < npartials; ++j) {
                    double v = off[(size_t)i * npartials + j] / scale;
                    m[i][j] = v;
                    m[j][i] = v;
                }
            }
            m.Write(name);
        };

        if (!P_diag.empty() && W_rec_raw > 0.0)
            writeSymmetric("interference_rec", P_diag, P_off, W_rec_raw);
        if (have_truth && W_gen_raw > 0.0)
            writeSymmetric("interference_gen", T_diag, T_off, W_gen_raw);

        // ---- 参数误差矩阵（线性化传播；MC 统计误差不在这里，属系统项）----
        // 需要统一参数 [Re p, Im p, θ]（legacy complex 模式没有 θ 布局，跳过）。
        if (!P_diag.empty() && params_.hasCouplingMatrix()
            && params.dtype() == torch::kFloat64
            && std::getenv("CTPWA_WR_NO_ERR") == nullptr)
        {
            auto t0 = std::chrono::high_resolution_clock::now();
            auto errs = computeMatrixErrors(params, hessian_in);
            bool wrote_err = false;
            if (errs.size() == 2) {
                auto writeTensor = [&](const char* name, const torch::Tensor& E) {
                    rootFile->cd();
                    TMatrixD m(npartials, npartials);
                    for (int i = 0; i < npartials; ++i)
                        for (int j = 0; j < npartials; ++j)
                            m[i][j] = E[i][j].item<double>();
                    m.Write(name);
                };
                // rec/gen 独立写：空 phsp 时 rec_err 为空但 gen_err 仍有效。
                if (errs[0].numel() > 0) {
                    writeTensor("interference_rec_err", errs[0]);
                    wrote_err = true;
                }
                if (errs[1].numel() > 0) {
                    writeTensor("interference_gen_err", errs[1]);
                    wrote_err = true;
                }
            }
            if (wrote_err) {
                auto t1 = std::chrono::high_resolution_clock::now();
                if (std::getenv("CTPWA_VERBOSE"))
                    std::printf("参数误差矩阵: %.1f ms\n",
                        std::chrono::duration<double, std::milli>(t1 - t0).count());
            } else {
                std::cerr << "参数误差矩阵未写出（Hessian 非正定/缺失，或对应样本为空）"
                          << std::endl;
            }
        }

        // 归一化/换算要用的样本量（Σw）与相干总额（乘回去就是原始积分）
        if (!P_diag.empty() || have_truth) {
            rootFile->cd();
            TTree* t = new TTree("integrals",
                "sum of weights and coherent totals of the interference matrices");
            t->Branch("N_rec", &N_rec_w, "N_rec/D");
            t->Branch("W_rec", &W_rec_raw, "W_rec/D");
            t->Branch("N_gen", &N_gen_w, "N_gen/D");
            t->Branch("W_gen", &W_gen_raw, "W_gen/D");
            t->Branch("npartials", &npartials, "npartials/I");
            t->Fill();
            t->Write();
            delete t;
        }
        // 关闭 ROOT 文件（root_guard 兜底异常路径）
        rootFile->Close();
        delete rootFile;
        rootFile = nullptr;

        // std::cout << "Data written to ROOT file: " << filename << std::endl;

        // 释放设备内存
        // 释放每个GPU的final_result和partial_result
        for (size_t i = 0; i < d_final_result_vec.size(); ++i) {
            if (d_final_result_vec[i] != nullptr) {
                cudaSetDevice(i);
                cudaFree(d_final_result_vec[i]);
            }
        }
        for (size_t i = 0; i < d_partial_result_vec.size(); ++i) {
            if (d_partial_result_vec[i] != nullptr) {
                cudaSetDevice(i);
                cudaFree(d_partial_result_vec[i]);
            }
        }
        // 释放四动量内存
        for (size_t i = 0; i < device_momenta_list.size(); ++i) {
            if (device_momenta_list[i] != nullptr) {
                cudaSetDevice(i);
                cudaFree(device_momenta_list[i]);
            }
        }
        cudaSetDevice(target_dev);
        // h_total_results / h_partial_results 由 h_guard 统一释放（含异常路径）。
    
}

void analysis::writeInterfResult(torch::Tensor params, const std::string& filename,
                           const std::vector<std::vector<int>>& pairs)
{
        TORCH_CHECK(ctpwa_is_cuda_like(params), "params 必须位于当前后端的设备上（CPU 构建用 CPU 张量）");
        TORCH_CHECK(!pairs.empty(), "pairs 不能为空");
        ctpwa::ensure_device(primary_dev_);
        auto wi_ev0 = events_offsets_, wi_amp0 = amp_offsets_;
        auto wi_guard = makeScopeExit([this, wi_ev0, wi_amp0]() {
            events_offsets_ = wi_ev0;
            amp_offsets_ = wi_amp0;
            ctpwa::ensure_device(primary_dev_);
        });
        int npartials = (int)nSLvectors_.size();
        std::vector<int> pair_list;
        std::vector<std::pair<int, int>> pair_named;
        for (const auto& pr : pairs) {
            TORCH_CHECK(pr.size() == 2, "每个 pair 需要 [i, j] 两元素");
            int i = pr[0], j = pr[1];
            TORCH_CHECK(i >= 0 && i < npartials && j >= 0 && j < npartials && i != j,
                "pair 下标越界或相同: ", i, ",", j);
            if (i > j) std::swap(i, j);
            pair_list.push_back(i * npartials - i * (i - 1) / 2 + (j - i));
            pair_named.emplace_back(i, j);
        }
        const int npairs = (int)pair_list.size();

        torch::Device dev = params.device();
        torch::Tensor extended_vector;
        int nt = 0;
        if (params_.hasCouplingMatrix()) {
            TORCH_CHECK(params.dtype() == torch::kFloat64,
                "params must be float64 with coupling matrix");
            const auto& cm = params_.couplingMatrix();
            int ncf = cm.n_free, na = cm.n_amps;
            nt = (int)params.size(0) - 2 * ncf;
            auto ext = torch::empty({2 * na + nt},
                torch::TensorOptions().dtype(torch::kFloat64).device(dev));
            params_.extendCouplingParams(params.data_ptr<double>(),
                ext.data_ptr<double>(), ncf, nt);
            extended_vector = torch::complex(
                ext.slice(0, 0, na).to(TORCH_FLOAT),
                ext.slice(0, na, 2 * na).to(TORCH_FLOAT));
            if (nt > 0) {
                auto theta = params.slice(0, 2 * ncf, params.size(0));
                theta_cur_ = theta.detach().clone();
                amp_calc_.reComputeAmps(d_all_amplitudes_,
                    reinterpret_cast<const double*>(theta.data_ptr()),
                    n_amplitudes_, events_offsets_, amp_offsets_, n_polar_,
                    dev.index());
            }
        } else {
            TORCH_CHECK(params.dtype() == TORCH_COMPLEX,
                "params must be complex128 in legacy mode");
            extended_vector = params_.extendVector(params, dev);
        }
        std::vector<torch::Tensor> ev_per_gpu;
        for (size_t i = 0; i < d_all_amplitudes_.size(); ++i)
            ev_per_gpu.push_back(extended_vector.to(ctpwa_dev(i)));

        int N_phsp = 0;
        for (size_t gpu = 0; gpu < d_all_amplitudes_.size(); ++gpu)
            N_phsp += events_[gpu][0];
        TORCH_CHECK(N_phsp > 0, "没有 phsp 事件");

        double* h_total = new double[N_phsp];
        double* h_partial = new double[(size_t)N_phsp * npartials];
        double* h_interf = new double[(size_t)N_phsp * npairs];
        auto h_guard = makeScopeExit([&]() {
            delete[] h_total;
            delete[] h_partial;
            delete[] h_interf;
        });

        int ev_cum = 0;
        for (size_t gpu = 0; gpu < d_all_amplitudes_.size(); ++gpu) {
            int ne = events_[gpu][0];
            if (ne == 0) continue;
            cudaSetDevice(gpu);
            // 每 GPU 自己的归约/索引缓冲：跨设备指针传给 computeResults 会非法访问。
            double* d_total_integral = nullptr;
            cudaMalloc(&d_total_integral, sizeof(double));
            int* d_pair_list = nullptr;
            cudaMalloc(&d_pair_list, npairs * sizeof(int));
            cudaMemcpy(d_pair_list, pair_list.data(), npairs * sizeof(int),
                cudaMemcpyHostToDevice);
            int* d_nsl = nullptr;
            cudaMalloc(&d_nsl, npartials * sizeof(int));
            cudaMemcpy(d_nsl, nSLvectors_.data(), npartials * sizeof(int),
                cudaMemcpyHostToDevice);
            double* d_total = nullptr;
            double* d_partial = nullptr;
            double* d_intf = nullptr;
            if (!phsp_freed_) {
                // phsp 驻留模式: 每 GPU 全尺寸结果缓冲
                cudaMalloc(&d_total, ne * sizeof(double));
                cudaMalloc(&d_partial, (size_t)ne * npartials * sizeof(double));
                cudaMemset(d_partial, 0, (size_t)ne * npartials * sizeof(double));
                cudaMalloc(&d_intf, (size_t)ne * npairs * sizeof(double));
            }
            // 本函数只用逐事例干涉项（d_intf）与 partial，矩阵传 nullptr
            cudaMemset(d_total_integral, 0, sizeof(double));

            if (!phsp_freed_) {
                const ctComplex* d_amp_res = d_all_amplitudes_[gpu];
                ctComplex* d_amp_dbl = nullptr;
                if (float_amps_ && !ctIsFloatComplex) {
                    size_t nElem = (size_t)ne * n_polar_ * n_amplitudes_;
                    d_amp_dbl = upcastAmpSegToDouble(d_amp_res, nElem);
                    d_amp_res = d_amp_dbl;
                }
                computeResults(d_amp_res,
                    reinterpret_cast<const ctComplex*>(ev_per_gpu[gpu].data_ptr()),
                    d_total, d_total_integral, d_partial, nullptr, d_intf,
                    nullptr, nullptr, nullptr,
                    d_pair_list, npairs,
                    d_nsl, npartials, ne, n_amplitudes_, n_polar_);
                if (d_amp_dbl) cudaFree(d_amp_dbl);
            } else {
                // phsp 流式模式: 按批重算 phsp 振幅 → computeResults 逐批累加
                const int PHSP_BATCH = 100000;
                long long per_ev_bytes = (long long)n_polar_ * n_amplitudes_ * 8;
                int batch_cap = PHSP_BATCH;
                if (per_ev_bytes > 0) {
                    long long cap = 200LL * 1024 * 1024 / per_ev_bytes;
                    if (cap < 2000) cap = 2000;
                    if (cap < batch_cap) batch_cap = (int)cap;
                }
                auto saved_ev = events_offsets_, saved_amp = amp_offsets_;
                for (int c0 = 0; c0 < ne; c0 += batch_cap) {
                    int nch = std::min(batch_cap, ne - c0);
                    std::vector<std::map<std::string, std::vector<LorentzVector>>> Vp4_chunk(n_gpus_);
                    for (int g = 0; g < n_gpus_; ++g) {
                        if (g == (int)gpu) {
                            for (const auto& [k, v] : Vp4_all_[gpu])
                                Vp4_chunk[g][k].assign(v.begin() + c0, v.begin() + c0 + nch);
                            events_offsets_[g] = {0, nch};
                            amp_offsets_[g] = {0, nch * n_polar_ * n_amplitudes_};
                        } else {
                            for (const auto& [k, v] : Vp4_all_[g])
                                Vp4_chunk[g][k].assign(v.begin(), v.begin() + 1);
                            events_offsets_[g] = {0, 1};
                            amp_offsets_[g] = {0, n_polar_ * n_amplitudes_};
                        }
                    }
                    std::vector<ctComplex*> d_chunk = calculateAmplitudes(Vp4_chunk);
                    cudaSetDevice(gpu);
                    double* d_ct;
                    cudaMalloc(&d_ct, nch * sizeof(double));
                    double* d_cp;
                    cudaMalloc(&d_cp, (size_t)nch * npartials * sizeof(double));
                    cudaMemset(d_cp, 0, (size_t)nch * npartials * sizeof(double));
                    double* d_cintf;
                    cudaMalloc(&d_cintf, (size_t)nch * npairs * sizeof(double));
                    {
                    const ctComplex* d_chunk_res = d_chunk[gpu];
                    ctComplex* d_chunk_dbl = nullptr;
                    if (float_amps_ && !ctIsFloatComplex) {
                        size_t nElem = (size_t)nch * n_polar_ * n_amplitudes_;
                        d_chunk_dbl = upcastAmpSegToDouble(d_chunk_res, nElem);
                        d_chunk_res = d_chunk_dbl;
                    }
                    computeResults(d_chunk_res,
                        reinterpret_cast<const ctComplex*>(ev_per_gpu[gpu].data_ptr()),
                        d_ct, d_total_integral, d_cp, nullptr, d_cintf,
                        nullptr, nullptr, nullptr,
                        d_pair_list, npairs,
                        d_nsl, npartials, nch, n_amplitudes_, n_polar_);
                    if (d_chunk_dbl) cudaFree(d_chunk_dbl);
                    }
                    cudaMemcpy(h_total + ev_cum + c0, d_ct, nch * sizeof(double),
                        cudaMemcpyDeviceToHost);
                    for (int p = 0; p < npartials; ++p)
                        cudaMemcpy(h_partial + (size_t)p * N_phsp + ev_cum + c0,
                            d_cp + (size_t)p * nch, nch * sizeof(double),
                            cudaMemcpyDeviceToHost);
                    for (int q = 0; q < npairs; ++q)
                        cudaMemcpy(h_interf + (size_t)q * N_phsp + ev_cum + c0,
                            d_cintf + (size_t)q * nch, nch * sizeof(double),
                            cudaMemcpyDeviceToHost);
                    cudaFree(d_ct); cudaFree(d_cp); cudaFree(d_cintf);
                    for (int g = 0; g < n_gpus_; ++g)
                        if (d_chunk[g]) { cudaSetDevice(g); cudaFree(d_chunk[g]); }
                    cudaSetDevice(gpu);
                }
                events_offsets_ = saved_ev;
                amp_offsets_ = saved_amp;
            }

            if (!phsp_freed_) {
                // phsp 驻留模式: 每 GPU 全尺寸结果缓冲 → 一次性拷到 host
                cudaMemcpy(h_total + ev_cum, d_total, ne * sizeof(double),
                    cudaMemcpyDeviceToHost);
                for (int p = 0; p < npartials; ++p)
                    cudaMemcpy(h_partial + (size_t)p * N_phsp + ev_cum,
                        d_partial + (size_t)p * ne, ne * sizeof(double),
                        cudaMemcpyDeviceToHost);
                for (int q = 0; q < npairs; ++q)
                    cudaMemcpy(h_interf + (size_t)q * N_phsp + ev_cum,
                        d_intf + (size_t)q * ne, ne * sizeof(double),
                        cudaMemcpyDeviceToHost);
            }
            // 流式模式: host 结果已在分块循环内逐批拷出
            ev_cum += ne;
            cudaFree(d_nsl);
            cudaFree(d_total_integral);
            cudaFree(d_pair_list);
            if (d_total) cudaFree(d_total);
            if (d_partial) cudaFree(d_partial);
            if (d_intf) cudaFree(d_intf);
        }
        cudaSetDevice(dev.index());

        double h_phsp_integral = 0.0;
        for (int i = 0; i < N_phsp; ++i) h_phsp_integral += h_total[i];
        double dataIntegral = data_total_weight_ - bkg_integral_;
        double normFactor = h_phsp_integral > 0
            ? dataIntegral / h_phsp_integral : 1.0;

        TFile* rootFile = new TFile(filename.c_str(), "RECREATE");
        auto wi_root_guard = makeScopeExit([&]() {
            if (rootFile) { rootFile->Close(); delete rootFile; rootFile = nullptr; }
        });
        TTree* tree = new TTree("saved_weight", "interference shapes per event");
        double total_weight = 0.0;
        tree->Branch("totalweight", &total_weight);
        std::vector<double> partial_weights(npartials, 0.0);
        for (int i = 0; i < npartials; ++i)
            tree->Branch(("weight_" + resonance_names_[i]).c_str(), &partial_weights[i]);
        std::vector<double> interf_vals(npairs, 0.0);
        for (int q = 0; q < npairs; ++q) {
            std::string bn = "interf_" + std::to_string(pair_named[q].first)
                + "_" + std::to_string(pair_named[q].second);
            tree->Branch(bn.c_str(), &interf_vals[q]);
        }
        // 末态四动量
        std::vector<std::string> mom_names;
        if (!Vp4_all_.empty())
            for (const auto& kv : Vp4_all_[0]) mom_names.push_back(kv.first);
        const int n_mom = (int)mom_names.size();
        std::vector<LorentzVector> mom_host((size_t)N_phsp * std::max(1, n_mom));
        {
            int off = 0;
            for (size_t gpu = 0; gpu < Vp4_all_.size(); ++gpu) {
                int ne = events_[gpu][0];
                for (int m = 0; m < n_mom; ++m) {
                    const auto& vec = Vp4_all_[gpu].at(mom_names[m]);
                    for (int e = 0; e < ne; ++e)
                        mom_host[(size_t)(off + e) * n_mom + m] = vec[e];
                }
                off += ne;
            }
        }
        std::vector<double> mom_vals(std::max(1, n_mom) * 4, 0.0);
        for (int m = 0; m < n_mom; ++m) {
            const std::string& nm = mom_names[m];
            tree->Branch((nm + "_px").c_str(), &mom_vals[m * 4 + 0]);
            tree->Branch((nm + "_py").c_str(), &mom_vals[m * 4 + 1]);
            tree->Branch((nm + "_pz").c_str(), &mom_vals[m * 4 + 2]);
            tree->Branch((nm + "_E").c_str(),  &mom_vals[m * 4 + 3]);
        }
        for (int i = 0; i < N_phsp; ++i) {
            total_weight = h_total[i] * normFactor;
            for (int j = 0; j < npartials; ++j)
                partial_weights[j] = h_partial[(size_t)j * N_phsp + i] * normFactor;
            for (int q = 0; q < npairs; ++q)
                interf_vals[q] = h_interf[(size_t)q * N_phsp + i] * normFactor;
            for (int m = 0; m < n_mom; ++m) {
                const LorentzVector& mv = mom_host[(size_t)i * n_mom + m];
                mom_vals[m * 4 + 0] = mv.Px;
                mom_vals[m * 4 + 1] = mv.Py;
                mom_vals[m * 4 + 2] = mv.Pz;
                mom_vals[m * 4 + 3] = mv.E;
            }
            tree->Fill();
        }
        tree->Write();
        rootFile->Close();
        delete rootFile;
        rootFile = nullptr;
        // h_* 由 h_guard 统一释放（含异常路径）。
        std::cout << "writeInterfResult: " << filename
                  << " (pairs=" << npairs << ", events=" << N_phsp << ")" << std::endl;
        return;
    
}

