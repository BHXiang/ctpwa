// src/Initialize.cu — analysis 的初始化与强度原语（P4：从 Analysis.h 内联定义搬出）
//
// 零风险迁出：只把"类内定义"改成"类内声明 + 这里的 analysis:: 定义"，
// 方法体一字未改，接口未动。
#include <Analysis.h>

void analysis::initialize(std::string config_file)
{

        // 读取配置文件
        // std::string config_file = "config.yml";
        // config_parser_(config_file);
        std::cout << "Reading config file: " << config_file << std::endl;
        std::cout << "Particles: " << config_parser_.getParticles().size()
            << std::endl;
        std::cout << "Decay chains: " << config_parser_.getDecayChains().size()
            << std::endl;
        std::cout << "Resonances: " << config_parser_.getResonances().size()
            << std::endl;
        std::cout << "Constraints: " << config_parser_.getConstraints().size()
            << std::endl;

        // 主设备统一固定为 cuda:0。d_phsp_matrix_、耦合表和统一参数向量都
        // 先放在主卡，再显式复制到其他卡；不能继承前一个多卡调用残留的
        // current device=1，否则后续 make_params(cuda:0) 与主卡定义冲突。
        primary_dev_ = 0;
        ctpwa::ensure_device(primary_dev_);
        params_.setPrimaryDevice(primary_dev_);

        // 初始化粒子信息
        initializeParticles();
        // 初始化极化状态
        initializePolarization();
        // 初始化衰变链
        initializeDecayChains();

        // // 获取配置信息
        // const auto &config_parser = calculator.getConfigParser();
        const auto& data_files = config_parser_.getDataFiles();
        const auto& data_order = config_parser_.getDataOrder();

        legends_ = config_parser_.getLegends();
        n_gls_ = n_amplitudes_;

        std::vector<std::string> particles_names;
        for (const auto& particle : particles_)
        {
            particles_names.push_back(particle.name);
        }
        // events_.clear();
        std::vector<int> init_events;
        init_events.clear();

        std::vector<std::map<std::string, std::vector<LorentzVector>>> Vp4_to_merge;

        // 计算相空间振幅
        std::cout << "Reading phase space samples..." << std::endl;
        auto Vp4_phsp = readMomentaFromDat(data_files.at("phsp"), data_order, particles_names);
        std::cout << "events: " << Vp4_phsp.begin()->second.size() << std::endl;
        init_events.push_back(Vp4_phsp.begin()->second.size());
        Vp4_to_merge.push_back(Vp4_phsp);

        // 计算数据振幅
        std::cout << "Reading data samples..." << std::endl;
        auto Vp4_data = readMomentaFromDat(data_files.at("data"), data_order, particles_names);
        std::cout << "events: " << Vp4_data.begin()->second.size() << std::endl;
        init_events.push_back(Vp4_data.begin()->second.size());
        Vp4_to_merge.push_back(Vp4_data);

        // 计算本底振幅
        if (data_files.count("bkg") > 0)
        {
            std::cout << "Reading background samples..." << std::endl;
            auto Vp4_bkg = readMomentaFromDat(data_files.at("bkg"), data_order, particles_names);
            std::cout << "events: " << Vp4_bkg.begin()->second.size() << std::endl;
            init_events.push_back(Vp4_bkg.begin()->second.size());
            Vp4_to_merge.push_back(Vp4_bkg);
        }

        // 设备检测：枚举 GPU 属性/显存，替换散落的 cudaGetDeviceCount
        device_mgr_.detect();
        n_gpus_ = device_mgr_.numDevices();

        // 精度解析（运行期档位; .so 编译 double 为默认产物）:
        //   precision:double → PrecMode::Double 全 double（16B 全表 + double 计算）
        //   precision:hybrid / auto（缺省）→ 旧 float-A 混合: A 存 float2(8B),
        //     核心计算仍 double（NLL 相对 double 漂移 ~1e-8）——大统计量默认省显存档
        //   precision:float   → PrecMode::Float 全 float 档（模板化双实例路线,
        //     存储与计算均 float; 逐批接线, 未接线入口显式报错）
        //   .so 编译 float:  precision:float/hybrid/auto → 原生 float 行为;
        //                    precision:double → 报错（float .so 无法提精度）
        std::string req = config_parser_.getPrecision();
        {
            if (req != "auto" && req != "hybrid" && req != "float" && req != "double") {
                std::cerr << "ERROR: 配置 precision=\"" << req
                          << "\" 无效（仅支持 auto | hybrid | float | double）" << std::endl;
                throw std::runtime_error("invalid precision in config");
            }
            const bool double_so = std::string(PRECISION_NAME) == "double";
            prec_mode_ = PrecMode::Double;
#if defined(CTPWA_BACKEND_CPU)
            // CPU 后端固定全 double：float2 档位的意义是省显存，RAM 上不需要，
            // 而且能直接砍掉 9 个 float2 专用 kernel（见 doc/cpu-backend-plan.md §1.3）。
            // 配置里的 precision 仍照收，只是全部映射到 double。
            if (req != "double") {
                std::cerr << "[ctpwa] CPU 后端：precision=" << req
                          << " 已按 double 运行（float 档位仅用于省显存）" << std::endl;
            }
            prec_mode_ = PrecMode::Double;
            float_amps_ = false;
#else
            if (req == "double") {
                if (!double_so) {
                    std::cerr << "ERROR: 配置文件请求 precision=double，但当前 .so 编译为 float；"
                                 "请用默认编译（double）重新 build_ext。float .so 仅供大统计量省显存。"
                              << std::endl;
                    throw std::runtime_error("double requested on float build");
                }
            } else if (!double_so) {
                // float .so: float/hybrid/auto 均原生 float（编译期已定），档位记 Float
                prec_mode_ = PrecMode::Float;
                float_amps_ = true;
            } else if (req == "float") {
                // double .so 上的"全 float 档"（模板化双实例；接线中）
                prec_mode_ = PrecMode::Float;
                float_amps_ = true;   // A 等大表先按 float2 存储（hybrid 同款布局）
            } else {
                // double .so 上 hybrid / auto: A float2 + double 核心（默认体验）
                prec_mode_ = PrecMode::Double;
                float_amps_ = true;
            }
#endif  // CTPWA_BACKEND_CPU
            // 精度档位 banner 注释掉（真实分析每次 ctor 刷屏）:
            // 精度信息由 DeviceManager.print() 的 "complex precision" 行提供。
            // if (prec_mode_ == PrecMode::Float) {
            //     std::cout << "[ctpwa] 全 float 档（precision:float）: 存储/计算 float;"
            //                  " 尚未接线的入口会显式报错，请先用 precision:hybrid/double" << std::endl;
            // } else if (float_amps_) {
            //     std::cout << "[ctpwa] 混合精度模式: A 存 float（省显存）, 核心计算 double";
            //     if (req == "auto" || req == "hybrid")
            //         std::cout << "（precision:" << req << " 默认; 需全 double 请在 config 写 precision: double）";
            //     std::cout << std::endl;
            // }
            amp_calc_.setFloatOutput(float_amps_);
            // d_dF 表存储档位: 仅全 float 档存 float2（hybrid 保持 double 语义）
            amp_calc_.setDFloat(prec_mode_ == PrecMode::Float);
            // 容量预检必须使用 A 的运行时存储档位：double .so + auto/hybrid
            // 实际是 float2(8B)，不能再按 sizeof(ctComplex)=16B 高估一倍。
            device_mgr_.setAmplitudeBytes(float_amps_ ? sizeof(float2) : sizeof(ctComplex));
        }
        if (n_gpus_ == 0) {
            std::cerr << "ERROR: 无可用 CUDA 设备。ctpwa 当前仅支持 GPU 计算"
                         "（CPU 后端尚未实现），无法继续。" << std::endl;
            throw std::runtime_error("no CUDA devices available");
        }
        // 运行时存储档位（打印在 DeviceManager 的 complex 编译精度行之后）:
        //   auto/hybrid: A float2(8B) + dF double2(16B) + double 核心
        //   float:       A/dF float2(8B) + float 核心（Hessian 走 double 上转）
        //   double:      全 double2(16B)
        {
            const char* a_st = float_amps_ ? "float2 (8B)" : "double2 (16B)";
            const char* d_st = (prec_mode_ == PrecMode::Float) ? "float2 (8B)"
                                                               : "double2 (16B)";
            const char* core =
                (prec_mode_ == PrecMode::Float) ? "float" : "double";
            std::string note = std::string("A ") + a_st + " | dF " + d_st +
                               " | 核心 " + core + " (precision: " + req + ")";
            device_mgr_.setStorageNote(note);
        }
        // DeviceManager 设备列表延迟到内存预检之后打印——那时 checkCapacity
        // 已把每 GPU 预计占用写入 estimated_usage_, memory 字段显示"预计/总量"
        // 而非"空闲/总量"。需设备信息本身时仍可调 ctpwa.DeviceManager().print()。

        initializeMultiGPUs(init_events);

        // ---- Constraints.free_phsp_amplitudes: phsp 流式（不驻留）模式判定 ----
        // 仅当所有共振态都没有 free 参数（质量/宽度全固定）时生效；
        // fix_var 只会移除、不会新增自由参数，故 config 层判定是保守且安全的。
        bool phsp_streamed = config_parser_.getFreePhspAmplitudes();
        if (phsp_streamed) {
            bool any_free_declared = false;
            for (const auto& [rname, rc] : config_parser_.getResonances()) {
                (void)rname;
                if (!rc.free.empty()) { any_free_declared = true; break; }
            }
            if (any_free_declared) {
                std::cerr << "WARNING: Constraints.free_phsp_amplitudes 需要所有共振态"
                             "质量/宽度固定（free 为空）；检测到 free 参数，开关被忽略"
                             "（phsp 保持驻留）。" << std::endl;
                phsp_streamed = false;
            }
        }

        // 内存预检：输入数据能否被设备集承载（含每 GPU 事件分布）
        {
            // 常驻振幅表按该 GPU 的 phsp+data+bkg 总事件数分配，不能用单个
            // 样本的最大值估；其他临时 buffer 用它作上界是可接受的保守估计。
            // phsp 流式模式下 phsp 不驻留，只统计 data/bkg。
            std::vector<int> resident_events(n_gpus_, 0);
            for (int gpu = 0; gpu < n_gpus_; ++gpu)
                for (size_t j = (phsp_streamed ? 1 : 0); j < events_[gpu].size(); ++j)
                    resident_events[gpu] += events_[gpu][j];
            bool has_bkg = (init_events.size() > 2 && init_events[2] > 0);
            // SL 表按独立的 chain×JP 级联组合分配，同一 JP 下的全部共振态组合
            // 共享一张表。无 trans/var_equal 时，固定 JP 的表在 A 物化后立即释放；
            // 此时构造期峰值 = max(此前自由 JP 的保留表 + 当前 JP 表)，而不是全表求和。
            const bool early_sl_release = earlyStaticSLReleaseEnabled();
            const auto& config_res = config_parser_.getResonances();
            int n_slcombs_total = 0;
            int n_slcombs_retained = 0;
            for (const auto& chain_info : chains_info_) {
                TORCH_CHECK(chain_info.slcombs_per_jp.size() == chain_info.intermediate_combs.size(),
                    "ChainInfo ", chain_info.name, " 的 SL 计数与 JP 组合数不一致");
                for (size_t ci = 0; ci < chain_info.intermediate_combs.size(); ++ci) {
                    const int nsl = chain_info.slcombs_per_jp[ci];
                    if (!early_sl_release) {
                        n_slcombs_total += nsl;
                        continue;
                    }
                    // 同一 cas 下只要有一个共振态组合含自由参数，整张 SL 表都要保留。
                    // fix_var 可能进一步把它固定，因此这里的保留估计只会偏保守。
                    bool keep_cas = false;
                    for (const auto& particle : chain_info.intermediate_combs[ci]) {
                        const auto key = std::make_pair(particle.name,
                            std::vector<int>{particle.spin, particle.parity});
                        const auto ri = chain_info.intermediate_resonance_map.find(key);
                        if (ri == chain_info.intermediate_resonance_map.end()) continue;
                        for (const auto& res : ri->second) {
                            const auto cfg = config_res.find(res.getName());
                            if (cfg != config_res.end() && !cfg->second.free.empty())
                                keep_cas = true;
                        }
                    }
                    n_slcombs_total = std::max(n_slcombs_total, n_slcombs_retained + nsl);
                    if (keep_cas) n_slcombs_retained += nsl;
                }
            }
            auto cap = device_mgr_.checkCapacity(
                resident_events, n_amplitudes_, n_polar_, n_slcombs_total, particles_.size(),
                has_bkg, params_.nFreeTheta());
            // 预检后打印设备列表（含每 GPU 预计占用）
            device_mgr_.print();
            if (cap.overall == DeviceManager::CapacityStatus::FAIL) {
                std::cerr << "ERROR: 内存预检失败——GPU " << cap.failing_device
                          << " 无法承载数据: " << cap.failing_buffer
                          << " 预计总计 " << cap.required_bytes / 1e6
                          << " MiB，可用 " << cap.available_bytes / 1e6
                          << " MiB（事件数 " << resident_events[cap.failing_device]
                          << "）。请减少事件数或使用多 GPU。" << std::endl;
                for (const auto& [name, bytes] : cap.breakdown) {
                    std::cerr << "    - " << name << ": "
                              << bytes / 1048576.0 << " MiB" << std::endl;
                }
                if (std::getenv("CTPWA_SKIP_MEM_CHECK") != nullptr) {
                    std::cerr << "WARNING: CTPWA_SKIP_MEM_CHECK 已设置，跳过预检；"
                                 "若实际显存不足会在 cudaMalloc 处报错。" << std::endl;
                } else {
                    throw std::runtime_error("input data exceeds device memory");
                }
            } else if (cap.overall == DeviceManager::CapacityStatus::WARN) {
                std::cerr << "WARNING: 内存占用超过可用显存的 80%，"
                             "建议减少事件数或增加 GPU。" << std::endl;
            } else {
                std::cout << "Memory check passed." << std::endl;
            }
        }

        std::cout << "Calculating amplitudes..." << std::endl;
        Vp4_all_ = mergeMaps(Vp4_to_merge, events_);
        events_offsets_.clear();
        amp_offsets_.clear();
        for (size_t i = 0; i < events_.size(); ++i)
        {
            std::vector<int> ev_offsets;
            std::vector<int> amp_offsets;
            int ev_sum = 0;
            int amp_sum = 0;
            ev_offsets.push_back(ev_sum);
            amp_offsets.push_back(amp_sum);
            for (size_t j = 0; j < events_[i].size(); ++j)
            {
                // std::cout << "Events in dataset " << i << ", type " << j << ": " << events_[i][j] << std::endl;
                ev_sum += events_[i][j];
                amp_sum += events_[i][j] * n_polar_ * n_amplitudes_;
                ev_offsets.push_back(ev_sum);
                amp_offsets.push_back(amp_sum);
                // std::cout << "  Cumulative events: " << ev_sum << ", Cumulative amplitudes: " << amp_sum << std::endl;
            }
            events_offsets_.push_back(ev_offsets);
            amp_offsets_.push_back(amp_offsets);
        }

        if (phsp_streamed) {
            // ---- phsp 流式模式：常驻表只含 data+bkg ----
            // 常驻表布局 [data | bkg]：data 在偏移 0，bkg 在 events_[gpu][1]·PA。
            // 消费端统一用 amp_offsets_[gpu][1]/[2] 取 data/bkg 基址：
            //   freed:  amp_offsets_ = {0, 0, N_data·PA, (N_data+N_bkg)·PA} → data=0, bkg=N_data·PA
            //   非 freed: amp_offsets_ = {0, N_phsp·PA, (N_phsp+N_data)·PA, ...}（虚拟，含 phsp）
            // 因此下面把成员 amp_offsets_ 改写为主表相对约定（data=0）。
            std::vector<std::map<std::string, std::vector<LorentzVector>>> Vp4_db(n_gpus_);
            for (size_t g = 0; g < (size_t)n_gpus_; ++g) {
                int nP = events_[g][0];
                int nD = events_[g][1];
                int nB = (events_[g].size() > 2) ? events_[g][2] : 0;
                for (const auto& [k, v] : Vp4_all_[g]) {
                    auto& dst = Vp4_db[g][k];
                    dst.assign(v.begin() + nP, v.begin() + nP + nD + nB);
                }
                events_offsets_[g] = {0, nD, nD + nB};
                amp_offsets_[g] = {0, nD * n_polar_ * n_amplitudes_,
                                   (nD + nB) * n_polar_ * n_amplitudes_};
            }
            d_all_amplitudes_ = calculateAmplitudes(Vp4_db, &amp_calc_);
            // 防御: config 层判定（无 free 声明）应与实际自由参数数一致
            if (amp_calc_.nFreeResParams() > 0) {
                throw std::runtime_error(
                    "free_phsp_amplitudes 与自由共振态参数检测不一致"
                    "（不应发生：free 参数只来自 Resonances.free）");
            }
            // 恢复事件偏移（虚拟布局，四动量索引用）；振幅偏移改写为主表相对约定
            for (size_t g = 0; g < (size_t)n_gpus_; ++g) {
                std::vector<int> ev_off = {0};
                int cum = 0;
                for (size_t j = 0; j < events_[g].size(); ++j) {
                    cum += events_[g][j];
                    ev_off.push_back(cum);
                }
                events_offsets_[g] = ev_off;
                std::vector<int> amp_off = {0, 0};
                int amp_cum = 0;
                for (size_t j = 1; j < events_[g].size(); ++j) {
                    amp_cum += events_[g][j] * n_polar_ * n_amplitudes_;
                    amp_off.push_back(amp_cum);
                }
                amp_offsets_[g] = amp_off;   // {0, 0, N_data·PA, (N_data+N_bkg)·PA}
            }
            phsp_freed_ = true;
        } else {
            d_all_amplitudes_ = calculateAmplitudes(Vp4_all_, &amp_calc_);
        }

#if defined(CTPWA_BACKEND_CPU)
        // 诊断：CPU 后端振幅表填充检查（CTPWA_CPU_DEBUG=1）
        if (getenv("CTPWA_CPU_DEBUG") && !d_all_amplitudes_.empty()) {
            for (size_t g = 0; g < d_all_amplitudes_.size(); ++g) {
                const long long n = (long long)amp_offsets_[g].back();
                if (n <= 0 || d_all_amplitudes_[g] == nullptr) continue;
                std::vector<ctComplex> h((size_t)n);
                cudaMemcpy(h.data(), d_all_amplitudes_[g],
                           (size_t)n * sizeof(ctComplex), cudaMemcpyDeviceToHost);
                long long n_nan = 0, n_zero = 0;
                double vmin = 1e300, vmax = -1e300, amax = 0.0;
                for (long long i = 0; i < n; ++i) {
                    const double re = (double)h[(size_t)i].x, im = (double)h[(size_t)i].y;
                    if (std::isnan(re) || std::isnan(im)) { ++n_nan; continue; }
                    if (re == 0.0 && im == 0.0) ++n_zero;
                    vmin = std::min(vmin, std::min(re, im));
                    vmax = std::max(vmax, std::max(re, im));
                    amax = std::max(amax, std::sqrt(re * re + im * im));
                }
                std::printf("[CPU-DBG] gpu %zu: n=%lld nan=%lld zero=%lld "
                            "min=%.6g max=%.6g |A|max=%.6g | first8:",
                            g, n, n_nan, n_zero, vmin, vmax, amax);
                for (int i = 0; i < 8 && i < n; ++i)
                    std::printf(" (%.4g,%.4g)", (double)h[(size_t)i].x,
                                (double)h[(size_t)i].y);
                std::printf("\n");
            }
        }
#endif

        // 设置共振态自由参数个数（calculateAmplitudes 内部已应用 var_equal 槽合并）
        if (!amp_calc_.empty()) {
            params_.setNFreeTheta(amp_calc_.nFreeResParams());
            // fix_var/var_equal 移除了部分槽: 用实际槽名重建 theta 参数名段，
            // 保证参数名与自由参数一一对应（getParamNames 长度不变量）
            const auto& slots = amp_calc_.slots();
            auto names = params_.paramNames();
            if (static_cast<int>(names.size()) >= n_coupling_names_) {
                names.resize(n_coupling_names_);
                for (const auto& s : slots) names.push_back(s.name);
                params_.setParamNames(names);
            }
        }

        // Constraints.gauss_constr: name → theta 下标映射（μ 默认取 slot 初值）
        if (!amp_calc_.empty() && !config_parser_.getGaussConstr().empty()) {
            const auto& gc = config_parser_.getGaussConstr();
            const auto& slots = amp_calc_.slots();
            std::map<std::string, int> name_to_idx;
            std::map<std::string, double> sigma, mu;
            for (size_t s = 0; s < slots.size(); ++s) {
                auto git = gc.find(slots[s].name);
                if (git != gc.end()) {
                    name_to_idx[slots[s].name] = static_cast<int>(s);
                    sigma[slots[s].name] = git->second;
                    mu[slots[s].name] = slots[s].init_value;
                }
            }
            params_.setGaussConstr(name_to_idx, sigma, mu);
        }

        // 输出d_all_amplitudes_[0]所有内容:
        // int Ntotal = 0;
        // for (size_t j = 0; j < events_[0].size(); ++j)
        // {
        //     Ntotal += events_[0][j];
        // }
        // ctComplex* h_amp = new ctComplex[Ntotal * n_polar_ * n_amplitudes_];
        // cudaMemcpy(h_amp, d_all_amplitudes_[0], Ntotal * n_polar_ * n_amplitudes_ * sizeof(ctComplex), cudaMemcpyDeviceToHost);
        // for (int i = 0; i < Ntotal * n_polar_ * n_amplitudes_; ++i)
        // {
        //     std::cout << "Amplitude[" << i << "] = " << h_amp[i].x << " + " << h_amp[i].y << "i" << std::endl;
        // }
        // delete[] h_amp;

        // data_weights_
        data_total_weight_ = 0.0;
        if (data_files.count("data_weights") > 0)
        {
            std::vector<int> data_events_per_gpu;
            for (size_t i = 0; i < events_.size(); ++i)
                data_events_per_gpu.push_back(events_[i][1]);
            data_weights_ = readWeightsFromFile(data_files.at("data_weights"), data_events_per_gpu);
            // Σ data 权重（phsp 归一化 N_eff 用；无权重 GPU 按事件数计）
            for (size_t gpu = 0; gpu < data_weights_.size(); ++gpu) {
                if (data_weights_[gpu] != nullptr && events_[gpu].size() > 1 && events_[gpu][1] > 0) {
                    cudaSetDevice(gpu);
                    thrust::device_ptr<double> dp(data_weights_[gpu]);
                    data_total_weight_ += thrust::reduce(dp, dp + events_[gpu][1]);
                }
                else if (events_[gpu].size() > 1) {
                    data_total_weight_ += events_[gpu][1];
                }
            }
        }
        else
        {
            for (size_t i = 0; i < events_.size(); ++i) {
                data_weights_.push_back(nullptr);
                if (events_[i].size() > 1)
                    data_total_weight_ += events_[i][1];
            }
        }

        // phsp_weights_
        if (data_files.count("phsp_weights") > 0)
        {
            std::vector<int> phsp_events_per_gpu;
            for (size_t i = 0; i < events_.size(); ++i)
                phsp_events_per_gpu.push_back(events_[i][0]);
            phsp_weights_ = readWeightsFromFile(data_files.at("phsp_weights"), phsp_events_per_gpu);
        }
        else
        {
            for (size_t i = 0; i < events_.size(); ++i)
                phsp_weights_.push_back(nullptr);
        }

        // bkg_weights_
        if (data_files.count("bkg_weights") > 0 && data_files.count("bkg") > 0)
        {
            std::vector<int> bkg_events_per_gpu;
            for (size_t i = 0; i < events_.size(); ++i)
                bkg_events_per_gpu.push_back(events_[i][2]);
            bkg_weights_ = readWeightsFromFile(data_files.at("bkg_weights"), bkg_events_per_gpu);

            // bkg_weigths_求和
            double h_bkg_integral = 0.0;
            for (size_t gpu = 0; gpu < d_all_amplitudes_.size(); ++gpu) {
                cudaSetDevice(gpu);
                if (events_[gpu][2] > 0) {
                    thrust::device_ptr<double> d_ptr_gpu(bkg_weights_[gpu]);
                    h_bkg_integral += thrust::reduce(d_ptr_gpu, d_ptr_gpu + events_[gpu][2]);
                }
            }
            bkg_integral_ = h_bkg_integral;
        }
        else
        {
            for (size_t i = 0; i < events_.size(); ++i)
            {
                bkg_weights_.push_back(nullptr);
                if (events_[i].size() > 2)
                    bkg_integral_ += events_[i][2];
            }
        }

        // // 打印d_all_amplitudes_[0]的所有元素，验证数据正确加载
        // ctComplex* h_amp = new ctComplex[n_amplitudes_ * n_polar_ * (init_events[0] + init_events[1])];
        // cudaMemcpy(h_amp, d_all_amplitudes_[0], n_amplitudes_ * n_polar_ * (init_events[0] + init_events[1]) * sizeof(ctComplex), cudaMemcpyDeviceToHost);
        // std::cout << "First amplitudes on GPU 0:" << std::endl;
        // for (int i = 0; i < n_amplitudes_ * n_polar_ * (init_events[0] + init_events[1]); ++i)
        // {
        //     std::cout << "  Amplitude[" << i << "] = " << h_amp[i].x << " + " << h_amp[i].y << "i" << std::endl;
        // }
        // delete[] h_amp;

        // phsp*phsp^T矩阵: A^H diag(w) A / (Σ w)，加权或均匀
        // Pre-compute total phsp weight for correct global normalization
        double W_total = 0.0;
        for (size_t gpu = 0; gpu < d_all_amplitudes_.size(); ++gpu) {
            int nP = events_[gpu][0];
            if (nP == 0) continue;
            if (gpu < phsp_weights_.size() && phsp_weights_[gpu] != nullptr) {
                cudaSetDevice(gpu);
                thrust::device_ptr<double> dp(phsp_weights_[gpu]);
                W_total += thrust::reduce(dp, dp + nP);
            } else {
                W_total += (double)nP;
            }
        }
        if (W_total <= 0.0) W_total = 1.0;

        ctpwa::ensure_device(primary_dev_);  // ensure d_phsp_matrix_ is allocated on primary GPU
        cudaMalloc(&d_phsp_matrix_, n_amplitudes_ * n_amplitudes_ * sizeof(ctComplex));
        cudaMemset(d_phsp_matrix_, 0, n_amplitudes_ * n_amplitudes_ * sizeof(ctComplex));

        if (phsp_freed_) {
            // ---- phsp 流式（不驻留）: 分块算 phsp 振幅 → 累加 float32 + double 矩阵 ----
            // 每批: calculateAmplitudes(单批 Vp4) → 批振幅临时缓冲 → 累加矩阵 → 释放。
            // 峰值 = 常驻 data/bkg 表 + 一批振幅，与 phsp 总量无关。
            cudaMalloc(&d_phsp_matrix_double_,
                (size_t)n_amplitudes_ * n_amplitudes_ * sizeof(cuDoubleComplex));
            cudaMemset(d_phsp_matrix_double_, 0,
                (size_t)n_amplitudes_ * n_amplitudes_ * sizeof(cuDoubleComplex));
            const int PHSP_BATCH = 100000;
            long long per_ev_bytes = (long long)n_polar_ * n_amplitudes_ * 8;
            int batch_cap = PHSP_BATCH;
            if (per_ev_bytes > 0) {
                long long cap = 200LL * 1024 * 1024 / per_ev_bytes;  // ≤ ~200MB/批
                if (cap < 2000) cap = 2000;
                if (cap < batch_cap) batch_cap = (int)cap;
            }
            std::cout << "[ctpwa] phsp 流式模式: 分块构建 phsp 矩阵 (batch="
                      << batch_cap << " 事件/批)..." << std::endl;
            auto saved_ev = events_offsets_, saved_amp = amp_offsets_;
            for (int gpu = 0; gpu < n_gpus_; ++gpu) {
                int nPhsp = events_[gpu][0];
                if (nPhsp == 0) continue;
                cudaSetDevice(gpu);
                cublasHandle_t h;
                cublasCreate(&h);
                for (int c0 = 0; c0 < nPhsp; c0 += batch_cap) {
                    int nch = std::min(batch_cap, nPhsp - c0);
                    // 批四动量（本 GPU 取 [c0, c0+nch)，其他 GPU 放 1 事件占位防空 map）
                    std::vector<std::map<std::string, std::vector<LorentzVector>>> Vp4_batch(n_gpus_);
                    for (int g = 0; g < n_gpus_; ++g) {
                        if (g == gpu) {
                            for (const auto& [k, v] : Vp4_all_[gpu])
                                Vp4_batch[g][k].assign(v.begin() + c0, v.begin() + c0 + nch);
                            events_offsets_[g] = {0, nch};
                            amp_offsets_[g] = {0, nch * n_polar_ * n_amplitudes_};
                        } else {
                            for (const auto& [k, v] : Vp4_all_[g])
                                Vp4_batch[g][k].assign(v.begin(), v.begin() + 1);
                            events_offsets_[g] = {0, 1};
                            amp_offsets_[g] = {0, n_polar_ * n_amplitudes_};
                        }
                    }
                    std::vector<ctComplex*> d_batch = calculateAmplitudes(Vp4_batch);
                    cudaSetDevice(gpu);
                    int nTot = nch * n_polar_ * n_amplitudes_;

                    // float 模式（double .so 混合精度）: 批振幅为 float2（8B/元素），
                    // 下方"原生 ctComplex 累加"与"double 累加"都按 ctComplex/cuDoubleComplex
                    // 读——先上转一份 double 段供两处共用（峰值 = 常驻表 + 一批 ×2，
                    // batch_cap ≤ ~200MB/批 → 有界，与 phsp 总量无关）。
                    ctComplex* d_batch_up = nullptr;   // 非空 = d_batch_r 指向它的 double 上转段
                    const ctComplex* d_batch_r = d_batch[gpu];
                    if (float_amps_ && !ctIsFloatComplex) {
                        d_batch_up = upcastAmpSegToDouble(d_batch[gpu], (size_t)nTot);
                        d_batch_r = d_batch_up;
                    }

                    // float32 累加（scale 后 CGEMM）
                    {
                        ctComplex* d_scaled;
                        cudaMalloc(&d_scaled, nTot * sizeof(ctComplex));
                        cudaMemcpy(d_scaled, d_batch_r, nTot * sizeof(ctComplex),
                                   cudaMemcpyDeviceToDevice);
                        const double* d_w = (gpu < (int)phsp_weights_.size())
                            ? phsp_weights_[gpu] : nullptr;
                        int grid = (nTot + 255) / 256;
                        CTPWA_LAUNCH(scalePhspAmpsKernel, grid, 256, 0, d_scaled,
                            d_w ? d_w + c0 : nullptr, nch, n_polar_, n_amplitudes_,
                            1.0 / W_total, 0);
                        ctComplex* d_m;
                        cudaMalloc(&d_m, (size_t)n_amplitudes_ * n_amplitudes_ * sizeof(ctComplex));
                        ctComplex alpha = ctMake(1.0f, 0.0f), beta = ctMake(0.0f, 0.0f);
                        CUBLAS_CGEMM(h, CUBLAS_OP_N, CUBLAS_OP_C,
                            n_amplitudes_, n_amplitudes_, nch * n_polar_,
                            &alpha, d_scaled, n_amplitudes_, d_scaled, n_amplitudes_,
                            &beta, d_m, n_amplitudes_);
                        if (gpu == primary_dev_) {
                            axpyComplex(d_phsp_matrix_, d_m, ctMake(1.0f, 0.0f),
                                        n_amplitudes_ * n_amplitudes_);
                        } else {
                            ctpwa::ensure_device(primary_dev_);
                            ctComplex* d_temp;
                            cudaMalloc(&d_temp, (size_t)n_amplitudes_ * n_amplitudes_ * sizeof(ctComplex));
                            cudaMemcpyPeer(d_temp, primary_dev_, d_m, gpu,
                                (size_t)n_amplitudes_ * n_amplitudes_ * sizeof(ctComplex));
                            axpyComplex(d_phsp_matrix_, d_temp, ctMake(1.0f, 0.0f),
                                        n_amplitudes_ * n_amplitudes_);
                            cudaFree(d_temp);
                            cudaSetDevice(gpu);
                        }
                        cudaFree(d_scaled);
                        cudaFree(d_m);
                    }

                    // double 累加（cast → ZGEMM）
                    {
                        cuDoubleComplex* d_batch_d;
                        cudaMalloc(&d_batch_d, (size_t)nTot * sizeof(cuDoubleComplex));
                        if (d_batch_up != nullptr) {
                            // double .so 上 cuDoubleComplex 与 ctComplex 布局相同:
                            // 直接 D2D 拷贝上转段（值=double 原生, 与 cast kernel 等价）
                            cudaMemcpy(d_batch_d, d_batch_up,
                                (size_t)nTot * sizeof(cuDoubleComplex),
                                cudaMemcpyDeviceToDevice);
                        } else {
                            int grid2 = (nTot + 255) / 256;
                            CTPWA_LAUNCH(castPhspBatchToDoubleKernel, grid2, 256, 0, d_batch[gpu], d_batch_d, nTot);
                        }
                        // 与 float32 分支**同一归一化**：A ← A·sqrt(w/W_total)。
                        // 之前漏了这一步，double 矩阵是 Σ A A^H（未加权/未除 W），
                        // 下游又乘回 W_total → W_rec 大 N 倍、有 phsp_weights 时整个
                        // interference_rec 变成无权重矩阵。
                        {
                            const double* d_w = (gpu < (int)phsp_weights_.size())
                                ? phsp_weights_[gpu] : nullptr;
                            scalePhspAmpsDouble(d_batch_d, d_w ? d_w + c0 : nullptr,
                                nch, n_polar_, n_amplitudes_, 1.0 / W_total);
                        }
                        cuDoubleComplex* d_md;
                        cudaMalloc(&d_md, (size_t)n_amplitudes_ * n_amplitudes_ * sizeof(cuDoubleComplex));
                        cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
                        cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);
                        cublasZgemm(h, CUBLAS_OP_N, CUBLAS_OP_C,
                            n_amplitudes_, n_amplitudes_, nch * n_polar_,
                            &alpha, d_batch_d, n_amplitudes_, d_batch_d, n_amplitudes_,
                            &beta, d_md, n_amplitudes_);
                        if (gpu == primary_dev_) {
                            axpyZComplex(d_phsp_matrix_double_, d_md,
                                         n_amplitudes_ * n_amplitudes_);
                        } else {
                            ctpwa::ensure_device(primary_dev_);
                            cuDoubleComplex* d_temp;
                            cudaMalloc(&d_temp, (size_t)n_amplitudes_ * n_amplitudes_ * sizeof(cuDoubleComplex));
                            cudaMemcpyPeer(d_temp, primary_dev_, d_md, gpu,
                                (size_t)n_amplitudes_ * n_amplitudes_ * sizeof(cuDoubleComplex));
                            axpyZComplex(d_phsp_matrix_double_, d_temp,
                                         n_amplitudes_ * n_amplitudes_);
                            cudaFree(d_temp);
                            cudaSetDevice(gpu);
                        }
                        cudaFree(d_batch_d);
                        cudaFree(d_md);
                    }

                    // 释放上转段（本 GPU）与批振幅（含占位 GPU 的 1 事件缓冲）
                    if (d_batch_up != nullptr) cudaFree(d_batch_up);
                    for (int g = 0; g < n_gpus_; ++g) {
                        if (d_batch[g]) { cudaSetDevice(g); cudaFree(d_batch[g]); }
                    }
                    cudaSetDevice(gpu);
                }
                cublasDestroy(h);
            }
            events_offsets_ = saved_ev;
            amp_offsets_ = saved_amp;
            ctpwa::ensure_device(primary_dev_);
            std::cout << "[ctpwa] phsp 矩阵（float32+double）构建完成，phsp 振幅不驻留。"
                      << std::endl;
        } else {
            for (size_t gpu = 0; gpu < d_all_amplitudes_.size(); ++gpu)
            {
                int nPhsp = events_[gpu][0];
                if (nPhsp == 0) continue;
                cudaSetDevice(gpu);

                // resident 模式构造期 B̄ 构建。float 模式下 A 为 float2（8B/元素），
                // B̄ 保持 double（phsp_factor 恒走 double 轨道）→ 按块上转 double 再
                // 走原 scale+CGEMM（峰值 = 单块，与 getHessian 分块重建同模式）。
                const int kPhspBatch = phspGramBatchEvents();
                for (int c0 = 0; c0 < nPhsp; c0 += kPhspBatch) {
                    int nch = std::min(kPhspBatch, nPhsp - c0);
                    int nch_tot = nch * n_polar_ * n_amplitudes_;
                    const double* d_w = (gpu < phsp_weights_.size()) ? phsp_weights_[gpu] : nullptr;
                    ctComplex* d_phsp_scaled;
                    ctpwaCheckedMalloc((void**)&d_phsp_scaled,
                        (size_t)nch_tot * sizeof(ctComplex), "phsp Gram scaled batch");
                    if (float_amps_ && !ctIsFloatComplex) {
                        // float2 A 块 → double 块（A 元素偏移 = c0*nPolar*nAmp，单位元素数）
                        const float2* srcF = reinterpret_cast<const float2*>(d_all_amplitudes_[gpu])
                            + (size_t)c0 * n_polar_ * n_amplitudes_;
                        CTPWA_LAUNCH(castF2ToDouble2Kernel, (nch_tot + 255) / 256, 256, 0, 
                            srcF, reinterpret_cast<cuDoubleComplex*>(d_phsp_scaled), nch_tot);
                    } else {
                        cudaMemcpy(d_phsp_scaled,
                                   d_all_amplitudes_[gpu] + (size_t)c0 * n_polar_ * n_amplitudes_,
                                   nch_tot * sizeof(ctComplex), cudaMemcpyDeviceToDevice);
                    }
                    int grid = (nch_tot + 255) / 256;
                    CTPWA_LAUNCH(scalePhspAmpsKernel, grid, 256, 0, d_phsp_scaled,
                        d_w ? d_w + c0 : nullptr, nch, n_polar_, n_amplitudes_,
                        1.0 / W_total, 0);

                    ctComplex* d_phsp_gpu;
                    ctpwaCheckedMalloc((void**)&d_phsp_gpu,
                        (size_t)n_amplitudes_ * n_amplitudes_ * sizeof(ctComplex),
                        "phsp Gram batch matrix");
                    cublasHandle_t h;
                    cublasCreate(&h);
                    ctComplex alpha = ctMake(1.0f, 0.0f);
                    ctComplex beta  = ctMake(0.0f, 0.0f);
                    CUBLAS_CGEMM(h, CUBLAS_OP_N, CUBLAS_OP_C, n_amplitudes_, n_amplitudes_, nch * n_polar_,
                        &alpha, d_phsp_scaled, n_amplitudes_, d_phsp_scaled, n_amplitudes_,
                        &beta, d_phsp_gpu, n_amplitudes_);
                    cublasDestroy(h);
                    cudaFree(d_phsp_scaled);

                    if (gpu == (size_t)primary_dev_) {
                        ctComplex one = ctMake(1.0f, 0.0f);
                        axpyComplex(d_phsp_matrix_, d_phsp_gpu, one, n_amplitudes_ * n_amplitudes_);
                        cudaFree(d_phsp_gpu);
                    } else {
                        ctpwa::ensure_device(primary_dev_);
                        ctComplex* d_temp;
                        cudaMalloc(&d_temp, n_amplitudes_ * n_amplitudes_ * sizeof(ctComplex));
                        cudaMemcpyPeer(d_temp, primary_dev_, d_phsp_gpu, gpu,
                            n_amplitudes_ * n_amplitudes_ * sizeof(ctComplex));
                        cudaSetDevice(gpu);
                        cudaFree(d_phsp_gpu);
                        ctpwa::ensure_device(primary_dev_);
                        ctComplex one = ctMake(1.0f, 0.0f);
                        axpyComplex(d_phsp_matrix_, d_temp, one, n_amplitudes_ * n_amplitudes_);
                        cudaFree(d_temp);
                        cudaSetDevice(gpu);
                    }
                }
            }
        }
        // // cudaFree(d_phsp);
        // // 打印矩阵d_phsp
        // ctComplex* h_phsp = new ctComplex[n_amplitudes_ * n_amplitudes_];
        // cudaMemcpy(h_phsp, d_phsp_matrix_, n_amplitudes_* n_amplitudes_ * sizeof(ctComplex), cudaMemcpyDeviceToHost);
        // std::cout << "Phase space matrix (phsp*phsp^T):" << std::endl;
        // for (int i = 0; i < n_amplitudes_; ++i) {
        //     for (int j = 0; j < n_amplitudes_; ++j) {
        //         std::cout << "  [" << i << "][" << j << "] = " << h_phsp[i * n_amplitudes_ + j].x << " + " << h_phsp[i * n_amplitudes_ + j].y << "j" << std::endl;
        //     }
        // }
        // delete[] h_phsp;

        // 静态 SL 缩减：仅保留含自由参数（含 trans/var_equal 关联）链组合的 slamp，
        // 固定链组合的 A 已在构造期物化 → 释放其 slamp（大样本可省数 GB/卡）。
        // 逃生阀 CTPWA_KEEP_SLAMP=1 可关闭。
        if (!amp_calc_.empty()) amp_calc_.releaseStaticCasData();

        std::cout << "Number of GPUs available: " << n_gpus_ << std::endl;
        // n_gls_ 是 SL 振幅条数（= 进入 cublas 的列数），不是分波数：多 SL 的链上
        // 两者差别很大（如 51 个分波 → 105 条 SL 振幅），而 writeResult 的模板桶/
        // 局部内存是按**分波数** nSLvectors_.size() 选的，所以两个都打出来。
        std::cout << "Number of partial waves: " << nSLvectors_.size()
                  << " (SL amplitudes: " << n_gls_ << ")" << std::endl;
        std::cout << "Number of res parameters: " << params_.nFreeTheta() << std::endl;
        std::cout << "Initialization complete." << std::endl;

        // 恢复主设备为当前设备：构造期大量 per-GPU 循环（SL 表/振幅/权重上传/
        // resident B̄ 构建）会把 raw current device 留在最后一张卡。torch 的
        // current-device 缓存在其首次 CUDA 调用时从 raw device 懒初始化——
        // 漏掉此恢复会让随后的 torch.device('cuda')/tensor 落到非主 GPU
        // （getNLL/getHessian 的 params 位置检查触发，测试套件双卡实测必现）。
        ctpwa::ensure_device(primary_dev_);
    
}

void analysis::initializeDecayChains()
{

        // Use DecayInfo to build coupling matrix from config
        DecayInfo info(config_parser_);
        if (!info.isValid()) return;

        // Copy data from DecayInfo
        amplitude_names_   = info.amplitudeNames();
        resonance_names_   = info.resonanceNames();
        chains_info_       = info.chainInfos();
        n_amplitudes_      = info.nAmplitudes();
        nSLvectors_        = info.nSLvectors();

        // Set coupling matrix + param names on Parameters
        if (info.hasCouplingMatrix()) {
            params_.setCouplingMatrix(info.couplingMatrix());
            auto all_names = info.paramNames();
            const auto& rnames = info.resonanceParamNames();
            all_names.insert(all_names.end(), rnames.begin(), rnames.end());
            params_.setParamNames(all_names);
            // 耦合参数名段长度（theta 段重建时保留）
            n_coupling_names_ = static_cast<int>(info.paramNames().size());

            // std::cout << "Coupling matrix: " << info.couplingMatrix().n_amps
            //           << " amplitudes → " << info.couplingMatrix().n_free
            //           << " free coupling params ("
            //           << info.couplingMatrix().n_chain_free << " chain + "
            //           << info.couplingMatrix().n_step_free << " step)" << std::endl;
        } else {
            // Legacy constraint mode
            auto constraints = config_parser_.getConstraints();
            std::vector<std::vector<int>> con_trans_id;
            std::vector<std::vector<std::complex<double>>> con_trans_values;
            for (const auto& constraint : constraints) {
                std::vector<std::vector<int>> amp_ids_con;
                for (const auto& amp_name : constraint.names) {
                    std::vector<int> amp_ids;
                    for (int i = 0; i < (int)amplitude_names_.size(); ++i)
                        if (amplitude_names_[i].find(amp_name) != std::string::npos)
                            amp_ids.push_back(i);
                    amp_ids_con.push_back(amp_ids);
                }
                if (amp_ids_con.empty() || amp_ids_con[0].empty()) continue;
                for (size_t i = 0; i < amp_ids_con[0].size(); ++i) {
                    std::vector<int> combination;
                    for (size_t j = 0; j < amp_ids_con.size(); ++j)
                        combination.push_back(amp_ids_con[j][i]);
                    con_trans_id.push_back(combination);
                    std::vector<std::complex<double>> values = {std::complex<double>(1.0, 1.0)};
                    for (const auto& val : constraint.values) values.push_back(val);
                    con_trans_values.push_back(values);
                }
            }
            params_.initialize(n_amplitudes_ - static_cast<int>(con_trans_id.size()),
                               con_trans_id, con_trans_values);
        }

        // vspace mode: direct amplitude mapping, preserving trans folding
        if (fit_mode_ == 1) {
            const auto& cm = info.couplingMatrix();

            CouplingMatrixResult id;
            id.n_amps = n_amplitudes_;
            id.n_step_free = 0;
            id.amp_step_params.resize(n_amplitudes_);
            id.amp_chain.resize(n_amplitudes_);
            // 保留 buildWithTrans 计算的 trans ratio（非折叠=1.0，折叠=-1.0 等）
            id.amp_chain_ratio = cm.amp_chain_ratio;

            // 判断是否需要"拆分"：检查是否有 active chain 包含多个 owner 振幅
            // 这发生在同一共振态有多个 SL 组合时
            std::map<int, std::vector<int>> chain_owners;
            for (int ai = 0; ai < n_amplitudes_; ++ai) {
                if (cm.amp_chain[ai] < 0) continue;   // 不按 ratio 过滤: trans 值可为 1.0
                chain_owners[cm.amp_chain[ai]].push_back(ai);
            }

            bool needs_split = false;
            for (const auto& [ci, owners] : chain_owners) {
                if (owners.size() > 1) { needs_split = true; break; }
            }

            // free param → 组代表链振幅名(折叠链共享参数时显示代表链)
            std::map<int, std::string> free_idx_to_name;

            if (!needs_split) {
                // 简单情况：每个 active chain 恰好 1 个 owner 振幅
                // buildWithTrans 的结果可直接用于 vspace
                id.amp_chain = cm.amp_chain;
                id.chain_names = cm.chain_names;
                id.n_chain_free = cm.n_chain_free;
                id.n_free = id.n_chain_free;
            } else {
                // 复杂情况：同一共振态有多个 SL 组合，需要拆分 chain param
                // 使用 amp_map 恢复原始 chain 归属和链内排序
                std::map<std::string, std::vector<int>> orig_chain_amps;
                for (const auto& am : cm.amp_map) {
                    orig_chain_amps[am.chain_key].push_back(am.amp_idx);
                }
                for (auto& [ck, amps] : orig_chain_amps) {
                    std::sort(amps.begin(), amps.end());
                }

                // 折叠判定: 映射到同一 active index 的多条原链共享参数。
                // 注意不能用 ratio ≠ 1 判定 —— trans 约束值本身可为 1.0,
                // 折叠链的 ratio(=约束值) 与「每条链一个参数」时的 ratio 都是 1.0。
                // 组内第一条原链(按字符串序, 即展开顺序)的振幅分配独立参数,
                // 同组其余原链按位置对齐到组代表的对应参数。
                int next_free = 0;
                std::map<int, std::map<std::string, std::vector<int>>> groups; // ac → ck → amps
                for (const auto& [ck, amps] : orig_chain_amps) {
                    if (amps.empty() || cm.amp_chain[amps[0]] < 0) continue;
                    groups[cm.amp_chain[amps[0]]][ck] = amps;
                }
                // active_chain → 组代表(第一条原链)各位置对应的 free param index
                std::map<int, std::vector<int>> ac_free_indices;

                for (const auto& [ac, ck_amps] : groups) {
                    for (const auto& [ck, amps] : ck_amps) {
                        bool is_owner = (ck == ck_amps.begin()->first);
                        const std::vector<int>& owner_amps = ck_amps.begin()->second;
                        if (is_owner) {
                            for (int ai : amps) {
                                id.amp_chain[ai] = next_free;
                                ac_free_indices[ac].push_back(next_free);
                                next_free++;
                            }
                        } else {
                            const auto& free_indices = ac_free_indices[ac];
                            for (size_t pos = 0; pos < amps.size()
                                 && pos < free_indices.size() && pos < owner_amps.size(); ++pos) {
                                id.amp_chain[amps[pos]] = free_indices[pos];
                            }
                        }
                    }
                }

                id.n_chain_free = next_free;
                id.n_free = next_free;
            }

            params_.setCouplingMatrix(id);

            // 参数名 = 最后一个映射到该 free param 的振幅名（与历史行为一致：
            // 折叠链共享参数时均指向同一个 index, 后写覆盖 = 展开序靠后者）
            for (int ai = 0; ai < n_amplitudes_; ++ai) {
                int f = id.amp_chain[ai];
                if (f < 0) continue;
                free_idx_to_name[f] = amplitude_names_[ai];
            }
            std::vector<std::string> vspace_names;
            for (int idx = 0; idx < id.n_chain_free; ++idx) {
                auto it = free_idx_to_name.find(idx);
                vspace_names.push_back(it != free_idx_to_name.end() ? it->second : "");
            }
            const auto& rnames = info.resonanceParamNames();
            vspace_names.insert(vspace_names.end(), rnames.begin(), rnames.end());
            params_.setParamNames(vspace_names);
        }
    
}

torch::Tensor analysis::getIntensity(torch::Tensor momenta, torch::Tensor params)
{

        TORCH_CHECK(initialized_, "analysis not initialized");
        TORCH_CHECK(momenta.defined() && momenta.dim() == 3 &&
                momenta.size(2) == 4,
            "getIntensity: momenta must have shape [N, n_final, 4] "
            "(E, px, py, pz)");
        TORCH_CHECK(momenta.scalar_type() == torch::kFloat64,
            "getIntensity: momenta must be float64");
        TORCH_CHECK(momenta.device().is_cpu(),
            "getIntensity: momenta must be a CPU tensor");
        TORCH_CHECK(ctpwa_is_cuda_like(params),
            "getIntensity: params 必须位于当前后端的设备上（CPU 构建用 CPU 张量）");
        TORCH_CHECK(params.dtype() == torch::kFloat64 && params.dim() == 1,
            "getIntensity: params must be 1-D float64");
        TORCH_CHECK(ctpwa_on_primary(params, primary_dev_),
            "getIntensity: params 必须位于主设备 (primary=" +
                std::to_string(primary_dev_) + ")");
        TORCH_CHECK(params.numel() == getNParams(),
            "getIntensity: params size mismatch: got ", params.numel(),
            ", expected ", getNParams());

        const auto& data_order = config_parser_.getDataOrder();
        TORCH_CHECK(momenta.size(1) == static_cast<int64_t>(data_order.size()),
            "getIntensity: momenta final-state count ", momenta.size(1),
            " != Data.order size ", data_order.size());

        const int64_t N = momenta.size(0);
        auto opts = torch::TensorOptions().dtype(torch::kFloat64);
        if (N == 0) return torch::empty({ 0 }, opts);

        auto mom = momenta.contiguous();
        auto acc = mom.accessor<double, 3>();

        // N < n_gpus 时只启用前 N 个 GPU；每个启用 GPU 至少 1 个事例。
        const int n_gpu = static_cast<int>(
            std::min<int64_t>(static_cast<int64_t>(n_gpus_), N));
        std::vector<int> counts(static_cast<size_t>(n_gpu), 0);
        const int base = static_cast<int>(N / n_gpu);
        const int rem = static_cast<int>(N % n_gpu);
        for (int g = 0; g < n_gpu; ++g) counts[g] = base + (g < rem ? 1 : 0);

        std::vector<std::map<std::string, std::vector<LorentzVector>>>
            Vp4(static_cast<size_t>(n_gpu));
        int ev_off = 0;
        for (int g = 0; g < n_gpu; ++g) {
            const int ng = counts[g];
            for (size_t a = 0; a < data_order.size(); ++a) {
                auto& v = Vp4[g][data_order[a]];
                v.resize(static_cast<size_t>(ng));
                for (int e = 0; e < ng; ++e) {
                    const int64_t i = ev_off + e;
                    v[e] = LorentzVector(acc[i][a][0], acc[i][a][1],
                        acc[i][a][2], acc[i][a][3]);
                }
            }
            ev_off += ng;
        }

        auto saved_ev = events_offsets_;
        auto saved_amp = amp_offsets_;

        AmpCalc tmp;
        std::vector<ctComplex*> amps;
        std::vector<ctComplex*> upcast;
        std::vector<ctComplex*> d_vec(static_cast<size_t>(n_gpu), nullptr);
        std::vector<double*> d_out(static_cast<size_t>(n_gpu), nullptr);
        std::vector<torch::Tensor> v_hold;
        bool released = false;

        auto cleanup = [&]() {
            for (int g = 0; g < static_cast<int>(upcast.size()); ++g)
                if (upcast[g]) { cudaSetDevice(g); cudaFree(upcast[g]); }
            upcast.clear();
            for (int g = 0; g < static_cast<int>(amps.size()); ++g)
                if (amps[g]) { cudaSetDevice(g); cudaFree(amps[g]); }
            amps.clear();
            for (int g = 0; g < static_cast<int>(d_out.size()); ++g)
                if (d_out[g]) { cudaSetDevice(g); cudaFree(d_out[g]); }
            d_out.clear();
            d_vec.clear();
            if (!released) {
                tmp.release();
                released = true;
            }
            events_offsets_ = saved_ev;
            amp_offsets_ = saved_amp;
            ctpwa::ensure_device(primary_dev_);
        };

        try {
            events_offsets_.assign(static_cast<size_t>(n_gpu),
                std::vector<int>{});
            amp_offsets_.assign(static_cast<size_t>(n_gpu),
                std::vector<int>{});
            for (int g = 0; g < n_gpu; ++g) {
                events_offsets_[g] = { 0, counts[g] };
                amp_offsets_[g] = { 0, counts[g] * n_polar_ * n_amplitudes_ };
            }

            // 临时 AmpCalc 必须与 analysis 的存储档位一致：calculateAmplitudes
            // 按 float_amps_ 分配 A 表，reComputeAmps 按 tmp.float_out_/dF_float_
            // 选择写入布局；不一致会把 double 写进 float2 表（实测全 NaN）。
            tmp.setFloatOutput(float_amps_);
            tmp.setDFloat(prec_mode_ == PrecMode::Float);

            amps = calculateAmplitudes(Vp4, &tmp, n_gpu);
            TORCH_CHECK(static_cast<int>(amps.size()) == n_gpu,
                "getIntensity: calculateAmplitudes returned ", amps.size(),
                " amplitude tables, expected ", n_gpu);

            torch::Tensor vector, theta;
            std::tie(vector, theta) = params_.splitParams(params);
            torch::Tensor v_ext = freeParamsToAmplitudes(vector);
            TORCH_CHECK(v_ext.numel() == n_amplitudes_,
                "getIntensity: extended vector size ", v_ext.numel(),
                " != n_amplitudes ", n_amplitudes_);
            TORCH_CHECK(theta.numel() == tmp.nFreeResParams(),
                "getIntensity: theta size ", theta.numel(),
                " != free resonance params ", tmp.nFreeResParams());
            if (tmp.nFreeResParams() > 0) {
                tmp.reComputeAmps(amps, theta.data_ptr<double>(),
                    n_amplitudes_, events_offsets_, amp_offsets_, n_polar_,
                    primary_dev_);
            }

            v_hold.reserve(static_cast<size_t>(n_gpu));
            for (int g = 0; g < n_gpu; ++g) {
                v_hold.push_back(v_ext.to(ctpwa_dev(g)));
                d_vec[g] =
                    reinterpret_cast<ctComplex*>(v_hold.back().data_ptr());
            }

            for (int g = 0; g < n_gpu; ++g) {
                cudaSetDevice(g);
                ctComplex* a = amps[g];
                if (float_amps_ && !ctIsFloatComplex && a) {
                    const size_t n_el = static_cast<size_t>(counts[g]) *
                        static_cast<size_t>(n_polar_) *
                        static_cast<size_t>(n_amplitudes_);
                    ctComplex* dbl = upcastAmpSegToDouble(a, n_el);
                    upcast.push_back(dbl);
                    a = dbl;
                }
                cudaError_t e = cudaMalloc(&d_out[g],
                    static_cast<size_t>(counts[g]) * sizeof(double));
                TORCH_CHECK(e == cudaSuccess,
                    "getIntensity: cudaMalloc(out) 失败: ",
                    cudaGetErrorString(e));
                computeTotalIntensity(a, d_vec[g], d_out[g], counts[g],
                    n_amplitudes_, n_polar_);
            }

            torch::Tensor out = torch::empty({ N }, opts);
            double* host = out.data_ptr<double>();
            int off = 0;
            for (int g = 0; g < n_gpu; ++g) {
                cudaSetDevice(g);
                cudaError_t e = cudaMemcpy(host + off, d_out[g],
                    static_cast<size_t>(counts[g]) * sizeof(double),
                    cudaMemcpyDeviceToHost);
                TORCH_CHECK(e == cudaSuccess,
                    "getIntensity: cudaMemcpy(D2H) 失败: ",
                    cudaGetErrorString(e));
                off += counts[g];
            }
            cleanup();
            return out;
        } catch (...) {
            cleanup();
            throw;
        }
    
}

