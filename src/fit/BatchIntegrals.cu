// src/BatchIntegrals.cu — 见 include/BatchIntegrals.h
#include <BatchIntegrals.h>

#include "BackendHelpers.h"
#include "DataReader.h"
#include "DeviceGuard.h"
#include "MixedPrecision.h"
#include "TruthIntegrals.h"

#include <TFile.h>
#include <TTree.h>

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>

namespace ctpwa {

double computeFileBatchIntegrals(
    const BatchIntegralCtx& ctx,
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
        const auto& data_files = (*ctx.data_files);
        TORCH_CHECK(data_files.count(file_key) > 0,
            std::string("No data file '") + file_key + "' in config");

        std::vector<std::string> particles_names;
        for (const auto& p : (*ctx.particles)) particles_names.push_back(p.name);

        // 每事件行数必须与读取器 readMomentaFromDat 一致：那边用
        // particleNames.size()（= 传进去的 data_order 大小）作为"每事件行数"。
        // 旧实现用 lines / (*ctx.particles).size()——Particles 含初态时口径不一致
        // （本例 4 vs 3），只会覆盖 data_order/Particles 比例的行，静默少算
        // （issue #3 ②）。
        const auto& data_order = (*ctx.data_order);
        const int ev_lines = data_order.empty() ? (int)(*ctx.particles).size()
                                               : (int)data_order.size();
        const auto& fileinfo = data_files.at(file_key);
        TORCH_CHECK(fileinfo.size() >= 2, "[", file_key, "] 需要 [类型, 文件] 两项");
        const std::string file_type = fileinfo[0];
        const std::string file = fileinfo[1];
        int total_events = 0;
        if (file_type == "ROOT" || file_type == "root") {
            // ROOT 输入: 事例数 = TTree 的 entries（变量分支约定与
            // readMomentaFromDat 一致: [ROOT, 文件, TTree, (分支名...)]）
            TORCH_CHECK(fileinfo.size() >= 3,
                "[", file_key, "] ROOT 输入需要 TTree 名: [ROOT, 文件, TTree]");
            TFile* f = TFile::Open(file.c_str(), "READ");
            TORCH_CHECK(f && !f->IsZombie(),
                "[", file_key, "] 打不开 ROOT 文件 ", file);
            TTree* t = (TTree*)f->Get(fileinfo[2].c_str());
            TORCH_CHECK(t != nullptr,
                "[", file_key, "] ROOT 文件 ", file, " 里找不到 TTree ", fileinfo[2]);
            total_events = (int)t->GetEntries();
            f->Close();
            delete f;
        } else {
            std::ifstream f(file);
            std::string line;
            int lines = 0;
            while (std::getline(f, line)) if (!line.empty()) ++lines;
            TORCH_CHECK(ev_lines > 0 && lines % ev_lines == 0,
                "[", file_key, "] 行数 ", lines, " 不是每事件行数 ", ev_lines,
                " 的整数倍（config 的 Data.order / Particles 与文件不匹配）");
            total_events = lines / ev_lines;
        }
        if (std::getenv("CTPWA_VERBOSE"))
            std::cout << file_key << ": " << total_events << " events" << std::endl;
        // 调用方按接口约定预零所有输出。空样本直接返回，避免为 0 个事件
        // 分配整套 batch workspace，也避免常数权重展开后被误判为 Σw=0。
        if (total_events == 0) return 0.0;

        // 逐事件权重（可选）: <file_key>_weights
        std::vector<double> h_weights;
        double weight_sum = static_cast<double>(total_events);
        const std::string weights_key = file_key + "_weights";
        if (data_files.count(weights_key) > 0) {
            h_weights = readWeightValuesFromFile(data_files.at(weights_key), total_events);
            TORCH_CHECK(h_weights.size() == static_cast<size_t>(total_events),
                "[", weights_key, "] 权重数 ", h_weights.size(),
                " 与 [", file_key, "] 事件数 ", total_events, " 不一致");
            weight_sum = 0.0;
            int n_neg = 0;
            for (double w : h_weights) {
                TORCH_CHECK(std::isfinite(w), "[", weights_key, "] 含非有限权重");
                if (w < 0.0) ++n_neg;
                weight_sum += w;
            }
            TORCH_CHECK(weight_sum > 0.0,
                "[", weights_key, "] Σw = ", weight_sum, " ≤ 0, 无法归一化");
            std::cout << "[" << weights_key << "] Σw = " << weight_sum
                      << " (N = " << total_events << ")" << std::endl;
            if (n_neg > 0)
                std::cerr << "Warning: [" << weights_key << "] 含 " << n_neg
                          << " 个负权重（计入 Σw = " << weight_sum << "）" << std::endl;
        }

        // 真值积分的 θ：本路径重建振幅用的是配置粒子对象（m0/Γ0 = 配置初值），
        // 所以必须把当前自由参数显式带进去（见 calculateAmplitudes 的 theta_override）。
        const std::vector<double> h_theta = ctx.current_theta_host();
        const double* d_theta = h_theta.empty() ? nullptr : h_theta.data();
        if (!h_theta.empty() && std::getenv("CTPWA_DEBUG_THETA") != nullptr)
            std::cout << "[" << file_key << "] theta override (n_free="
                      << h_theta.size() << ")" << std::endl;

        // Batched accumulation（CTPWA_TRUTH_BATCH 可覆盖批大小: 小显存/极大数据集调优，
        // 也是 batching 路径的回归开关）
        int batch_size = 100000;
        if (const char* env = std::getenv("CTPWA_TRUTH_BATCH")) {
            const int v = std::atoi(env);
            if (v > 0) batch_size = v;
            std::cout << "[" << file_key << "] batch_size = " << batch_size
                      << " (CTPWA_TRUTH_BATCH)" << std::endl;
        }
        auto saved_ev = (*ctx.events_offsets), saved_amp = (*ctx.amp_offsets);
        // 批扫描会临时改写 (*ctx.events_offsets)/(*ctx.amp_offsets)；任何异常（文件缺失、
        // cudaMalloc 失败、kernel 报错）都必须还原，否则 analysis 对象后续的
        // getNLL/writeResult 会拿批偏移去索引驻留表。
        auto offset_guard = makeScopeExit([&]() {
            (*ctx.events_offsets) = saved_ev;
            (*ctx.amp_offsets) = saved_amp;
            ctpwa::ensure_device(ctx.primary_dev);
        });
        // 旧实现在 computeBranchingFractions 内每次调用都 cudaMalloc/cudaFree
        // 两个大缓冲（ngls×1e5×npolar×16B ≈ 139 MB）：15M 事件的 truth = 150 batch
        // × (1+2·n_free) 次调用 → 上千次大块 alloc/free → 碎片化 → 分配失败
        // → nullptr 进 kernel → illegal memory access（issue #3 ①）。
        auto memprobe = [&](const char* tag, int start) {
            if (std::getenv("CTPWA_MEMPROBE") == nullptr) return;
            size_t mf = 0, mt = 0; cudaMemGetInfo(&mf, &mt);
            printf("[memprobe] %-22s start=%7d free=%.1f MiB\n", tag, start,
                   (double)mf / 1048576.0);
        };
        memprobe("enter", -1);
        const size_t ws_cplx = (size_t)batch_size * (size_t)std::max(1, ctx.n_polar);
        const size_t ws_mat  = (size_t)std::max(1, ctx.n_amplitudes) * ws_cplx;
        std::vector<ctComplex*> ws_complex(ctx.n_gpus, nullptr);
        std::vector<ctComplex*> ws_matrix(ctx.n_gpus, nullptr);
        std::vector<ctComplex*> ws_upcast(ctx.n_gpus, nullptr);
        std::vector<double*> ws_weights(ctx.n_gpus, nullptr);   // 仅加权时分配
        // 分配到一半失败（OOM）时也要释放已分配的部分；正常路径在末尾置空。
        auto ws_guard = makeScopeExit([&]() {
            for (int g = 0; g < ctx.n_gpus; ++g) {
                ctpwa::ensure_device(g);
                if (ws_complex[g]) cudaFree(ws_complex[g]);
                if (ws_matrix[g])  cudaFree(ws_matrix[g]);
                if (ws_upcast[g])  cudaFree(ws_upcast[g]);
                if (ws_weights[g]) cudaFree(ws_weights[g]);
            }
            ctpwa::ensure_device(ctx.primary_dev);
        });
        // 当前批的 A 表；异常（kernel 报错等）时也要释放。
        MultiGpuStorage<ctComplex> d_batch_amps;
        auto batch_guard = makeScopeExit([&]() {
            d_batch_amps.clear();
            ctpwa::ensure_device(ctx.primary_dev);
        });
        for (int g = 0; g < ctx.n_gpus; ++g) {
            ctpwa::ensure_device(g);
            TORCH_CHECK(cudaMalloc(&ws_complex[g], ws_cplx * sizeof(ctComplex)) == cudaSuccess,
                "[", file_key, "] workspace cudaMalloc 失败 (",
                ws_cplx * sizeof(ctComplex), " B)");
            TORCH_CHECK(cudaMalloc(&ws_matrix[g], ws_mat * sizeof(ctComplex)) == cudaSuccess,
                "[", file_key, "] workspace cudaMalloc 失败 (",
                ws_mat * sizeof(ctComplex), " B)");
            if (ctx.float_amps && !ctIsFloatComplex) {
                TORCH_CHECK(cudaMalloc(&ws_upcast[g], ws_mat * sizeof(cuDoubleComplex)) == cudaSuccess,
                    "[", file_key, "] upcast workspace cudaMalloc 失败 (",
                    ws_mat * sizeof(cuDoubleComplex), " B)");
            }
            if (!h_weights.empty()) {
                TORCH_CHECK(cudaMalloc(&ws_weights[g], batch_size * sizeof(double)) == cudaSuccess,
                    "[", file_key, "_weights] workspace cudaMalloc 失败 (",
                    batch_size * sizeof(double), " B)");
            }
        }

        for (int start = 0; start < total_events; start += batch_size) {
            int n_batch = std::min(batch_size, total_events - start);

            auto Vp4_batch = readMomentaFromDat(data_files.at(file_key),
                data_order, particles_names, n_batch, start);

            // Split across GPUs. 尾批/小样本可能少于 GPU 数：只给有事件的
            // GPU 建 map，AmpCasDecay::computeSLAmps 不允许空 map。正常大批次
            // active_gpus==ctx.n_gpus，数据划分、workspace 和热循环完全不变。
            const int active_gpus = std::min(ctx.n_gpus, n_batch);
            std::vector<int> batch_ev_per_gpu(active_gpus, 0);
            int base_b = n_batch / active_gpus, rem_b = n_batch % active_gpus;
            for (int g = 0; g < active_gpus; ++g)
                batch_ev_per_gpu[g] = base_b + (g < rem_b ? 1 : 0);

            std::vector<std::map<std::string, std::vector<LorentzVector>>> Vp4_tpg(active_gpus);
            int ev_off = 0;
            for (int g = 0; g < active_gpus; ++g) {
                int nev = batch_ev_per_gpu[g];
                for (const auto& [k, v] : Vp4_batch)
                    Vp4_tpg[g][k].assign(v.begin() + ev_off, v.begin() + ev_off + nev);
                ev_off += nev;
            }

            std::vector<std::vector<int>> t_ev_off(active_gpus), t_amp_off(active_gpus);
            for (int g = 0; g < active_gpus; ++g) {
                int nev = batch_ev_per_gpu[g];
                t_ev_off[g] = {0, nev};
                t_amp_off[g] = {0, nev * ctx.n_polar * ctx.n_amplitudes};
            }
            // 每 GPU 一段连续权重，切分口径与 batch_ev_per_gpu / Vp4_tpg 一致
            if (!h_weights.empty()) {
                int ev_off_w = 0;
                for (int g = 0; g < active_gpus; ++g) {
                    int nev = batch_ev_per_gpu[g];
                    ctpwa::ensure_device(g);
                    TORCH_CHECK(cudaMemcpy(ws_weights[g],
                            h_weights.data() + start + ev_off_w,
                            nev * sizeof(double), cudaMemcpyHostToDevice) == cudaSuccess,
                        "[", weights_key, "] cudaMemcpy 失败 (start=", start,
                        ", gpu=", g, ", n=", nev, ")");
                    ev_off_w += nev;
                }
            }

            (*ctx.events_offsets) = t_ev_off; (*ctx.amp_offsets) = t_amp_off;
            d_batch_amps = ctx.calculate_amplitudes(Vp4_tpg, active_gpus, d_theta);
            memprobe("after-amps", start);

            // Center (+ square)
            computeTruthIntegrals(ev_center, d_batch_amps.data(), batch_ev_per_gpu,
                out_partial, out_scattering, out_square, npartials,
                ws_complex, ws_matrix, ws_upcast, ws_weights,
                (*ctx.n_slvectors), ctx.float_amps, ctx.n_polar, ctx.n_amplitudes,
                ctx.primary_dev);

            // Jacobian perturbations (reuse same batch amplitudes)
            if (out_p_partial != nullptr) {
                for (int j = 0; j < (int)ev_perturbed_p.size(); ++j) {
                    computeTruthIntegrals(ev_perturbed_p[j], d_batch_amps.data(), batch_ev_per_gpu,
                        out_p_partial + j * npartials,
                        out_p_scattering + j * npartials * npartials, nullptr, npartials,
                        ws_complex, ws_matrix, ws_upcast, ws_weights,
                (*ctx.n_slvectors), ctx.float_amps, ctx.n_polar, ctx.n_amplitudes,
                ctx.primary_dev);
                    computeTruthIntegrals(ev_perturbed_m[j], d_batch_amps.data(), batch_ev_per_gpu,
                        out_m_partial + j * npartials,
                        out_m_scattering + j * npartials * npartials, nullptr, npartials,
                        ws_complex, ws_matrix, ws_upcast, ws_weights,
                (*ctx.n_slvectors), ctx.float_amps, ctx.n_polar, ctx.n_amplitudes,
                ctx.primary_dev);
                }
            }

            memprobe("after-integrals", start);
            // Free batch amplitudes
            d_batch_amps.clear();
            memprobe("after-amp-free", start);
        }
        // 统一释放 workspace（单次分配 → 单次释放，不再有上千次 alloc/free）
        for (int g = 0; g < ctx.n_gpus; ++g) {
            ctpwa::ensure_device(g);
            if (ws_complex[g]) { cudaFree(ws_complex[g]); ws_complex[g] = nullptr; }
            if (ws_matrix[g])  { cudaFree(ws_matrix[g]);  ws_matrix[g] = nullptr; }
            if (ws_upcast[g])  { cudaFree(ws_upcast[g]);  ws_upcast[g] = nullptr; }
            if (ws_weights[g]) { cudaFree(ws_weights[g]); ws_weights[g] = nullptr; }
        }
        memprobe("exit(after ws free)", -1);
        (*ctx.events_offsets) = saved_ev; (*ctx.amp_offsets) = saved_amp;
        ctpwa::ensure_device(ctx.primary_dev);   // 恢复主设备（内部多 GPU 循环后 raw 停在最后卡）
        return weight_sum;
    
}

}  // namespace ctpwa
