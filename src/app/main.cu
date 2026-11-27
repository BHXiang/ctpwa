// main.cu — 编译入口 & pybind11 绑定
//
// P1（真 TU）：DeviceManager / Config / PhaseSpace / Info / ResModel / Resonance
// 已经是独立翻译单元，这里只包含头文件。唯一仍以文本方式内联的是 Analysis.cu
// —— 它没有头文件（analysis 只供本 TU 的 pybind 绑定使用）。
//
// 收益：改 Config.cu 只重编 Config.cu + 链接（此前会重编 12.8k 行的整个 TU）；
// 并且编译器第一次能告诉你"谁越界用了谁"。

#include <pybind11/pybind11.h>

#include "Backend.h"   // ctpwa::backend_name() 等

#include <Amplitude.cuh>

#include <DeviceManager.cuh>

#include <ComputeHessian.cuh>

#include <AmpGen.cuh>
#include <ComputeNLL.cuh>
#include <ComputeResults.cuh>
#include <ComputeBF.cuh>
#include <Config.cuh>
#include <PhaseSpace.cuh>
#include <Figure.cuh>
#include <Analysis.h>
#include <Info.cuh>
#include <Parameters.cuh>
#include <SymbolicDiff.cuh>  // Node/deriv for modelDeriv+buildModelAST
#include <ResModel.cuh>
#include <CustomExpr.cuh>
#include <Resonance.cuh>


// ---------------------------------------------------------------------------
// 后端探针：_probe() 供 ctpwa/__init__.py 在 import 时决定用哪个后端。
// 只做 alloc 或 torch.cuda.is_available() 都不够——「驱动看得见卡、但 kernel 起不来」
// （编译架构与卡不匹配、卡被 MIG 切分或占满）必须靠真启动一次才能发现。
// ---------------------------------------------------------------------------
CTPWA_KERNEL void ctpwaProbeKernel(ctComplex* out)
{
    const int i = CTPWA_TID();
    if (i == 0) out[0] = ctMake(1.0, 0.0);
}

static void bind_ctpwa(pybind11::module_& m)
{
    // Ensure torch's Python-side bindings exist before any at::Tensor is
    // returned.  Without this, `import ctpwa` (without a prior `import
    // torch`) can crash inside pybind's at::Tensor type caster.
    pybind11::module_::import("torch");

    m.doc() = "ctpwa";

    // 分配 → 启动 1 个工作项的探针 kernel → 同步 → 读回校验。
    // 任一步失败（无卡 / 驱动不匹配 / 架构不符 / 卡被占）都返回 false，
    // 由 Python 门面回退到 CPU 后端。
    m.def("_probe", []() -> bool {
        ctComplex* d = static_cast<ctComplex*>(ctpwa::alloc(sizeof(ctComplex)));
        if (!d) return false;
        bool ok = true;
        CTPWA_LAUNCH(ctpwaProbeKernel, 1, 1, 0, d);
        CTPWA_SYNC();
#if defined(CTPWA_HAS_CUDA)
        ok = (cudaGetLastError() == cudaSuccess);
        if (ok) {
            ctComplex h = ctMake(0.0, 0.0);
            ok = (cudaMemcpy(&h, d, sizeof(ctComplex), cudaMemcpyDeviceToHost) ==
                  cudaSuccess) && (h.x == 1.0);
        }
#else
        ok = (d[0].x == 1.0);
#endif
        ctpwa::dealloc(d);
        return ok;
    }, "后端自检：分配 + 启动探针 kernel + 读回。失败即视为该后端不可用");

    // 当前 .so 的后端与编译信息（用户报问题时第一手信息；测试据此跳过不适用的用例）
    m.def("backend", &ctpwa::backend_name,
          "编译进本 .so 的后端：'cpu' 或 'cuda'");
    m.def("build_info", []() {
        pybind11::dict d;
        d["backend"] = ctpwa::backend_name();
        d["precision"] = PRECISION_NAME;
        d["complex_bytes"] = (int)sizeof(ctComplex);
        d["has_cuda"] =
#if defined(CTPWA_HAS_CUDA)
            true;
#else
            false;
#endif
        return d;
    }, "本 .so 的构建信息（backend / precision / complex_bytes / has_cuda）");

    // ---- DIYMC phase-space generator (native GENBOD) ----
    pybind11::class_<PhaseSpaceGenerator>(m, "PhaseSpaceGenerator")
        .def(pybind11::init<const std::string&, const std::string&>(),
             pybind11::arg("config_file"), pybind11::arg("mother") = "",
             "PhaseSpaceGenerator(config_file, mother=''): generate flat "
             "n-body phase space from config.yml. Only Particles/Data.order/"
             "DecayChains are read; data files are not opened.")
        .def("mother", &PhaseSpaceGenerator::mother)
        .def("motherMass", &PhaseSpaceGenerator::motherMass)
        .def("setMotherMass", &PhaseSpaceGenerator::setMotherMass,
             pybind11::arg("mass"))
        .def("finalStates", &PhaseSpaceGenerator::finalStates)
        .def("nFinal", &PhaseSpaceGenerator::nFinal)
        .def("weightMax", &PhaseSpaceGenerator::weightMax)
        .def("lastTrials", &PhaseSpaceGenerator::lastTrials)
        .def("generate", &PhaseSpaceGenerator::generate,
             pybind11::arg("n"), pybind11::arg("seed") = 0,
             "generate(n, seed=0): flat phase-space events, float64 CPU "
             "[n, n_final, 4] in (E, px, py, pz), final states in Data.order.")
        .def("generateWeighted", &PhaseSpaceGenerator::generateWeighted,
             pybind11::arg("n"), pybind11::arg("seed") = 0,
             "generateWeighted(n, seed=0): GENBOD events + weights "
             "(events, weights).")
        .def("write",
             [](const PhaseSpaceGenerator& g, const torch::Tensor& events,
                 const std::string& filename, pybind11::object weights) {
                 if (weights.is_none())
                     g.write(events, filename);
                 else
                     g.write(events, filename, weights.cast<torch::Tensor>());
             },
             pybind11::arg("events"), pybind11::arg("filename"),
             pybind11::arg("weights") = pybind11::none(),
             "write(events, filename, weights=None): .dat (E px py pz) or "
             ".root (TTree 'mc' with TLorentzVector branches).");

    pybind11::class_<ChainView>(m, "ChainView")
        .def_readonly("name", &ChainView::name)
        .def_readonly("topology", &ChainView::topology)
        .def_readonly("steps", &ChainView::steps)
        .def_readonly("intermediates", &ChainView::intermediates)
        .def_readonly("amplitude_names", &ChainView::amplitude_names)
        .def("exactchains", &ChainView::exactchains, pybind11::arg("containing") = "",
             "该链全部完整链串(chains_exact 格式); containing 非空时只返回含它的串")
        .def("amplitudes", &ChainView::amplitudes, "该链波名(_LS 格式)")
        .def("counts", &ChainView::counts, "返回 [中间态数, 共振态总数, 完整链串数, 振幅数]")
        .def("print", &ChainView::print);

    pybind11::class_<DecayInfo>(m, "DecayInfo")
        .def(pybind11::init<const std::string&>(), pybind11::arg("config_file") = "config.yml")
        .def("isValid", &DecayInfo::isValid)
        .def("nAmplitudes", &DecayInfo::nAmplitudes)
        .def("nFreeParams", &DecayInfo::nFreeParams)
        .def("amplitudeNames", &DecayInfo::amplitudeNames)
        .def("paramNames", &DecayInfo::paramNames)
        .def("resonanceNames", &DecayInfo::resonanceNames)
        .def("resonanceParamNames", &DecayInfo::resonanceParamNames)
        .def("hasCouplingMatrix", &DecayInfo::hasCouplingMatrix)
        .def("print", &DecayInfo::print, pybind11::arg("level") = 1,
             "0=总览 1=链概览(默认) 2=链明细+完整链串 3=全部振幅")
        .def("summary", &DecayInfo::summary, "只打总览(粒子/链数/振幅数/链串数)")
        .def("chains", &DecayInfo::chains, "分层浏览: 返回 [ChainView, ...]")
        .def("exactchains", &DecayInfo::exactchains,
             pybind11::arg("chain") = -1, pybind11::arg("containing") = "",
             "完整链串(chains_exact 格式); chain<0=全部链, containing 非空时过滤")
        .def("printExactChains", &DecayInfo::printExactChains,
             pybind11::arg("containing") = "",
             "扁平打印全部完整链串(一行一条, 首行 # 计数); 可直接存为 chains_exact 外部文件")
        .def("printParamNames", &DecayInfo::printParamNames,
             pybind11::arg("containing") = "",
             "扁平打印全部拟合参数名(chain×step: 链/步耦合 + 末尾 θ 段; 下标与 parameters.txt 一致)")
        .def("amplitudes", &DecayInfo::amplitudes,
             pybind11::arg("chain") = -1, pybind11::arg("resonance") = "",
             "波名(_LS 格式); chain<0=全部链, resonance 非空时按名字子串过滤");

    pybind11::class_<DeviceManager>(m, "DeviceManager")
        .def(pybind11::init<>())
        .def("detect", &DeviceManager::detect)
        .def("numDevices", &DeviceManager::numDevices)
        .def("hasDevices", &DeviceManager::hasDevices)
        .def("print", &DeviceManager::print)
        .def("deviceName", [](const DeviceManager& dm, int i) {
            return dm.device(i).name; })
        .def("deviceMemoryTotal", [](const DeviceManager& dm, int i) {
            return dm.device(i).total_memory; })
        .def("deviceMemoryFree", [](const DeviceManager& dm, int i) {
            return dm.device(i).free_memory; })
        .def("deviceComputeCapability", [](const DeviceManager& dm, int i) {
            const auto& d = dm.device(i);
            return std::make_pair(d.cc_major, d.cc_minor); })
        .def("estimateMemory", [](const DeviceManager& dm, int n_events,
                                   int n_amps, int n_pol, int n_sl, int n_part,
                                   bool has_bkg) {
            auto m = dm.estimate(n_events, n_amps, n_pol, n_sl, n_part, has_bkg);
            return std::make_pair(m.total_bytes_gpu, m.total_bytes_other); })
        .def("checkCapacity", [](const DeviceManager& dm,
                                  std::vector<int> events_per_gpu,
                                  int n_amps, int n_pol, int n_sl, int n_part,
                                  bool has_bkg) {
            auto r = dm.checkCapacity(events_per_gpu, n_amps, n_pol, n_sl,
                                      n_part, has_bkg);
            return std::make_tuple((int)r.overall, r.failing_device,
                                   r.failing_buffer, r.required_bytes,
                                   r.available_bytes); })
        .def("complexSize", &DeviceManager::complexSize)
        .def("setComplexPrecision", [](DeviceManager& dm, int p) {
            dm.setComplexPrecision(p == 0 ? ComplexPrecision::Float
                                          : ComplexPrecision::Double); })
        .def("complexPrecision", [](const DeviceManager& dm) {
            return (int)dm.complexPrecision(); })
        .def("compiledPrecision", [](const DeviceManager&) {
            return std::string(PRECISION_NAME); })
        .def("setAmplitudeBytes", &DeviceManager::setAmplitudeBytes,
             pybind11::arg("bytes"), "设置运行时驻留振幅 A 的每元素字节数（8/16）")
        .def("amplitudeBytes", &DeviceManager::amplitudeBytes);

    pybind11::class_<analysis>(m, "analysis")
        .def(pybind11::init<const std::string&, int>(),
             pybind11::arg("config_file") = "config.yml",
             pybind11::arg("fit_mode") = 0,
             "analysis(config_file='config.yml', fit_mode=0): fit_mode 0=FREEPARAMS "
             "(chain×step, 默认), 1=VSPACE (逐振幅)。参数化在构造期决定，"
             "需要 VSPACE 必须在此处传 1。")
        .def("getNLL", &analysis::getNLL, pybind11::arg("params"),
             "Compute NLL. params: [real(v), imag(v), theta] float64")
        .def("release", &analysis::release,
             "释放本对象持有的全部设备资源并标记为不可用；析构时自动调用。"
             "长进程反复构造 analysis 时无需依赖进程退出回收显存。")
        .def("setFitMode", &analysis::setFitMode, pybind11::arg("mode"),
             "Set fit mode: 0=FREEPARAMS (chain×step, default), 1=VSPACE (direct amplitudes). "
             "注意：构造后调用改不了参数化（仅改 getNVector/getParamNames 语义），"
             "真正启用 VSPACE 请用 ctpwa.analysis(config, 1)。")
        .def("getFitMode", &analysis::getFitMode)
        .def("getNVector", &analysis::getNVector)
        .def("getNFreeTheta", &analysis::getNFreeTheta)
        .def("getNParams", &analysis::getNParams)
        .def("getParamNames", &analysis::getParamNames)
        .def("getSLVectors", &analysis::getSLVectors)
        .def("getLegends", &analysis::getLegends,
             "config legends 规则按展开链顺序解析出的图例名列表 (getLegends())")
        .def("writeResult",
             [](analysis& a, torch::Tensor params, const std::string& filename,
                int is_saved_weight, std::vector<int> waves, pybind11::object hessian) {
                 torch::Tensor h;
                 if (!hessian.is_none()) h = hessian.cast<torch::Tensor>();
                 a.writeResult(params, filename, is_saved_weight, waves, h);
             },
             pybind11::arg("params"), pybind11::arg("filename"),
             pybind11::arg("is_saved_weight") = 0,
             pybind11::arg("waves") = std::vector<int>(),
             pybind11::arg("hessian") = pybind11::none(),
             "Save weights/histograms. waves: 可选分波下标子集, 只画 |Σ_{i∈S}A_i·v_i|² 的分布"
             "（空=全部）; is_saved_weight=1 时额外导出逐事件 TTree. "
             "hessian: 可选统一 Hessian, 给了就顺便写 interference_rec_err/"
             "interference_gen_err（参数误差, 省一次 Hessian 计算）")
        .def("getHessian", &analysis::getHessian, pybind11::arg("params"),
             "Full Hessian (2n+P)×(2n+P). params: [real(v), imag(v), theta] float64")
        // [可删·v0.3.22 标记] 逐事例干涉树；与 writeResult 的 saved_weight 撞名，
        // 建议合并进 writeResult(..., pairs=...)。目前 example/tests 无调用方。
        .def("writeInterfResult", &analysis::writeInterfResult,
             pybind11::arg("params"), pybind11::arg("filename"),
             pybind11::arg("pairs"),
             "保存指定波对的逐事件干涉形状到 TTree saved_weight "
             "(totalweight/weight_<i>/interf_<i>_<j>/末态四动量); pairs=[[i,j],...]")
        .def("getNDevices", &analysis::getNDevices)
        .def("getDataTensor", &analysis::getDataTensor)
        .def("getPhspTensor", pybind11::overload_cast<>(&analysis::getPhspTensor,
             pybind11::const_))
        .def("getPhspTensor", pybind11::overload_cast<int>(&analysis::getPhspTensor,
             pybind11::const_), pybind11::arg("gpu"),
             "读取指定 GPU 的 phsp 振幅；多卡独立重建需遍历 getNDevices() 并拼接。")
        .def("getExtendedVector", [](analysis& a, const torch::Tensor& v) {
                 // 折叠表在构造期上传到主 GPU，coupling kernel 固定在其上 →
                 // 传别的卡的向量会跨卡读表（多卡下非法访问）。这里显式挡住。
#if defined(CTPWA_BACKEND_CPU)
                 TORCH_CHECK(v.device().is_cpu(),
                             "coupling_vector must be a CPU tensor (CPU backend)");
#else
                 TORCH_CHECK(v.is_cuda(), "coupling_vector must be on CUDA");
                 TORCH_CHECK(v.device().index() == a.getParams().primaryDevice(),
                     "coupling_vector 必须位于主 GPU (cuda:",
                     a.getParams().primaryDevice(), ")，当前在 cuda:",
                     v.device().index(), "（折叠表固定在主卡）");
#endif
                 return a.freeParamsToAmplitudes(v);
             },
             pybind11::arg("coupling_vector"),
             "自由耦合向量 (complex [n_free], 主 GPU) → 扩展振幅耦合向量 v_ext "
             "(complex [n_amps])。用于校验 trans/var_equal 折叠与链×步参数化："
             "两模型的 v_ext 相同 ⇔ 拟合结果相同。")
        // .def("getTruthTensor", &analysis::getTruthTensor)
        // [可删·v0.3.22 标记] 中心值 = diag(interference_gen)/Σdiag（ROOT 里就有），
        // 误差是非相干口径、与 ROOT 的 interference_gen_err 不是同一个量。
        // 删除前先把 example/fit.py 的调用改掉。
        .def("getFitFractions", pybind11::overload_cast<torch::Tensor>(
                 &analysis::getFitFractions),
             pybind11::arg("vector"),
             "Fit fractions: FF_i = ∫|A_i|² / Σ_j∫|A_j|² (纯形状份额, 无效率, "
             "只用 phsp_truth → 与效率 MC/归一化无关, 跨实验可比). "
             "配了 phsp_truth_weights 时积分改为 Σw|A_i|² (带模型产生的 truth 样本 "
             "用 w∝1/|M_gen|² 重加权即可当平坦 PHSP MC). "
             "Σ_i FF_i = 1; 绝对分支比 = BF_total × FF_i. "
             "返回 [npartials, 2] = [center, error].")
        .def("getFitFractions", pybind11::overload_cast<torch::Tensor, torch::Tensor>(
                 &analysis::getFitFractions),
             pybind11::arg("vector"), pybind11::arg("hessian"),
             "getFitFractions(vector, hessian): hessian 为可选统一 Hessian "
             "(与拟合 getHessian 同源), 用于误差传播。")
        .def("getEfficiency", pybind11::overload_cast<torch::Tensor>(
                 &analysis::getEfficiency),
             pybind11::arg("vector"),
             "分波效率: ε_i = (Σ_{phsp}w|A_i|²/Σw) / (Σ_{phsp_truth}w|A_i|²/Σw), "
             "phsp(如 cut 后 MC)=带效率样本, phsp_truth=无效率 MC truth, "
             "即分波加权的探测/选择效率 ∫|A_i|²ε(x)dΦ/∫|A_i|²dΦ. "
             "权重来自 phsp_weights / phsp_truth_weights (缺省 w≡1), "
             "统计误差取 S=Σ(w|A_i|²)² (tf-pwa _eval_integral 同款). "
             "依赖拟合结果(振幅形状含共振参数), 用拟合后 vector 计算. "
             "返回 [npartials, 2] = [center, error], "
             "误差 = 参数误差(Jacobian@H⁻¹@Jᵀ) ⊕ MC 统计误差(tf-pwa add_int_error 同款). "
             "缺 phsp 或 phsp_truth 时返回空张量 [0,2].")
        .def("getEfficiency", pybind11::overload_cast<torch::Tensor, torch::Tensor>(
                 &analysis::getEfficiency),
             pybind11::arg("vector"), pybind11::arg("hessian"),
             "getEfficiency(vector, hessian): hessian 为可选统一 Hessian, "
             "用于参数误差传播。")
        .def("getBkgTensor", &analysis::getBkgTensor)
        .def("getBkgWeightsTensor", &analysis::getBkgWeightsTensor)
        .def("saveSLAmps", &analysis::saveSLAmps)
        .def("getSLAmpsTensor", &analysis::getSLAmpsTensor)
        .def("getConstraintsIndex", &analysis::getConstraintsIndex)
        .def("getConstraintsValues", &analysis::getConstraintsValues)
        .def("getAmplitudeNames", &analysis::getAmplitudeNames)
        .def("getNPolarizations", &analysis::getNPolarizations)
        .def("getIntensity", &analysis::getIntensity,
             pybind11::arg("momenta"), pybind11::arg("params"),
             "getIntensity(momenta, params): 任意四动量的逐事件强度 "
             "Σ_pol |Σ_i A_i v_i|²。momenta 为 CPU float64 [N, n_final, 4] "
             "(E,px,py,pz)，末态顺序 = Data.order；返回 CPU float64 [N]。"
             "归一化任意，只保证相对值。")
        .def("generateMC", &analysis::generateMC,
             pybind11::arg("n"), pybind11::arg("params"),
             pybind11::arg("seed") = 0,
             pybind11::arg("mode") = "unweighted",
             pybind11::arg("batch") = 100000,
             pybind11::arg("max_weight") = pybind11::none(),
             pybind11::arg("max_trials") = pybind11::none(),
             "generateMC(n, params, seed=0, mode='unweighted', batch=100000, "
             "max_weight=None, max_trials=None): 用当前模型产生 MC。"
             "unweighted = 平坦相空间 + |M|² 舍选（等权）；"
             "weighted = 平坦事例 + I 权重。返回 dict: "
             "{'events': [N,n_final,4], 'weights': [N] 或空, 'stats': {...}}。")
        .def("writeMC",
             [](analysis& a, torch::Tensor events, const std::string& filename,
                 pybind11::object weights) {
                 if (weights.is_none())
                     a.writeMC(events, filename);
                 else
                     a.writeMC(events, filename, weights);
             },
             pybind11::arg("events"), pybind11::arg("filename"),
             pybind11::arg("weights") = pybind11::none(),
             "writeMC(events, filename, weights=None): .dat (E px py pz) 或 "
             ".root (TTree 'mc'，TLorentzVector 分支)。")
        .def("reCalcAmp", &analysis::reCalcAmp)
        .def("getFreeResParams", &analysis::getFreeResParams)
        .def("isValid", &analysis::isValid);
}

// ---------------------------------------------------------------------------
// 模块入口：同一个源码树编出三个可能的模块名
//   -DCTPWA_MODULE_CUDA → _cuda.so   （GPU 后端）
//   -DCTPWA_MODULE_CPU  → _cpu.so    （CPU 后端，纯 g++ 构建）
//   都不定义            → ctpwa.so   （旧式单模块，兼容本地开发）
// pybind11 的文件名与 PyInit_<name> 必须一致，故这里用字面量而不是宏拼接。
// ---------------------------------------------------------------------------
#if defined(CTPWA_MODULE_CUDA)
PYBIND11_MODULE(_cuda, m) { bind_ctpwa(m); }
#elif defined(CTPWA_MODULE_CPU)
PYBIND11_MODULE(_cpu, m) { bind_ctpwa(m); }
#else
PYBIND11_MODULE(ctpwa, m) { bind_ctpwa(m); }
#endif
