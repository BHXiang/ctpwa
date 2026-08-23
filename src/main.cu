// main.cu — 编译入口 & pybind11 绑定

#include <pybind11/pybind11.h>

#include <Amplitude.cuh>

#include <DeviceManager.cuh>
// DeviceManager.cu inline (no device code)
#include "DeviceManager.cu"

#include <ComputeHessian.cuh>

#include <AmpGen.cuh>
// #include "ComputeGrad.cu"
#include <ComputeNLL.cuh>
#include <ComputeResults.cuh>
#include <ComputeBF.cuh>
#include <Config.cuh>
// Config inline (host-only, no device code)
#include "Config.cu"
#include <Figure.cuh>
// Analysis inline (no header for analysis class)
#include "Analysis.cu"
#include <Info.cuh>
// Info inline (host-only)
#include "Info.cu"
#include <Parameters.cuh>
#include <SymbolicDiff.cuh>  // Node/deriv for modelDeriv+buildModelAST
#include <ResModel.cuh>
// ResModel inline (templates in header, only host code left)
#include "ResModel.cu"
#include <CustomExpr.cuh>
#include <Resonance.cuh>
// Resonance inline (host-only, no device code)
#include "Resonance.cu"

PYBIND11_MODULE(ctpwa, m)
{
    m.doc() = "ctpwa";

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
        .def("print", &DecayInfo::print);

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
            return std::string(PRECISION_NAME); });

    pybind11::class_<analysis>(m, "analysis")
        .def(pybind11::init<const std::string&>(), pybind11::arg("config_file") = "config.yml")
        .def("getNLL", &analysis::getNLL, pybind11::arg("params"),
             "Compute NLL. params: [real(v), imag(v), theta] float64")
        .def("setFitMode", &analysis::setFitMode, pybind11::arg("mode"),
             "Set fit mode: 0=FREEPARAMS (chain×step), 1=VSPACE (direct amplitudes, default)")
        .def("getFitMode", &analysis::getFitMode)
        .def("getNVector", &analysis::getNVector)
        .def("getNFreeTheta", &analysis::getNFreeTheta)
        .def("getNParams", &analysis::getNParams)
        .def("getParamNames", &analysis::getParamNames)
        .def("getSLVectors", &analysis::getSLVectors)
        .def("writeResult", &analysis::writeResult,
             pybind11::arg("params"), pybind11::arg("filename"),
             pybind11::arg("is_saved_weight") = 0,
             pybind11::arg("waves") = std::vector<int>(),
             "Save weights/histograms. waves: 可选分波下标子集, 只画 |Σ_{i∈S}A_i·v_i|² 的分布"
             "（空=全部）; is_saved_weight=1 时额外导出逐事件 TTree")
        .def("getHessian", &analysis::getHessian, pybind11::arg("params"),
             "Full Hessian (2n+P)×(2n+P). params: [real(v), imag(v), theta] float64")
        .def("writeInterfResult", &analysis::writeInterfResult,
             pybind11::arg("params"), pybind11::arg("filename"),
             pybind11::arg("pairs"),
             "保存指定波对的逐事件干涉形状到 TTree saved_weight "
             "(totalweight/weight_<i>/interf_<i>_<j>/末态四动量); pairs=[[i,j],...]")
        .def("getDataTensor", &analysis::getDataTensor)
        .def("getPhspTensor", &analysis::getPhspTensor)
        // .def("getTruthTensor", &analysis::getTruthTensor)
        .def("getFitFractions", &analysis::getFitFractions,
             pybind11::arg("vector"), pybind11::arg("hessian") = torch::Tensor(),
             "Fit fractions: FF_i = ∫|A_i|² / Σ_j∫|A_j|² (纯形状份额, 无效率, "
             "只用 phsp_truth → 与效率 MC/归一化无关, 跨实验可比). "
             "Σ_i FF_i = 1; 绝对分支比 = BF_total × FF_i. "
             "返回 [npartials, 2] = [center, error]. "
             "hessian: 可选统一 Hessian (与拟合 getHessian 同源), 用于误差传播。")
        .def("getBkgTensor", &analysis::getBkgTensor)
        .def("getBkgWeightsTensor", &analysis::getBkgWeightsTensor)
        .def("saveSLAmps", &analysis::saveSLAmps)
        .def("getSLAmpsTensor", &analysis::getSLAmpsTensor)
        .def("getConstraintsIndex", &analysis::getConstraintsIndex)
        .def("getConstraintsValues", &analysis::getConstraintsValues)
        .def("getAmplitudeNames", &analysis::getAmplitudeNames)
        .def("getNPolarizations", &analysis::getNPolarizations)
        .def("reCalcAmp", &analysis::reCalcAmp)
        .def("getFreeResParams", &analysis::getFreeResParams)
        .def("isValid", &analysis::isValid);
}
