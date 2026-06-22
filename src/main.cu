// main.cu — 编译入口 & pybind11 绑定

#include <pybind11/pybind11.h>

#include "Amplitude.cu"

#include "ComputeHessian.cu"

#include "AmpGen.cu"
// #include "ComputeGrad.cu"
#include "ComputeNLL.cu"
#include "ComputeResults.cu"
#include "ComputeBF.cu"
#include "Config.cu"
#include "Figure.cu"
#include "Analysis.cu"
#include "Info.cu"
#include "Parameters.cu"
#include "ResModel.cu"
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

    pybind11::class_<analysis>(m, "analysis")
        .def(pybind11::init<const std::string&>(), pybind11::arg("config_file") = "config.yml")
        .def("getNLL", &analysis::getNLL, pybind11::arg("params"),
             "Compute NLL. params: [real(v), imag(v), theta] float64")
        .def("getNVector", &analysis::getNVector)
        .def("getNFreeTheta", &analysis::getNFreeTheta)
        .def("getNParams", &analysis::getNParams)
        .def("getParamNames", &analysis::getParamNames)
        .def("getSLVectors", &analysis::getSLVectors)
        .def("writeResult", &analysis::writeResult)
        .def("getHessian", &analysis::getHessian, pybind11::arg("params"),
             "Full Hessian (2n+P)×(2n+P). params: [real(v), imag(v), theta] float64")
        .def("getDataTensor", &analysis::getDataTensor)
        .def("getPhspTensor", &analysis::getPhspTensor)
        // .def("getTruthTensor", &analysis::getTruthTensor)
        .def("testBWRHessian", &analysis::testBWRHessian,
             pybind11::arg("m"), pybind11::arg("m0"), pybind11::arg("g0"),
             pybind11::arg("L"), pybind11::arg("q"), pybind11::arg("q0"), pybind11::arg("d") = 3.0,
             "Test AutoDiff BWR Hessian. Returns [R_re,R_im, dRe_dm0,dRe_dg0,dIm_dm0,dIm_dg0, d2Re_dm02,d2Re_dm0dg0,d2Re_dg02, d2Im_dm02,d2Im_dm0dg0,d2Im_dg02]")
        .def("getBranchFractions", &analysis::getBranchFractions)
        .def("getBkgTensor", &analysis::getBkgTensor)
        .def("getBkgWeightsTensor", &analysis::getBkgWeightsTensor)
        .def("saveSLAmps", &analysis::saveSLAmps)
        .def("getConstraintsIndex", &analysis::getConstraintsIndex)
        .def("getConstraintsValues", &analysis::getConstraintsValues)
        .def("getAmplitudeNames", &analysis::getAmplitudeNames)
        .def("getNPolarizations", &analysis::getNPolarizations)
        .def("reCalcAmp", &analysis::reCalcAmp)
        .def("getFreeResParams", &analysis::getFreeResParams)
        .def("isValid", &analysis::isValid);
}
