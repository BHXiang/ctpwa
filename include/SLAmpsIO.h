// include/SLAmpsIO.h — SL 振幅表的导出（torch 张量 / 裸二进制）
//
// P4：两个函数原本是 `analysis` 的 const 成员，实际只用到两样东西——
// `amp_calc_` 的 cas 列表与 `phsp_freed_` 标志。抽成自由函数后可以直接被
// 调试脚本/测试调用，不必先造一个 analysis。
#pragma once

#include <AmpGen.cuh>   // AmpCalc

#include <torch/extension.h>

#include <string>

namespace ctpwa {

// 全部链组合的 SL 振幅拼成 [totalSL, nEv, nPol] 的 CPU complex128 张量。
// phsp_freed = 流式 phsp 模式（表已被释放）时直接报错。
torch::Tensor slAmpsTensor(const AmpCalc& amp_calc, bool phsp_freed);

// 把第一条链组合的 SL 振幅按裸 complex<double> 顺序写文件。
void saveSLAmpsToFile(const AmpCalc& amp_calc, const std::string& filename, bool phsp_freed);

// 取驻留振幅矩阵的一段 [offset_el, offset_el+nEl) 转成该设备上的 torch 张量
// （float 存储档位会先上转成 double）。float_amps = 驻留 A 是否为 float2。
torch::Tensor ampSegmentToTensor(ctComplex* base, int64_t offset_el, int64_t nEl,
                                 int gpu, bool float_amps);

}  // namespace ctpwa
