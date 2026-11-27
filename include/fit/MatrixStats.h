// include/MatrixStats.h — 矩阵统计（纯张量数学，与 analysis 状态无关）
//
// P4 的一个小切口：`couplingCovariance` 原本是 `analysis` 的 const 成员，却**一个成员
// 变量都不读**——它只依赖入参（联合 Hessian、耦合维数）。这类函数留在 god class 里
// 有两个代价：无法单测，且让"这个类到底管什么"更模糊。
//
// 抽成自由函数后：调用方显式传参，函数体可以脱离 `analysis` 存在，未来矩阵统计
// （误差传播、协方差、矩阵组装）都往这个模块里放。
#pragma once

#include <torch/extension.h>

namespace ctpwa {

// 从统一 Hessian 取"耦合自由方向"的边际协方差。
//   H      : [2n+ntheta]×[2n+ntheta] float64 统一 Hessian（对称化后求逆）
//   n      : 耦合数（前 2n 维是 [Re p, Im p]，其中 v0 的 Re/Im 固定，不参与）
//   nf_out : 输出自由方向数
//   ok     : 输出是否成功（非正定/求逆失败 → false，调用方应跳过误差矩阵）
// 返回 [nf, nf] 协方差；失败或无自由方向时返回空张量。
torch::Tensor couplingCovariance(const torch::Tensor& H, int n,
                                 int& nf_out, bool& ok);

}  // namespace ctpwa
