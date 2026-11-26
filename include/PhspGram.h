// include/PhspGram.h — phsp Gram 矩阵 → 归一化干涉矩阵（显式输入，无类状态依赖）
//
// P4：`analysis::phspMatrixFromGram` 只读 7 个成员（events/phsp_weights/B̄ 指针/
// n_amplitudes/nSLvectors/primary_dev），却因为住在 god class 里而无法单测、也无法
// 被别的模块复用。这里把输入显式化成一个只读结构体，函数本身成为纯计算。
//
// 这是 P4 的复用模板：**先显式化输入，再搬函数**——比直接搬要安全的，因为编译器会
// 逐个字段告诉调用方漏了什么。
#pragma once

#include <torch/extension.h>

#include <vector>

namespace ctpwa {

// 只读输入（全部指向 analysis 持有的状态；不拥有、不修改）
struct PhspGramInputs {
    const std::vector<std::vector<int>>* events = nullptr;      // 每 GPU 事件数
    const std::vector<double*>* phsp_weights = nullptr;         // 每 GPU phsp 权重（可空）
    const std::vector<int>* nSLvectors = nullptr;               // 每分波的振幅数（分块边界）
    const void* d_matrix = nullptr;         // B̄（ctComplex*，float 档）
    const void* d_matrix_double = nullptr;  // B̄（cuDoubleComplex*，double 档；优先生效）
    int n_amplitudes = 0;                   // B̄ 维度
    int primary_dev = 0;                    // B̄ 所在设备
};

// B̄ 收缩成 npartials×npartials 归一化干涉矩阵 M[p][q]。
//   vector  : 扩展后的振幅向量 [n_amplitudes]（与 .so 复数精度一致）
//   W_out   : 输出 Σ_{p,q≥p} M[p][q]（可选）
//   Wtotal  : 输出 phsp 总权重（可选）
torch::Tensor phspMatrixFromGram(const PhspGramInputs& in,
                                 const torch::Tensor& vector,
                                 double* W_out = nullptr,
                                 double* Wtotal_out = nullptr);

}  // namespace ctpwa
