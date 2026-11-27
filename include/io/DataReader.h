// include/DataReader.h — 输入数据读取（纯 host：文本 .dat 与 ROOT TTree）
//
// modularization-plan P4 的第一刀：这两个函数原本以自由函数形式住在
// `Analysis.cu`（6300 行 god class 所在的那个 TU）里，虽然逻辑上早已与设备无关，
// 但仍然随 `analysis` 一起编译、一起被 12.8k 行的 TU 拖着。
//
// 抽出来之后：
//   * 新增一种输入格式 = 改这个模块，不碰拟合/振幅/设备代码；
//   * 可以脱离 GPU 单独测（返回值是 STL 容器）；
//   * `Analysis.cu` 少 330 行（7624 行）。
//
// 注意：把权重**上传到设备**的 `readWeightsFromFile` 仍留在 Analysis.cu——
// 它返回 `std::vector<double*>`（设备指针），属于 loader/device 层，不是纯 io。
#pragma once

#include <Amplitude.cuh>   // LorentzVector

#include <map>
#include <string>
#include <vector>

// 从 .dat / ROOT 读取四动量。
//   fileinfo    : {类型, 文件名[, TTree 名]}，类型为 "dat" 或 "ROOT"
//   particleNames : 全部粒子名（用于校验）
//   particlelists : 需要读出的粒子名（返回的 map 键）
// 返回 name -> 每事件四动量；出错时返回已读到的部分（并打印原因），与旧行为一致。
std::map<std::string, std::vector<LorentzVector>> readMomentaFromDat(
    const std::vector<std::string>& fileinfo,
    const std::vector<std::string>& particleNames,
    const std::vector<std::string>& particlelists,
    int nEvents = -1,
    int offset = 0);

// 从 .dat / ROOT 读取权重（纯 host）。expected = 期望权重个数（不足补零/超出截断）。
std::vector<double> readWeightValuesFromFile(
    const std::vector<std::string>& fileinfo,
    long expected);
