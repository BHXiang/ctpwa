#ifndef JITCUSTOM_CUH
#define JITCUSTOM_CUH

// ============================================================================
// 自定义节点求值 JIT（NVRTC）：符号微分字节码 → CUDA C → 运行时编译
//
// 背景: 跨模块 device 函数指针不被 CUDA 支持（cuModuleGetFunction 只暴露
// __global__ 内核，__device__ 函数无法从其他编译单元调用），因此采用两段式:
//   pass-1（NVRTC 编译的物化内核，每 block 一个 module）: 在 (σ, 事件, SL)
//   网格上计算每个 JIT 节点在 aux 字节码上的 (F, ∂F/∂θ[, ∂²F/∂θ²])，
//   写入 device buffer；
//   pass-2（computeCustomAmpsKernel / computeCustomHessianKernel）: 改读
//   buffer 替代解释器 evalCustomAll。buffer 为空/slice=-1 → 回退解释器。
//
// 任何失败（栈深 > 32 / 未知 opcode / NVRTC 错误）→ 整块回退解释器
// （与现有行为一致）。
// ============================================================================

#include <cstddef>
#include <string>
#include <vector>

// 每节点 per-launch 描述（device 侧参数；与生成代码里的结构逐字段一致）
struct JitNodeFlat {
    int node_idx;              // 链节点下标（d_slComb 索引）
    int mother_idx, daug1_idx, daug2_idx;
    double mass0, mass1, mass2;     // DecayNode.mass[0..2]（固定质量；≤0=共振态参数）
    double bf_d;                    // DecayNode.bf_d
    int param_offset;               // all_params 中该共振态参数偏移
    int param_count;                // P
    int slice_base;                 // 本节点 slice 在 out 缓冲中的 double 偏移（per gpu）
    int nvals;                      // 本节点 slice 宽度（grad: 2+2P；full: 2+2P+2P²）
};

// pass-1 内核 per-launch 参数
struct JitArgs {
    const void* d_momenta;      // DeviceMomenta*（grad: 块内事件域；full: 全局域）
    const void* d_mom_tab;      // [nSigma] DeviceMomenta 数组（σ 拓扑）或 null
    const double* all_params;   // 块参数缓冲
    const int* res_param_off;   // [256] 粒子 → param_offset（无参数 -1；q0 链子粒子质量用）
    const void* d_slComb;       // SL*（L 按 (节点, SL) 取）
    double* out;                // 输出缓冲（物化结果 [slice][s][evt][sl][nvals]）
    int decayChain_size;
    int nEvents;
    int evt_offset;             // 事件域偏移（grad: 0；full: hessian 的 evt_off）
    int nSL;
    int nSigma;
    int nJitNodes;
    JitNodeFlat nodes[16];
};

// host 侧节点规格（AmpGen 构建，传给 jitCompileBlock）
struct JitNodeSpec {
    int node_idx, mother_idx, daug1_idx, daug2_idx;
    double mass0, mass1, mass2;
    double bf_d;
    int param_offset, param_count;
};

// 单个 block 的 JIT 状态（host；AmpCalc::ResBlock 持有）
struct JitBlockState {
    bool built = false;        // 已尝试构建（懒，一次）
    bool enabled = false;      // 编译成功（模块二进制已缓存；prep 时按 gpu 加载）
    int hessian_target = -1;   // full 变体只物化该 JIT 节点下标（-1=无）
    std::vector<JitNodeSpec> nodes;      // 节点序（链序；slice 序与此一致）
    std::string cache_key;     // 模块缓存键（含 SM arch；prep 时按 gpu 上下文加载）
    // per-gpu 持久状态
    std::vector<bool> fn_loaded;    // [gpu] CUmodule 已加载到该 context
    std::vector<void*> f_grad_g;    // [gpu] CUfunction（物化内核）
    std::vector<void*> f_full_g;    // [gpu] CUfunction（可能 null：无 hessian 目标）
    std::vector<bool> grad_ready;   // [gpu]
    std::vector<double*> grad_buf;  // [gpu]
    std::vector<size_t> grad_buf_sz;  // [gpu] 已分配字节（≤ 时复用）
    std::vector<int*> grad_slice;   // [gpu] [decayChain_size] 节点→slice double 偏移（-1 解释器）
    std::vector<int*> res_off_dev;  // [gpu] res_param_off 设备副本
    std::vector<JitArgs> args_grad; // [gpu]
    std::vector<bool> full_ready;   // [gpu]
    std::vector<double*> full_buf;  // [gpu]（单 slice：目标节点）
    std::vector<size_t> full_buf_sz;  // [gpu] 已分配字节
    std::vector<JitArgs> args_full; // [gpu]
};

// host API（定义在 src/JITCustom.cu）
//
// 构建/编译（懒；同一 aux 内容缓存命中共享模块）。失败 → enabled=false。
// aux_list[i] = nodes[i] 对应共振态的 aux（[P, n_seg, 段...]），P 取 aux[0]。
bool jitCompileBlock(JitBlockState& st,
                     const std::vector<JitNodeSpec>& nodes,
                     const std::vector<std::vector<double>>& aux_list,
                     int hessian_target);

// 每 gpu 一次性准备（grad 变体；amp 路径，evt_offset=0）
bool jitPrepareGrad(JitBlockState& st, int gpu,
                    const void* d_momenta, const void* d_mom_tab,
                    const double* all_params, const int* h_res_param_off,
                    const void* d_slComb, int decayChain_size,
                    int nEvents, int nSL, int nSigma);

// 每 gpu 一次性准备（full 变体；hessian 路径，evt_offset 由调用方给）
bool jitPrepareFull(JitBlockState& st, int gpu,
                    const void* d_momenta, const void* d_mom_tab,
                    const double* all_params, const int* h_res_param_off,
                    const void* d_slComb, int decayChain_size,
                    int nEvents, int evt_offset, int nSL, int nSigma);

// 启动物化内核（默认 stream；与消费者同流 → 顺序保证）
void jitLaunchGrad(const JitBlockState& st, int gpu);
void jitLaunchFull(const JitBlockState& st, int gpu);

// 释放 per-gpu 资源（module 由进程级缓存持有，不释放）
void jitDestroy(JitBlockState& st);

// CTPWA_NO_JIT=1 → 禁用（数值对拍用）
bool jitEnabled();

#endif // JITCUSTOM_CUH
