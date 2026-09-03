#ifndef AMPGEN_CUH
#define AMPGEN_CUH

#include <Amplitude.cuh>
#include <Config.cuh>
#include <ResModel.cuh>
#include <Resonance.cuh>
#include <SymbolicDiff.cuh>  // Q0MassDep（getDaughterMassDep 签名）

#include <array>
#include "ComplexType.h"
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <unordered_map>
#include <vector>
#include <JITCustom.cuh>

// 自旋-轨道组合结构体
struct SL {
    int S; // 2s+1
    int L;
    __host__ __device__ SL(int s = 0, int l = 0) : S(s), L(l) {}
};

// // 粒子信息结构体
// struct Particle
// {
//     std::string name;
//     int spin;
//     int parity;
//     double mass;
//     std::string tex;
// };

// 衰变节点结构体
struct DecayNode {
    int mother_idx; // 母粒子索引
    int daug1_idx;  // 子粒子1索引
    int daug2_idx;  // 子粒子2索引
    double mass[3] = { -1, -1, -1 };
    // 势垒因子（三级作用域决议后的最终值）: has_bf 是否施加；bf_d 势垒半径
    bool has_bf = true;
    double bf_d = 3.0;
    // 该节点 SL 列表的最小 L（tf 宽度约定 Lmin: BWR 宽度 Γ 与波无关，恒用最低 L）
    int l_min = 0;
};

// 工具函数声明 (定义在 AmpGen.cu)
__device__ double breakup_momentum(double m, double m1, double m2);

// 设备端四动量结构体
struct DeviceMomenta {
    LorentzVector* momenta;    // 所有粒子的四动量
    int n_events;              // 事件数量
    int n_particles_per_event; // 每个事件的粒子数量

    __host__ __device__ DeviceMomenta()
        : momenta(nullptr), n_events(0), n_particles_per_event(0)
    {
    }

    // 获取指定事件和粒子索引的四动量
    __device__ LorentzVector getMomentum(int event_idx, int particle_idx) const;
};

// 两体衰变振幅类
class Amp2BD {
public:
    Amp2BD(std::array<int, 3> jvalues, std::array<int, 3> parities,
           bool identical_daughters = false, bool is_boson = true,
           int maxL = -1, bool p_break = false, bool has_bf = true,
           double bf_d = 3.0,
           std::set<std::pair<int, int>> sl_filter = {});
    const std::vector<SL>& getSL() const { return spinOrbitCombinations_; }
    const std::array<int, 3>& getJValues() const { return jvalues_; }
    const std::array<int, 3>& getParities() const { return parities_; }
    bool getPbreak() const { return p_break_; }
    bool getHasBf() const { return has_bf_; }
    double getBfD() const { return bf_d_; }

private:
    std::vector<SL> ComSL(const std::array<int, 3>& spins,
        const std::array<int, 3>& parities);
    std::array<int, 3> jvalues_;
    std::array<int, 3> parities_;
    bool identical_daughters_;
    bool is_boson_; // true=boson(symmetric, +), false=fermion(antisymmetric, -)
    int maxL_;      // 轨道角动量上限; -1 = 无限制
    bool p_break_;  // 宇称是否破缺（弱衰变）
    bool has_bf_;   // 是否有势垒因子（三级作用域决议后的最终值）
    double bf_d_;   // 势垒半径 d
    std::set<std::pair<int, int>> sl_filter_; // 允许的 (S, L) 分波白名单; 空 = 全允许
    std::vector<SL> spinOrbitCombinations_;
};

// 级联衰变振幅类
class AmpCasDecay {
private:
    struct DecayNodeHost {
        Amp2BD amp;
        std::string mother;
        std::string daug1;
        std::string daug2;
    };

    struct ParticleInfo {
        int spin;
        int parity;
        double mass;
    };

    // 全同粒子置换拓扑（coset）：
    // d_slamp_tab_[gpu] = [nSigma × nEv×nSL×nPol]，σ=0 行 = 恒等 SL 振幅
    std::vector<thrust::complex<double>*> d_slamp_tab_;  // [gpu]
    std::vector<std::vector<DeviceMomenta*>> d_mom_sigma_; // [σ][gpu]：σ 拓扑的重建四动量（σ=0 复用 d_momenta_）
    std::vector<DeviceMomenta*> d_mom_tab_;  // [gpu]：[nSigma] DeviceMomenta 值数组（kernel 用）
    std::vector<double*> d_sign_tab_;     // [gpu]: [nSigma]（sign[0]=+1）
    std::vector<DeviceMomenta*> d_momenta_;// = nullptr;
    std::vector<DecayNode*> d_decayNodes_;// = nullptr;
    std::vector<SL*> d_slCombination_;// = nullptr;
    std::vector<int*> d_polarization_map_;  // GPU polarization mask: output_idx -> tensor_idx
    std::vector<int> h_polarization_map_;   // host copy, used to upload to GPU in computeSLAmps
    size_t nPolarizations_total_;           // total tensor polarizations (before masking)
    // 跨链全同粒子组：(成员名列表, is_boson)；computeSLAmps 中生成子树感知的置换拓扑
    std::vector<std::pair<std::vector<std::string>, bool>> identical_groups_;
    std::vector<std::map<std::string, int>> h_perm_maps_;  // host: [σ≥1] name→idx（重建动量用）
    std::vector<double> h_signs_;                // host: [nSigma]（σ=0 恒等 +1）
    // // 每批数据大小
    // int* batchSizes_;

    std::map<std::string, int> particleToIndex_;
    std::vector<DecayNodeHost> decayChain_;
    std::string chain_name_;  // 所属衰变链名（config.yml 顶层 key，如 decay1）
    std::map<std::string, ParticleInfo> particleMap_;
    std::vector<std::string> particleNames_;

    std::vector<size_t> nEvents_;
    size_t nSLCombs_;
    size_t nPolarizations_;

    void addParticleIfNotExists(const std::string& name, int spin, int parity, double mass);
    std::vector<DeviceMomenta*> convertToDeviceMomenta(
        const std::vector<std::map<std::string, std::vector<LorentzVector>>>& finalMomenta,
        const std::map<std::string, int>& particleToIndex,
        const std::vector<DecayNodeHost>& decayChain);
    // 生成全同粒子置换拓扑（coset）；computeSLAmps 内调用
    void buildPermTopologies();

public:
    AmpCasDecay(const std::vector<Particle>& particles);
    ~AmpCasDecay();

    void addDecay(const Amp2BD& amp, const std::string& mother, const std::string& daug1, const std::string& daug2);
    std::vector<std::vector<SL>> getSLCombinations() const;
    // int computeNPolarizations(const std::map<std::string, std::vector<LorentzVector>>& finalMomenta);
    void setNPolarizations(const int nPolarizations) { nPolarizations_ = nPolarizations; }
    void setNPolarizationsTotal(const int nTotal) { nPolarizations_total_ = nTotal; }
    void setChainName(const std::string& name) { chain_name_ = name; }
    void setPolarizationMap(const std::vector<int>& map);
    // 全同粒子组：(成员名, is_boson)；computeSLAmps 时生成置换拓扑（coset）
    void setIdenticalGroups(const std::vector<std::pair<std::vector<std::string>, bool>>& groups);
    void setBatchSizes(const std::vector<int>& batchSizes);

    void computeSLAmps(const std::vector<std::map<std::string, std::vector<LorentzVector>>>& finalMomenta);
    // void getAmps(Resonance &resonance);
    // ctComplex *getAmps(const std::vector<Resonance> &resonances);
    void getAmps(std::vector<ctComplex*>& d_amplitudes,
        const std::vector<Resonance>& resonances,
        const int site,
        const int n_amplitudes,
        const std::vector<std::vector<int>>& event_offsets,
        const std::vector<std::vector<int>>& amp_offsets,
        bool out_float = false);  // true: 振幅写 float2（混合精度，双倍率内存）

    // 衰变顶点 SL 列表的最小 L（tf 宽度约定 Lmin；未找到返回 0）
    int getNodeLMin(const std::string& mother_name) const;

    // Getter函数
    size_t getNSLCombs() const { return nSLCombs_; }
    size_t getNPolarizations() const { return nPolarizations_; }
    size_t getNPolarizationsTotal() const { return nPolarizations_total_; }
    int getDecayChainSize() const { return static_cast<int>(decayChain_.size()); }
    const std::vector<size_t>& getNEventsVec() const { return nEvents_; }
    // σ=0 行（恒等 SL 振幅）——tab 的起始指针
    const std::vector<thrust::complex<double>*>& getSLAmps() const { return d_slamp_tab_; }
    // 全同粒子置换拓扑（coset）访问
    int getNSigma() const { return static_cast<int>(h_signs_.size()); }
    const std::vector<thrust::complex<double>*>& getSLAmpsTab() const { return d_slamp_tab_; }
    const std::vector<std::vector<DeviceMomenta*>>& getMomentaSigma() const { return d_mom_sigma_; }
    const std::vector<DeviceMomenta*>& getMomentaTab() const { return d_mom_tab_; }
    const std::vector<double*>& getSignsTab() const { return d_sign_tab_; }
    const std::vector<double>& getSignsHost() const { return h_signs_; }
    const std::vector<DeviceMomenta*>& getMomenta() const { return d_momenta_; }
    const std::vector<DecayNode*>& getDecayNodes() const { return d_decayNodes_; }
    const std::vector<SL*>& getDeviceSLCombs() const { return d_slCombination_; }
    const std::vector<int*>& getPolarizationMap() const { return d_polarization_map_; }
    int getParticleIndex(const std::string& tag) const {
        auto it = particleToIndex_.find(tag);
        return (it != particleToIndex_.end()) ? it->second : -1;
    }

    // 查询粒子 mother_idx 的衰变节点子粒子质量依赖（host）。
    // 规则: 固定质量(config) → FixedMass; 子粒子=target(共振态,无固定质量) → M0Param;
    //       否则(事件质量) → EventMass。
    // 返回 false 若该粒子无衰变节点。
    bool getDaughterMassDep(int mother_idx, int target_idx,
        Q0MassDep& md1_dep, double& md1_fixed,
        Q0MassDep& md2_dep, double& md2_fixed) const;

    // host 侧衰变节点（索引/质量/势垒因子已决议）—— JIT 计划构建用
    // （与 computeSLAmps 中上传 device DecayNode 的构造逻辑一致）
    std::vector<DecayNode> getHostDecayNodes() const;
};

// 符号微分统一 kernel 的每 block 描述（设备端；一次启动处理多个 block）
// （结构名 ADBlockDesc 为历史遗留，内容已是 DSL aux 元数据）
struct ADBlockDesc {
    const DeviceMomenta* d_momenta;
    const SL* d_slComb;
    const thrust::complex<double>* d_slamp_tab;  // [nSigma × nEv×nSL×nPol]（σ=0 行=恒等）
    const DeviceResonance* d_res;
    const double* d_all_params;
    const double* d_all_channels;
    const DecayNode* d_decayNodes;
    const int* d_param_map;      // [Nfree]: 自由参数下标
    ctComplex* d_dF_tab;         // 输出 ∂F/∂θ [nSigma × nEvents*nSL*Nfree]（σ=0 行=恒等）
    const double* d_jit_out;     // JIT 物化 buffer（F/dF，[slice][s][evt][sl][2+2P]；null → 解释器）
    const int* d_jit_slice;      // [decayChain_size] 节点 → slice double 偏移（-1 → 解释器）
    // 全同粒子置换拓扑（coset）
    int nSigma;                  // 置换项数（含恒等）
    const DeviceMomenta* d_mom_tab;  // [nSigma] DeviceMomenta 数组（σ 拓扑的重建动量）
    const double* d_sign_tab;    // [nSigma]（sign[0]=+1）
    int resonance_count;
    int nFree;                   // 本 block 自由参数条目数（= 消费端 Nlocal；conjugate 块=owner）
    int res_dF_offset[8];        // per-resonance: base offset in d_dF_tab local free index
    int res_dF_count[8];         // per-resonance: number of free params (0 if fixed)
    int decayChain_size;
    int nEvents;                 // 本 GPU 上该 block 的事件数
    int nSL;                     // 本 block 的 SL 组合数
    int nPolar;
    int sl_start;                // grid.x 累积偏移（组内）
    int site;                    // 振幅列偏移
};

// ============================================================
// AmpCalc: 管理共振态参数扫描
// ============================================================
class AmpCalc {
public:
    // 一个共振态组合块 = 原来一次 getAmps 调用
    struct ResBlock {
        int cas_idx;                                  // 指向 cas_list_[cas_idx]
        std::vector<DeviceResonance*> d_resonances;   // 每个 GPU 一份，持久化，OWNED
        std::vector<double*> d_all_params;            // 每个 GPU：flat 自由参数数组
        std::vector<double*> d_all_channels;          // 每个 GPU：flat channel masses（Flatte）
        std::vector<ctComplex*> d_T;                  // 每个 GPU：有效耦合 T_{e,p}（nEvents×nPolar）
        std::vector<ctComplex*> d_dF;                 // 每个 GPU：∂F/∂θ 复数导数 [nEv×nSL×Nfree]（符号微分 aux 输出）
        int resonance_count;
        std::vector<int> res_dF_offset_;               // per-resonance: d_dF_tab local free index offset
        std::vector<int> res_dF_count_;                // per-resonance: number of free params
        int site;                                     // gls_index，对应 d_all_amplitudes 的列偏移
        int nFree = 0;                                // 该块自由参数数（conjugate 块=owner 的）
        std::vector<int> free_global_idx;             // d_dF[j] → slots_[j] 全局索引（host）
        std::vector<int> free_param_idx;              // d_dF[j] → 参数下标（host，d_param_map）
        // ---- 持久化映射缓冲（懒分配；内容在 addBlock 后固定，避免每次调用 cudaMalloc/cudaFree）----
        std::vector<int*> d_res_idx_;                 // 每 GPU：本块 slot → 共振态索引（updateResonanceParams 用）
        std::vector<int*> d_param_idx_;               // 每 GPU：本块 slot → params[] 下标
        std::vector<int*> d_global_offset_;           // 每 GPU：本块 slot → 全局 slot 下标
        std::vector<int*> d_param_map_;               // 每 GPU：free_param_idx 设备副本（符号微分 kernel 用）
        std::vector<int*> d_global_idx_;              // 每 GPU：free_global_idx 设备副本（梯度累加用）
        // ---- JIT（NVRTC 字节码→原生代码物化；失败回退解释器）----
        JitBlockState jit;
    };

    // 参数槽：一个自由参数 → 对应哪个 block 的哪个共振态的哪个 params 下标
    struct ParamSlot {
        int block_idx;   // 哪个 ResBlock
        int res_idx;     // 该 block 的哪个共振态
        int param_idx;   // params[] 下标 (0=mass, 1=width, 2=r/g_pi, 3=g_K)
        double init_value;  // 初始值
        double lower;       // 下界
        double upper;       // 上界
        std::string name;   // 可读名: "resonance_param"（trans 折叠后唯一）
    };

    AmpCalc() = default;
    ~AmpCalc();

    // 由 calculateAmplitudes 调用：接管 cas 和共振态组合
    // free_indices: 与 resonances 对应，每个共振态的自由参数下标；空=不拟合，{-1}=全部
    // free_ranges:  与 resonances 对应，每个共振态的 [[lower,upper],...]; 空=使用默认
    void addBlock(std::shared_ptr<AmpCasDecay> cas,
                  const std::vector<Resonance>& resonances,
                  int site,
                  const std::vector<std::vector<int>>& free_indices,
                  const std::vector<std::vector<std::vector<double>>>& free_ranges,
                  const std::set<std::string>& skip_slots_for = {},
                  const std::map<std::string, std::string>& conjugate_name_map = {},
                  const std::map<std::string, std::pair<double, double>>& var_range_override = {});

    // 应用 var_equal 约束：每组 [n1, n2, ...] 共享一个参数槽。
    // owner = 组内第一个有槽的名字；其余成员移除自己的槽，
    // 并在 reComputeAmps 中把 owner 的参数值广播到成员的 DeviceResonance。
    // 必须在所有 addBlock 之后、首次使用参数之前调用。
    void applyVarEqual(const std::vector<std::vector<std::string>>& var_equal_groups);

    // 用新参数重算所有振幅
    // params_dev: d_params 所在设备（调用方的主 GPU）。不能依赖运行时当前设备——
    // torch 的 .to(cuda:1) 等操作会把它切走，导致广播从错误设备读取（双卡 NaN 根因）。
    void reComputeAmps(std::vector<ctComplex*>& d_amplitudes,
                       const double* d_params,             // GPU [nFreeResParams]
                       int n_amplitudes,
                       const std::vector<std::vector<int>>& event_offsets,
                       const std::vector<std::vector<int>>& amp_offsets,
                       size_t n_polar,
                       int params_dev);

    // 预计算有效耦合 T_{r,e,p} = Σ_i v_i * sl_i （对每个 block）
    // 必须在 computeResonanceGradient 之前调用，且在耦合向量 v 改变后重新调用
    void computeEffectiveCoupling(const ctComplex* d_v, int n_amplitudes);

    // 计算共振态参数梯度 ∂NLL/∂θ
    // d_w: 来自 computeFactorNLL 的 w = S/I [nEvents × nPolar]（每 GPU 一份）
    // d_grad_res: 输出 [nFreeResParams] double，在 primary GPU 上
    // d_v_per_gpu: 每GPU的耦合向量指针，避免跨设备访问
    void computeResonanceGradient(
        const std::vector<ctComplex*>& d_w,
        const std::vector<int>& n_events,
        double* d_grad_res,
        double sign = 1.0,
        const std::vector<int>& t_offset = {},
        const std::vector<ctComplex*>& d_v_per_gpu = {});

    // 计算共振态参数 Hessian 增量贡献 ∂²L/∂θ∂θ
    // L = Σ_e w_e · log I_e (per-event 加权 log-likelihood)
    // d_hess: 累加到 [P×P]，在 GPU 0，列存储
    // d_event_weights[gpu]: per-event 权重数组（null=使用 default_weight）
    // default_weight: 当 d_event_weights[gpu] 为 null 时使用（如 data=-1, bkg=+0.5）
    // d_v_per_gpu: 每GPU的耦合向量指针（interleaved格式）
    // d_amp_per_gpu: 每GPU的振幅指针
    void computeUnifiedHessian(
        const std::vector<int>& n_events,
        double* d_hess, int hess_ld,
        const std::vector<int>& t_offset,
        double default_weight,
        const std::vector<ctComplex*>& d_v_per_gpu,
        const std::vector<ctComplex*>& d_amp_per_gpu,
        int n_amp_total,
        const std::vector<double*>& d_event_weights = {},
        double* d_phsp_I = nullptr,
        double* d_phsp_grad = nullptr,
        double* d_phsp_hessA = nullptr,
        double* d_mixed_out = nullptr,
        double* d_phsp_mixed_sum = nullptr,
        double* d_phsp_mixed_t3 = nullptr);

    int nFreeResParams() const { return static_cast<int>(slots_.size()); }
    bool empty() const { return blocks_.empty(); }
    const std::vector<ParamSlot>& slots() const { return slots_; }
    const std::vector<std::shared_ptr<AmpCasDecay>>& casList() const { return cas_list_; }
    // 混合精度: 振幅缓冲按 float2 存储（config precision:float）。由 analysis
    // initialize 设置；reComputeAmps 的写出与读取按此选实例。
    void setFloatOutput(bool v) { float_out_ = v; }
    bool floatOutput() const { return float_out_; }

private:
    bool float_out_ = false;
    std::vector<std::shared_ptr<AmpCasDecay>> cas_list_;   // 持有所有权，SL 数据不释放
    std::vector<ResBlock> blocks_;
    std::vector<ParamSlot> slots_;
    // 主机端模板（固定参数值用于恢复）
    std::vector<std::vector<DeviceResonance>> h_templates_;
    std::vector<std::vector<double>> h_param_templates_;
    std::vector<std::vector<double>> h_channel_templates_;
    // conjugate_broadcast_: 跨链同名共振态参数共享
    // {conjugate_block_idx, conjugate_res_idx} → {owner_block_idx, owner_res_idx}
    std::map<std::pair<int,int>, std::pair<int,int>> conjugate_broadcast_;
    // var_equal ghost: 被合并成员的 (block, res, param) → owner 的全局槽下标。
    // ghost 条目挂到成员所在 block 的自由参数列表尾部：updateResonanceParamsKernel
    // 用 owner 槽的值写成员参数（值广播），梯度/Hessian 的导数累加到 owner 槽。
    struct VarEqualGhost {
        int member_block, member_res, member_param;
        int owner_global;
    };
    std::vector<VarEqualGhost> var_equal_ghosts_;
    // Track which (block_idx, res_idx) first registered each resonance name
    std::map<std::string, std::pair<int,int>> resonance_owners_;
    // ---- 持久化跨调用缓冲（懒分配；避免每次调用 cudaMalloc/cudaFree + 隐含同步）----
    std::vector<double*> d_params_per_gpu_;           // 每 GPU：d_params 广播副本（n_free 固定）
    std::vector<double*> d_grad_per_gpu_;             // computeResonanceGradient 的每 GPU 临时梯度
    // amp/event offsets 内容缓存（fit 循环中内容不变，内容变化时自动重传）
    std::vector<std::vector<int>> cached_ev_off_;     // 每 GPU 上次上传的 event_offsets 副本
    std::vector<std::vector<int>> cached_amp_off_;    // 每 GPU 上次上传的 amp_offsets 副本
    std::vector<int*> d_ev_off_cache_;                // 每 GPU 设备端 event_offsets（懒分配）
    std::vector<int*> d_amp_off_cache_;               // 每 GPU 设备端 amp_offsets（懒分配）
    // 合并符号微分 kernel 的 block 描述缓冲（每 GPU 一份，懒分配）
    std::vector<ADBlockDesc*> d_ad_desc_;             // 每 GPU：desc 数组
    int d_ad_desc_cap_ = 0;                           // 每 GPU 已分配容量（block 数）
};

// 核函数声明
__global__ void computeSLAmpKernel(
    thrust::complex<double>* d_amp, thrust::complex<double>* d_amp_buffer,
    const DeviceMomenta* d_momenta, const DecayNode* d_decayNodes,
    const int* d_dj, const int* d_dj1, const int* d_dj2,
    const SL* d_slCombination, int num_sl, int num_events, int num_polar,
    int decayChain_size, int buffer_size_per_event, int num_batchs,
    int start_event,
    const int* d_polarization_map, int num_polar_total);

__global__ void addSLAmpsKernel(
    thrust::complex<double>* d_amp,
    const thrust::complex<double>* d_add,
    int total_size, double sign);

// nPolar 分块实验变体（CHUNK=0 全量 / CHUNK>0 分块，CTPWA_POL_MODE 分派）
template<int CHUNK, int MAXPOL, typename Out = ctComplex>
__global__ void
computeAmpsKernelT(Out* amplitudes,                      // 输出振幅(计算恒 double, Out=float2 时混合精度存储)
    const DeviceMomenta* d_momenta,        // 所有事件的四动量数据
    const SL* slCombinations,              // SL组合数据
    const thrust::complex<double>* slamp_tab, // SL振幅 [nSigma × nSL×nPol×nEv]
    int nSigma,                            // 置换项数（含恒等）
    const DeviceMomenta* d_mom_tab,        // [nSigma] 重建动量数组
    const double* d_sign_tab,              // [nSigma]
    const DeviceResonance* resonances,     // 共振态数组
    int resonance_count,                   // 共振态数量
    const double* d_all_params,            // 所有共振态的自由参数（flat）
    const double* d_all_channels,          // Flatte channel masses（flat）
    const DecayNode* decayChain,           // 衰变链信息
    int decayChain_size, int nEvents, int nSLComb, int nPolar,
    const int* amp_offsets, const int* event_offsets,
    int num_amp_offsets, int n_amplitudes, int site, int k_start);

// 共振态参数梯度 kernel：对 block 的 Nfree 个自由参数计算 ∂NLL/∂θ
// （d_dF 由 reComputeAmps 的 computeCustomAmpsKernel 预计算；本 kernel 纯读取）
template <int Nfree>
__global__ void resonanceGradientKernel(
    const ctComplex* d_w,                  // [nEvents × nPolar] w = S/I
    const thrust::complex<double>* d_slamp_tab,  // [nSigma × nSL×nPol×nEv_total]
    const ctComplex* d_v,                  // 耦合向量（site 偏移）
    const ctComplex* d_dF_tab,             // [nSigma × nEv_total×nSL×Nfree]
    const int* d_global_idx,               // [Nfree]：全局 slots_ 下标
    double* d_grad,                        // 输出（累加）
    int nEvents, int nPolar, int nSLComb, double sign,
    int evt_off, int site, int n_events_total,
    int nSigma, const double* d_sign_tab); // 置换拓扑（σ=0 恒等 +1）


// Cross-TU kernel declarations（rdc 多 TU 链接需要）
__global__ void daxpy_kernel(double* y, const double* x, double alpha, int n);

__global__ void multiplicativeCouplingKernel(
    ctComplex* d_v, const double* d_params,
    const int* d_amp_chain,
    const int* d_step_offsets,
    const int* d_step_data,
    const double* d_amp_chain_ratio,
    int n_amps, int n_step_free, int n_free);

__global__ void multiplicativeGradientKernel(
    double* d_grad_p,
    const ctComplex* d_grad_v,
    const ctComplex* d_v,
    const double* d_params,
    const int* d_amp_chain,
    const int* d_step_offsets,
    const int* d_step_data,
    int n_amps, int n_step_free, int n_free);

#endif // AMPGEN_CUH
