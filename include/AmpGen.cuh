#ifndef AMPGEN_CUH
#define AMPGEN_CUH

#include <Amplitude.cuh>
#include <AutoDiff.cuh>
#include <Config.cuh>
#include <ResModel.cuh>
#include <Resonance.cuh>

#include <array>
#include <cuComplex.h>
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
};

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
           int maxL = -1, bool p_break = false, bool is_bf = true);
    const std::vector<SL>& getSL() const { return spinOrbitCombinations_; }
    const std::array<int, 3>& getJValues() const { return jvalues_; }
    const std::array<int, 3>& getParities() const { return parities_; }
    bool getPbreak() const { return p_break_; }
    bool getIsBf() const { return is_bf_; }

private:
    std::vector<SL> ComSL(const std::array<int, 3>& spins,
        const std::array<int, 3>& parities);
    std::array<int, 3> jvalues_;
    std::array<int, 3> parities_;
    bool identical_daughters_;
    bool is_boson_; // true=boson(symmetric, +), false=fermion(antisymmetric, -)
    int maxL_;      // 轨道角动量上限; -1 = 无限制
    bool p_break_;  // 宇称是否破缺（弱衰变）
    bool is_bf_;    // 是否施加势垒因子
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

    // thrust::complex<double> *d_slamps_ = nullptr;
    // DeviceMomenta *d_momenta_ = nullptr;
    // DecayNode *d_decayNodes_ = nullptr;
    // SL *d_slCombination_ = nullptr;
    std::vector<thrust::complex<double>*> d_slamps_;// = nullptr;
    std::vector<DeviceMomenta*> d_momenta_;// = nullptr;
    std::vector<DecayNode*> d_decayNodes_;// = nullptr;
    std::vector<SL*> d_slCombination_;// = nullptr;
    std::vector<int*> d_polarization_map_;  // GPU polarization mask: output_idx -> tensor_idx
    std::vector<int> h_polarization_map_;   // host copy, used to upload to GPU in computeSLAmps
    size_t nPolarizations_total_;           // total tensor polarizations (before masking)
    // 跨链全同粒子交换拓扑
    std::vector<std::map<std::string, int>> permuted_mappings_; // exchanged name→idx maps
    bool identical_boson_ = true;           // Bose(symmetric sum) / Fermi(alternating)
    // // 每批数据大小
    // int* batchSizes_;

    std::map<std::string, int> particleToIndex_;
    std::vector<DecayNodeHost> decayChain_;
    std::map<std::string, ParticleInfo> particleMap_;
    std::vector<std::string> particleNames_;

    std::vector<size_t> nEvents_;
    size_t nSLCombs_;
    size_t nPolarizations_;

    void addParticleIfNotExists(const std::string& name, int spin, int parity, double mass);
    // void computeNPolarizations_(const std::map<std::string,
    // std::vector<LorentzVector>> &finalMomenta); DeviceMomenta
    // *convertToDeviceMomenta(const std::map<std::string,
    // std::vector<LorentzVector>> &finalMomenta, const std::map<std::string,
    // int> &particleToIndex, const std::vector<DecayNodeHost> &decayChain, int
    // start_event, int batch_size);
    std::vector<DeviceMomenta*> convertToDeviceMomenta(
        const std::vector<std::map<std::string, std::vector<LorentzVector>>>& finalMomenta,
        const std::map<std::string, int>& particleToIndex,
        const std::vector<DecayNodeHost>& decayChain);

public:
    AmpCasDecay(const std::vector<Particle>& particles);
    ~AmpCasDecay();

    void addDecay(const Amp2BD& amp, const std::string& mother, const std::string& daug1, const std::string& daug2);
    std::vector<std::vector<SL>> getSLCombinations() const;
    // int computeNPolarizations(const std::map<std::string, std::vector<LorentzVector>>& finalMomenta);
    void setNPolarizations(const int nPolarizations) { nPolarizations_ = nPolarizations; }
    void setNPolarizationsTotal(const int nTotal) { nPolarizations_total_ = nTotal; }
    void setPolarizationMap(const std::vector<int>& map);
    void setPermutedMappings(const std::vector<std::map<std::string, int>>& maps, bool is_boson);
    void setBatchSizes(const std::vector<int>& batchSizes);

    void computeSLAmps(const std::vector<std::map<std::string, std::vector<LorentzVector>>>& finalMomenta);
    // void getAmps(Resonance &resonance);
    // cuComplex *getAmps(const std::vector<Resonance> &resonances);
    void getAmps(std::vector<cuComplex*>& d_amplitudes,
        const std::vector<Resonance>& resonances,
        const int site,
        const int n_amplitudes,
        const std::vector<std::vector<int>>& event_offsets,
        const std::vector<std::vector<int>>& amp_offsets,
        double bf_d = 3.0);

    // Getter函数
    size_t getNSLCombs() const { return nSLCombs_; }
    size_t getNPolarizations() const { return nPolarizations_; }
    size_t getNPolarizationsTotal() const { return nPolarizations_total_; }
    int getDecayChainSize() const { return static_cast<int>(decayChain_.size()); }
    const std::vector<size_t>& getNEventsVec() const { return nEvents_; }
    const std::vector<thrust::complex<double>*>& getSLAmps() const { return d_slamps_; }
    const std::vector<DeviceMomenta*>& getMomenta() const { return d_momenta_; }
    const std::vector<DecayNode*>& getDecayNodes() const { return d_decayNodes_; }
    const std::vector<SL*>& getDeviceSLCombs() const { return d_slCombination_; }
    const std::vector<int*>& getPolarizationMap() const { return d_polarization_map_; }
    int getParticleIndex(const std::string& tag) const {
        auto it = particleToIndex_.find(tag);
        return (it != particleToIndex_.end()) ? it->second : -1;
    }
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
        int resonance_count;
        int site;                                     // gls_index，对应 d_all_amplitudes 的列偏移
    };

    // 参数槽：一个自由参数 → 对应哪个 block 的哪个共振态的哪个 params 下标
    struct ParamSlot {
        int block_idx;   // 哪个 ResBlock
        int res_idx;     // 该 block 的哪个共振态
        int param_idx;   // params[] 下标 (0=mass, 1=width, 2=r/g_pi, 3=g_K)
    };

    AmpCalc() = default;
    ~AmpCalc();

    // 由 calculateAmplitudes 调用：接管 cas 和共振态组合
    // scan_indices: 与 resonances 对应，每个共振态的自由参数下标；空=不扫，{-1}=全扫
    void addBlock(std::shared_ptr<AmpCasDecay> cas,
                  const std::vector<Resonance>& resonances,
                  int site,
                  const std::vector<std::vector<int>>& scan_indices);

    // 用新参数重算所有振幅
    void reComputeAmps(std::vector<cuComplex*>& d_amplitudes,
                       const double* d_params,             // GPU [nFreeResParams]
                       int n_amplitudes,
                       const std::vector<std::vector<int>>& event_offsets,
                       const std::vector<std::vector<int>>& amp_offsets,
                       size_t n_polar,
                       double bf_d = 3.0);

    int nFreeResParams() const { return static_cast<int>(slots_.size()); }
    bool empty() const { return blocks_.empty(); }

private:
    std::vector<std::shared_ptr<AmpCasDecay>> cas_list_;   // 持有所有权，SL 数据不释放
    std::vector<ResBlock> blocks_;
    std::vector<ParamSlot> slots_;
    // 主机端模板（固定参数值用于恢复）
    std::vector<std::vector<DeviceResonance>> h_templates_;
    std::vector<std::vector<double>> h_param_templates_;
    std::vector<std::vector<double>> h_channel_templates_;
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

__global__ void
computeAmpsKernel(cuComplex* amplitudes,                 // 输出振幅
    const DeviceMomenta* d_momenta,        // 所有事件的四动量数据
    const SL* slCombinations,              // SL组合数据
    const thrust::complex<double>* slamps, // SL振幅
    const DeviceResonance* resonances,     // 共振态数组
    int resonance_count,                   // 共振态数量
    const double* d_all_params,            // 所有共振态的自由参数（flat）
    const double* d_all_channels,          // Flatte channel masses（flat）
    const DecayNode* decayChain,           // 衰变链信息
    int decayChain_size, int nEvents, int nSLComb, int nPolar,
    const int* amp_offsets, const int* event_offsets,
    int num_amp_offsets, int n_amplitudes, int site,
    double bf_d);

#endif // AMPGEN_CUH
