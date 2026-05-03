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
    Amp2BD(std::array<int, 3> jvalues, std::array<int, 3> parities);
    const std::vector<SL>& getSL() const { return spinOrbitCombinations_; }
    const std::array<int, 3>& getJValues() const { return jvalues_; }
    const std::array<int, 3>& getParities() const { return parities_; }

private:
    std::vector<SL> ComSL(const std::array<int, 3>& spins,
        const std::array<int, 3>& parities);
    std::array<int, 3> jvalues_;
    std::array<int, 3> parities_;
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
    void setBatchSizes(const std::vector<int>& batchSizes);

    void computeSLAmps(const std::vector<std::map<std::string, std::vector<LorentzVector>>>& finalMomenta);
    // void getAmps(Resonance &resonance);
    // cuComplex *getAmps(const std::vector<Resonance> &resonances);
    void getAmps(std::vector<cuComplex*>& d_amplitudes,
        const std::vector<Resonance>& resonances,
        const int site,
        const int n_amplitudes,
        const std::vector<std::vector<int>>& event_offsets,
        const std::vector<std::vector<int>>& amp_offsets);

    // Getter函数
    size_t getNSLCombs() const { return nSLCombs_; }
    // size_t getNEvents() const { return nEvents_; }
    // size_t getNPolarizations() const { return nPolarizations_; }
};

// 核函数声明
__global__ void computeSLAmpKernel(
    thrust::complex<double>* d_amp, thrust::complex<double>* d_amp_buffer,
    const DeviceMomenta* d_momenta, const DecayNode* d_decayNodes,
    const int* d_dj, const int* d_dj1, const int* d_dj2,
    const SL* d_slCombination, int num_sl, int num_events, int num_polar,
    // int decayChain_size, int buffer_size_per_event);
    int decayChain_size, int buffer_size_per_event, int num_batchs,
    int start_event);

__global__ void
computeAmpsKernel(cuComplex* amplitudes,                 // 输出振幅
    const DeviceMomenta* d_momenta,        // 所有事件的四动量数据
    const SL* slCombinations,              // SL组合数据
    const thrust::complex<double>* slamps, // SL振幅
    const DeviceResonance* resonances,     // 共振态数组
    int resonance_count,                   // 共振态数量
    const DecayNode* decayChain,           // 衰变链信息
    int decayChain_size, int nEvents, int nSLComb, int nPolar,
    const int* amp_offsets, const int* event_offsets,
    int num_amp_offsets, int n_amplitudes, int site);

#endif // AMPGEN_CUH
