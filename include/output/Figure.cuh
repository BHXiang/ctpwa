#ifndef FIGURE_CUH
#define FIGURE_CUH

#include <AmpGen.cuh>
#include <Amplitude.cuh>

#include <TFile.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TLorentzVector.h>
#include <TROOT.h>
#include <map>
#include <vector>

// thrust
#include <thrust/binary_search.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

// 直方图配置结构体（包含直方图对象）
struct MassHistConfig {
    std::string name;
    std::string title;
    std::vector<std::string> particles;
    int bins;
    std::vector<double> range;
    std::vector<std::string> tex;

    MassHistConfig(const std::string &n, const std::string &t,
                   const std::vector<std::string> &p, int b,
                   const std::vector<double> &r,
                   const std::vector<std::string> &te = {})
        : name(n), title(t), particles(p), bins(b), range(r), tex(te)
    {
    }
};

struct AngleHistConfig {
    std::string name;
    std::string title;
    std::vector<std::vector<std::string>> particles;
    int bins;
    std::vector<double> range;
    std::vector<std::string> tex;

    AngleHistConfig(const std::string &n, const std::string &t,
                    const std::vector<std::vector<std::string>> &p, int b,
                    const std::vector<double> &r,
                    const std::vector<std::string> &te = {})
        : name(n), title(t), particles(p), bins(b), range(r), tex(te)
    {
    }
};

struct DalitzHistConfig {
    std::string name;
    std::string title;
    std::vector<std::vector<std::string>> particles;
    std::vector<int> bins;
    std::vector<std::vector<double>> range;
    std::vector<std::string> tex;

    DalitzHistConfig(const std::string &n, const std::string &t,
                     const std::vector<std::vector<std::string>> &p,
                     const std::vector<int> &b,
                     const std::vector<std::vector<double>> &r,
                     const std::vector<std::string> &te = {})
        : name(n), title(t), particles(p), bins(b), range(r), tex(te)
    {
    }
};

void CalculateMassHist(LorentzVector *device_momenta,
                       const std::map<std::string, int> &particleToIndex,
                       const std::vector<MassHistConfig> &histConfigs,
                       double *weights, std::vector<TH1F *> &outputHistograms,
                       int nEvents, int nParticles);
void CalculateAngleHist(LorentzVector *device_momenta,
                        const std::map<std::string, int> &particleToIndex,
                        const std::vector<AngleHistConfig> &histConfigs,
                        double *weights, std::vector<TH1F *> &outputHistograms,
                        int nEvents, int nParticles);
void CalculateDalitzHist(LorentzVector *device_momenta,
                         const std::map<std::string, int> &particleToIndex,
                         const std::vector<DalitzHistConfig> &histConfigs,
                         double *weights, std::vector<TH2F *> &outputHistograms,
                         int nEvents, int nParticles);
// 统一观测直方图 (type=="obs"): 1d → output1d, 2d → output2d (与 obsConfigs 同序)
// weights 可空(数据=1); motherIdx = 顶层母粒子 index (角度轴, -1 禁用)
void CalculateObsHist(LorentzVector *device_momenta,
                      const std::map<std::string, int> &particleToIndex,
                      const std::vector<PlotConfig> &obsConfigs,
                      double *weights, std::vector<TH1F *> &output1d,
                      std::vector<TH2F *> &output2d,
                      int nEvents, int nParticles, int motherIdx);

#endif // FIGURE_CUH