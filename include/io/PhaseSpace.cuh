#ifndef PHASESPACE_CUH
#define PHASESPACE_CUH

// DIYMC: n-body phase-space generator (native GENBOD / Raubold-Lynch).
//
// Algorithm follows the published GENBOD recipe (CERNLIB W515; F. James,
// "Monte Carlo Phase Space", CERN 68-15, 1968).  This is a clean-room host
// implementation: it does not copy ROOT's TGenPhaseSpace source.
//
// Generated momenta are in the mother rest frame (unless a mother momentum is
// set in a future version).  Units are GeV, and the output layout is
// [event, final_state, (E, px, py, pz)] with final states in Data.order.

#include <cstdint>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <torch/extension.h>

#include <Amplitude.cuh>
#include <Config.cuh>

class PhaseSpaceGenerator
{
public:
    // config_file: full config.yml.  Only Particles/Data.order/DecayChains are
    // read; data files are NOT opened, so this works without real data.
    explicit PhaseSpaceGenerator(const std::string& config_file,
                                 const std::string& mother = "");

    // Reuse an already-parsed parser (e.g. from analysis::config_parser_).
    explicit PhaseSpaceGenerator(const ConfigParser& parser,
                                 const std::string& mother = "");

    // ---- configuration view ----
    const std::string& mother() const { return mother_; }
    double motherMass() const { return mother_mass_; }
    // Override the mother mass (e.g. for an off-shell / line-shape study).
    void setMotherMass(double mass);
    const std::vector<std::string>& finalStates() const { return final_states_; }
    int nFinal() const { return static_cast<int>(final_states_.size()); }
    // Maximum GENBOD weight after normalisation (<= 1).
    double weightMax() const { return wtmax_; }
    // Number of GENBOD trials used by the last generate() call.
    int64_t lastTrials() const { return last_trials_; }

    // ---- generation ----
    // Flat phase space: GENBOD events are accepted with probability wt, so the
    // result is unweighted dPhi.  Returns float64 CPU tensor [n, n_final, 4].
    torch::Tensor generate(int64_t n, uint64_t seed = 0);

    // Weighted GENBOD sample: returns (events, weights), both length n.
    // Events follow the GENBOD proposal density; use weights for dPhi.
    std::pair<torch::Tensor, torch::Tensor> generateWeighted(int64_t n,
                                                             uint64_t seed = 0);

    // Write events to .dat ("E px py pz" per line, Data.order) or .root
    // (TTree "mc" with one TLorentzVector branch per final state, optional
    // "weight" branch).  weights may be an empty tensor for unweighted output.
    void write(const torch::Tensor& events, const std::string& filename,
               const torch::Tensor& weights = torch::Tensor()) const;

private:
    void initialize(const ConfigParser& parser, const std::string& mother);
    void rebuild();
    std::string inferMother() const;
    double massOf(const std::string& name) const;
    static double pdk(double a, double b, double c);
    void generateOne(std::mt19937_64& rng,
                     std::vector<LorentzVector>& momenta,
                     double& weight) const;

    ConfigParser parser_;
    std::string mother_;
    double mother_mass_ = 0.0;
    std::vector<std::string> final_states_;
    std::vector<double> masses_;
    double tecmtm_ = 0.0;   // mother mass minus sum of daughter masses
    double wtmax_ = 1.0;    // 1 / max(prod PDK)
    int64_t last_trials_ = 0;
};

#endif // PHASESPACE_CUH
