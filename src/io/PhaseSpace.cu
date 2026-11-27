// PhaseSpace.cu — native n-body phase-space generator (GENBOD).
//
// This file is included by src/main.cu (same inline pattern as Config.cu /
// Analysis.cu) because it needs ConfigParser's definitions, which live in
// Config.cu.  Do not add it to setup.py as a separate translation unit.

#include <PhaseSpace.cuh>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <set>
#include <stdexcept>
#include <string>

#include <TFile.h>
#include <TLorentzVector.h>
#include <TTree.h>

namespace {
constexpr double kPi = 3.14159265358979323846;

std::string lower_ext(const std::string& s)
{
    std::string t = s;
    std::transform(t.begin(), t.end(), t.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return t;
}
} // namespace

// ============================================================
// construction / validation
// ============================================================

PhaseSpaceGenerator::PhaseSpaceGenerator(const std::string& config_file,
    const std::string& mother)
{
    ConfigParser parser(config_file);
    initialize(parser, mother);
}

PhaseSpaceGenerator::PhaseSpaceGenerator(const ConfigParser& parser,
    const std::string& mother)
{
    initialize(parser, mother);
}

void PhaseSpaceGenerator::initialize(const ConfigParser& parser,
    const std::string& mother)
{
    parser_ = parser;

    final_states_ = parser_.getDataOrder();
    if (final_states_.empty()) {
        // Fallback: final states are particles that appear as daughters but
        // never as mothers in the decay chains.
        std::set<std::string> mothers, daughters;
        for (const auto& chain : parser_.getDecayChains()) {
            for (const auto& step : chain.decay_steps) {
                mothers.insert(step.mother);
                for (const auto& d : step.daughters) daughters.insert(d);
            }
        }
        for (const auto& d : daughters)
            if (mothers.find(d) == mothers.end()) final_states_.push_back(d);
        if (final_states_.empty())
            throw std::runtime_error(
                "PhaseSpaceGenerator: Data.order is empty and no final states "
                "could be inferred from DecayChains");
    }

    // Duplicate names in Data.order would make the DAT reader ambiguous.
    {
        std::set<std::string> seen;
        for (const auto& name : final_states_) {
            if (!seen.insert(name).second)
                throw std::runtime_error(
                    "PhaseSpaceGenerator: duplicate final-state name in "
                    "Data.order: " + name);
        }
    }

    mother_ = mother.empty() ? inferMother() : mother;

    // The mother must be a known particle and must not be listed as a final
    // state (the ctpwa reader reconstructs it from the final-state sum).
    mother_mass_ = massOf(mother_);
    if (std::find(final_states_.begin(), final_states_.end(), mother_) !=
        final_states_.end())
        throw std::runtime_error(
            "PhaseSpaceGenerator: mother '" + mother_ +
            "' appears in Data.order; Data.order must list final states only");

    masses_.clear();
    masses_.reserve(final_states_.size());
    for (const auto& name : final_states_) masses_.push_back(massOf(name));

    if (masses_.size() < 2)
        throw std::runtime_error(
            "PhaseSpaceGenerator: at least 2 final-state particles are "
            "required, got " + std::to_string(masses_.size()));

    rebuild();
}

std::string PhaseSpaceGenerator::inferMother() const
{
    std::set<std::string> mothers, daughters;
    for (const auto& chain : parser_.getDecayChains()) {
        for (const auto& step : chain.decay_steps) {
            mothers.insert(step.mother);
            for (const auto& d : step.daughters) daughters.insert(d);
        }
    }
    std::vector<std::string> top;
    for (const auto& m : mothers)
        if (daughters.find(m) == daughters.end()) top.push_back(m);

    if (top.size() == 1) return top[0];
    if (top.size() > 1) {
        std::string names;
        for (const auto& n : top) names += (names.empty() ? "" : ", ") + n;
        throw std::runtime_error(
            "PhaseSpaceGenerator: multiple top-level mothers in DecayChains ("
            + names + "); pass mother=... explicitly");
    }

    // No chains: fall back to the unique particle not in Data.order.
    std::vector<std::string> cand;
    for (const auto& p : parser_.getParticles()) {
        if (std::find(final_states_.begin(), final_states_.end(), p.name) ==
            final_states_.end())
            cand.push_back(p.name);
    }
    if (cand.size() == 1) return cand[0];
    throw std::runtime_error(
        "PhaseSpaceGenerator: cannot infer the mother particle; pass "
        "mother=... explicitly");
}

double PhaseSpaceGenerator::massOf(const std::string& name) const
{
    for (const auto& p : parser_.getParticles())
        if (p.name == name) return p.mass;
    throw std::runtime_error(
        "PhaseSpaceGenerator: particle '" + name + "' is not defined in "
        "Particles");
}

void PhaseSpaceGenerator::rebuild()
{
    double sum = 0.0;
    for (double m : masses_) sum += m;
    tecmtm_ = mother_mass_ - sum;
    if (!(tecmtm_ > 1e-12))
        throw std::runtime_error(
            "PhaseSpaceGenerator: mother mass " +
            std::to_string(mother_mass_) + " GeV is below the final-state "
            "threshold " + std::to_string(sum) + " GeV");

    double emmax = tecmtm_ + masses_[0];
    double emmin = 0.0;
    double prod = 1.0;
    for (size_t i = 1; i < masses_.size(); ++i) {
        emmin += masses_[i - 1];
        emmax += masses_[i];
        prod *= pdk(emmax, emmin, masses_[i]);
    }
    if (!(prod > 0.0) || !std::isfinite(prod))
        throw std::runtime_error(
            "PhaseSpaceGenerator: invalid GENBOD maximum weight (check the "
            "mother/daughter masses)");
    wtmax_ = 1.0 / prod;
    if (!(wtmax_ > 0.0) || !std::isfinite(wtmax_))
        throw std::runtime_error(
            "PhaseSpaceGenerator: non-finite GENBOD maximum weight");
}

void PhaseSpaceGenerator::setMotherMass(double mass)
{
    if (!(mass > 0.0) || !std::isfinite(mass))
        throw std::runtime_error(
            "PhaseSpaceGenerator::setMotherMass: mass must be positive and "
            "finite");
    mother_mass_ = mass;
    rebuild();
}

// ============================================================
// GENBOD
// ============================================================

double PhaseSpaceGenerator::pdk(double a, double b, double c)
{
    double x = (a - b - c) * (a + b + c) * (a - b + c) * (a + b - c);
    if (x <= 0.0) return 0.0;
    return std::sqrt(x) / (2.0 * a);
}

void PhaseSpaceGenerator::generateOne(std::mt19937_64& rng,
    std::vector<LorentzVector>& momenta, double& weight) const
{
    const int n = static_cast<int>(masses_.size());
    std::uniform_real_distribution<double> unif(0.0, 1.0);

    // Ordered random numbers -> nested invariant masses.
    std::vector<double> rno(n, 0.0);
    rno[n - 1] = 1.0;
    for (int i = 1; i < n - 1; ++i) rno[i] = unif(rng);
    std::sort(rno.begin() + 1, rno.begin() + (n - 1));

    std::vector<double> inv(n, 0.0);
    double cum = 0.0;
    for (int i = 0; i < n; ++i) {
        cum += masses_[i];
        inv[i] = rno[i] * tecmtm_ + cum;
    }

    std::vector<double> pd(n - 1, 0.0);
    for (int i = 0; i < n - 1; ++i)
        pd[i] = pdk(inv[i + 1], inv[i], masses_[i + 1]);

    double wt = wtmax_;
    for (double x : pd) wt *= x;

    momenta.assign(n, LorentzVector());
    momenta[0] = LorentzVector(
        std::sqrt(pd[0] * pd[0] + masses_[0] * masses_[0]), 0.0, pd[0], 0.0);

    for (int i = 1; i < n; ++i) {
        momenta[i] = LorentzVector(
            std::sqrt(pd[i - 1] * pd[i - 1] + masses_[i] * masses_[i]),
            0.0, -pd[i - 1], 0.0);

        // Random rotation of all particles built so far: around z, then y
        // (same order as the GENBOD reference).
        const double cz = 2.0 * unif(rng) - 1.0;
        const double sz = std::sqrt(std::max(0.0, 1.0 - cz * cz));
        const double ang = 2.0 * kPi * unif(rng);
        const double cy = std::cos(ang);
        const double sy = std::sin(ang);
        for (int j = 0; j <= i; ++j) {
            double x = momenta[j].Px;
            double y = momenta[j].Py;
            momenta[j].Px = cz * x - sz * y;
            momenta[j].Py = sz * x + cz * y;

            x = momenta[j].Px;
            double z = momenta[j].Pz;
            momenta[j].Px = cy * x - sy * z;
            momenta[j].Pz = sy * x + cy * z;
        }

        if (i == n - 1) break;

        // Boost the subsystem into the rest frame of the next mother.
        const double beta = pd[i] / std::sqrt(pd[i] * pd[i] + inv[i] * inv[i]);
        for (int j = 0; j <= i; ++j) momenta[j].Boost(0.0, beta, 0.0);
    }

    // Numerical guard: the GENBOD weight is normalised to <= 1.
    if (!(wt > 0.0) || !std::isfinite(wt)) wt = 0.0;
    if (wt > 1.0) wt = 1.0;
    weight = wt;
}

torch::Tensor PhaseSpaceGenerator::generate(int64_t n, uint64_t seed)
{
    if (n < 0)
        throw std::runtime_error("PhaseSpaceGenerator::generate: n < 0");

    const int64_t nf = static_cast<int64_t>(final_states_.size());
    auto opts = torch::TensorOptions().dtype(torch::kFloat64);
    if (n == 0)
        return torch::empty({ 0, nf, 4 }, opts);

    std::vector<double> out(static_cast<size_t>(n) * nf * 4, 0.0);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    std::vector<LorentzVector> mom;

    int64_t accepted = 0;
    last_trials_ = 0;
    while (accepted < n) {
        double wt = 0.0;
        generateOne(rng, mom, wt);
        ++last_trials_;
        if (unif(rng) < wt) {
            double* dst = out.data() +
                static_cast<size_t>(accepted) * nf * 4;
            for (int64_t a = 0; a < nf; ++a) {
                dst[4 * a + 0] = mom[a].E;
                dst[4 * a + 1] = mom[a].Px;
                dst[4 * a + 2] = mom[a].Py;
                dst[4 * a + 3] = mom[a].Pz;
            }
            ++accepted;
        }
    }

    return torch::from_blob(out.data(), { n, nf, 4 }, opts).clone();
}

std::pair<torch::Tensor, torch::Tensor>
PhaseSpaceGenerator::generateWeighted(int64_t n, uint64_t seed)
{
    if (n < 0)
        throw std::runtime_error(
            "PhaseSpaceGenerator::generateWeighted: n < 0");

    const int64_t nf = static_cast<int64_t>(final_states_.size());
    auto opts = torch::TensorOptions().dtype(torch::kFloat64);
    if (n == 0)
        return { torch::empty({ 0, nf, 4 }, opts),
            torch::empty({ 0 }, opts) };

    std::vector<double> out(static_cast<size_t>(n) * nf * 4, 0.0);
    std::vector<double> weights(static_cast<size_t>(n), 0.0);
    std::mt19937_64 rng(seed);
    std::vector<LorentzVector> mom;

    for (int64_t i = 0; i < n; ++i) {
        double wt = 0.0;
        generateOne(rng, mom, wt);
        weights[static_cast<size_t>(i)] = wt;
        double* dst = out.data() + static_cast<size_t>(i) * nf * 4;
        for (int64_t a = 0; a < nf; ++a) {
            dst[4 * a + 0] = mom[a].E;
            dst[4 * a + 1] = mom[a].Px;
            dst[4 * a + 2] = mom[a].Py;
            dst[4 * a + 3] = mom[a].Pz;
        }
    }
    last_trials_ = n;

    return { torch::from_blob(out.data(), { n, nf, 4 }, opts).clone(),
        torch::from_blob(weights.data(), { n }, opts).clone() };
}

// ============================================================
// output
// ============================================================

void PhaseSpaceGenerator::write(const torch::Tensor& events,
    const std::string& filename, const torch::Tensor& weights) const
{
    TORCH_CHECK(events.defined(), "PhaseSpaceGenerator::write: events is undefined");
    torch::Tensor ev = events.detach().to(torch::kCPU).contiguous();
    TORCH_CHECK(ev.scalar_type() == torch::kFloat64,
        "PhaseSpaceGenerator::write: events must be float64");
    TORCH_CHECK(ev.dim() == 3 && ev.size(1) == nFinal() && ev.size(2) == 4,
        "PhaseSpaceGenerator::write: events must have shape [n, ",
        nFinal(), ", 4] (E, px, py, pz)");

    torch::Tensor w;
    bool has_w = false;
    if (weights.defined() && weights.numel() > 0) {
        w = weights.detach().to(torch::kCPU).contiguous();
        TORCH_CHECK(w.scalar_type() == torch::kFloat64 && w.dim() == 1 &&
                w.numel() == ev.size(0),
            "PhaseSpaceGenerator::write: weights must be float64 [n]");
        has_w = true;
    }

    auto acc = ev.accessor<double, 3>();
    const int64_t n = ev.size(0);
    const int nf = nFinal();

    const std::string lower = lower_ext(filename);
    const bool is_root =
        lower.size() >= 5 && lower.substr(lower.size() - 5) == ".root";

    if (!is_root) {
        TORCH_CHECK(!has_w,
            "PhaseSpaceGenerator::write: DAT output does not support "
            "per-event weights; use a .root filename or drop weights");
        std::ofstream out(filename);
        if (!out)
            throw std::runtime_error(
                "PhaseSpaceGenerator::write: cannot open " + filename);
        out << std::setprecision(17);
        for (int64_t i = 0; i < n; ++i)
            for (int a = 0; a < nf; ++a)
                out << acc[i][a][0] << ' ' << acc[i][a][1] << ' '
                    << acc[i][a][2] << ' ' << acc[i][a][3] << '\n';
        out.close();
        return;
    }

    TFile file(filename.c_str(), "RECREATE");
    if (file.IsZombie())
        throw std::runtime_error(
            "PhaseSpaceGenerator::write: cannot create ROOT file " + filename);

    TTree tree("mc", "ctpwa DIYMC phase-space sample");
    std::vector<TLorentzVector> lv(nf);
    std::vector<TLorentzVector*> lvp(nf);
    for (int a = 0; a < nf; ++a) {
        lvp[a] = &lv[a];
        tree.Branch(final_states_[a].c_str(), "TLorentzVector", &lvp[a]);
    }

    double weight = 1.0;
    if (has_w) tree.Branch("weight", &weight, "weight/D");

    if (has_w) {
        auto wacc = w.accessor<double, 1>();
        for (int64_t i = 0; i < n; ++i) {
            for (int a = 0; a < nf; ++a)
                lv[a].SetPxPyPzE(acc[i][a][1], acc[i][a][2], acc[i][a][3],
                    acc[i][a][0]);
            weight = wacc[i];
            tree.Fill();
        }
    } else {
        for (int64_t i = 0; i < n; ++i) {
            for (int a = 0; a < nf; ++a)
                lv[a].SetPxPyPzE(acc[i][a][1], acc[i][a][2], acc[i][a][3],
                    acc[i][a][0]);
            tree.Fill();
        }
    }

    tree.Write();
    file.Close();
}
