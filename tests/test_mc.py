"""L0/L1 tests for the DIYMC phase-space generator (native GENBOD).

These tests do not need CUDA: PhaseSpaceGenerator only parses config.yml
(Particles/Data.order/DecayChains) and generates four-momenta on the CPU.
"""

import math

import numpy as np
import pytest
import torch

import ctpwa
from conftest import CONFIGS_DIR

M_JPSI = 3.0969
M_K = 0.4937
M_ETA = 0.5478


def _gen(config="simple.yml", mother=""):
    return ctpwa.PhaseSpaceGenerator(str(CONFIGS_DIR / config), mother)


def test_metadata():
    gen = _gen()
    assert gen.mother() == "Jpsi"
    assert list(gen.finalStates()) == ["Kp", "Km", "eta"]
    assert gen.nFinal() == 3
    assert abs(gen.motherMass() - M_JPSI) < 1e-12
    # J/psi -> K K eta, 3-body GENBOD fWtMax (matches ROOT's GetWtMax)
    assert abs(gen.weightMax() - 0.6393028545698538) < 1e-12


def test_conservation_and_mass_shell():
    gen = _gen()
    ev = gen.generate(5000, seed=42)
    assert ev.shape == (5000, 3, 4)
    assert ev.dtype == torch.float64
    assert ev.device.type == "cpu"

    E = ev[..., 0]
    p = ev[..., 1:]
    total_E = E.sum(dim=1)
    total_p = p.sum(dim=1)

    torch.testing.assert_close(
        total_E, torch.full_like(total_E, M_JPSI), atol=1e-9, rtol=0
    )
    assert total_p.abs().max().item() < 1e-9

    masses = torch.tensor([M_K, M_K, M_ETA], dtype=torch.float64)
    m2 = E**2 - (p**2).sum(dim=-1)
    expected = (masses**2).expand_as(m2)
    torch.testing.assert_close(m2, expected, atol=1e-8, rtol=0)


@pytest.mark.parametrize("n_final", [4, 5])
def test_multibody_conservation(tmp_path, n_final):
    """GENBOD must work beyond 3 bodies (4/5-body conservation + mass shell)."""
    mass_mother = 3.686
    masses = [0.1396] * (n_final - 1) + [0.5478]
    names = [f"P{i}" for i in range(n_final)]
    particles = "\n".join(
        f"  {nm}: {{J: 0, P: -1, mass: {m}}}" for nm, m in zip(names, masses)
    )
    cfg = tmp_path / f"n{n_final}.yml"
    cfg.write_text(
        f"""
Particles:
  X: {{J: 1, P: -1, mass: {mass_mother}}}
{particles}
Data:
  order: [{', '.join(names)}]
"""
    )
    gen = ctpwa.PhaseSpaceGenerator(str(cfg), "X")
    ev = gen.generate(2000, seed=13)
    assert ev.shape == (2000, n_final, 4)

    E = ev[..., 0]
    p = ev[..., 1:]
    total_E = E.sum(dim=1)
    torch.testing.assert_close(
        total_E, torch.full_like(total_E, mass_mother), atol=1e-9, rtol=0
    )
    assert p.sum(dim=1).abs().max().item() < 1e-9

    m2 = E**2 - (p**2).sum(dim=-1)
    expected = (torch.tensor(masses, dtype=torch.float64) ** 2).expand_as(m2)
    torch.testing.assert_close(m2, expected, atol=1e-8, rtol=0)


def test_set_mother_mass():
    gen = _gen()
    old_wmax = gen.weightMax()
    gen.setMotherMass(3.2)
    assert gen.motherMass() == 3.2
    assert gen.weightMax() != old_wmax
    ev = gen.generate(200, seed=2)
    total_E = ev[..., 0].sum(dim=1)
    torch.testing.assert_close(
        total_E, torch.full_like(total_E, 3.2), atol=1e-9, rtol=0
    )


def test_seed_reproducibility():
    gen = _gen()
    a = gen.generate(200, seed=7)
    b = gen.generate(200, seed=7)
    c = gen.generate(200, seed=8)
    assert torch.equal(a, b), "same seed must reproduce identical events"
    assert not torch.equal(a, c), "different seed should differ"


def test_two_body_weight_is_one(tmp_path):
    """GENBOD degenerates to the exact two-body decay; wt == 1."""
    cfg = tmp_path / "two_body.yml"
    cfg.write_text(
        """
Particles:
  X: {J: 1, P: -1, mass: 1.5}
  K1: {J: 0, P: -1, mass: 0.4937}
  K2: {J: 0, P: -1, mass: 0.4937}
Data:
  order: [K1, K2]
"""
    )
    gen = ctpwa.PhaseSpaceGenerator(str(cfg), "X")
    ev, w = gen.generateWeighted(100, seed=3)
    assert ev.shape == (100, 2, 4)
    assert torch.allclose(w, torch.ones_like(w), atol=0, rtol=0)

    total_E = ev[..., 0].sum(dim=1)
    total_p = ev[..., 1:].sum(dim=1)
    torch.testing.assert_close(
        total_E, torch.full_like(total_E, 1.5), atol=1e-12, rtol=0
    )
    assert total_p.abs().max().item() < 1e-12

    # back-to-back and equal momentum magnitude
    assert torch.allclose(ev[:, 0, 1:], -ev[:, 1, 1:], atol=1e-12)


def test_flat_dalitz_inner_region():
    """After wt rejection the 3-body Dalitz density must be flat.

    We use a central square that is fully inside the physical region for
    J/psi -> K+ K- eta (checked analytically: all three pair masses stay
    between their thresholds and M - m_other).
    """
    gen = _gen()
    ev = gen.generate(200_000, seed=11)
    E = ev[..., 0]
    p = ev[..., 1:]

    def pair_m2(i, j):
        s = ev[:, i, :] + ev[:, j, :]
        return s[:, 0] ** 2 - s[:, 1] ** 2 - s[:, 2] ** 2 - s[:, 3] ** 2

    m2_12 = pair_m2(0, 1).numpy()  # K+ K-
    m2_23 = pair_m2(1, 2).numpy()  # K- eta

    # Central square in (m^2_12, m^2_23): m_12, m_23 in [1.5, 1.8] GeV.
    # In m^2 coordinates the Dalitz boundary is linear, so checking the four
    # corners (all physical for J/psi -> K K eta) makes the whole square safe.
    lo, hi = 1.5**2, 1.8**2
    sel = (m2_12 > lo) & (m2_12 < hi) & (m2_23 > lo) & (m2_23 < hi)
    n_sel = int(sel.sum())
    assert n_sel > 2000, f"too few events in the test region: {n_sel}"

    counts, _, _ = np.histogram2d(
        m2_12[sel], m2_23[sel], bins=[10, 10], range=[[lo, hi], [lo, hi]]
    )
    mean = counts.mean()
    chi2 = float(((counts - mean) ** 2 / mean).sum())
    ndf = counts.size - 1
    assert chi2 / ndf < 2.0, f"Dalitz chi2/ndf={chi2/ndf:.2f} (n={n_sel})"


def test_dat_roundtrip(tmp_path):
    gen = _gen()
    ev = gen.generate(10, seed=5)
    path = tmp_path / "flat.dat"
    gen.write(ev, str(path))

    raw = np.loadtxt(path)
    assert raw.shape == (10 * 3, 4)
    back = torch.from_numpy(raw.reshape(10, 3, 4))
    torch.testing.assert_close(back, ev, atol=1e-15, rtol=0)


def test_root_write_exists(tmp_path):
    gen = _gen()
    ev = gen.generate(10, seed=6)
    path = tmp_path / "flat.root"
    gen.write(ev, str(path))
    assert path.exists() and path.stat().st_size > 0

    # Read back with uproot if available: branch names and TLorentzVector
    # contents must match the written events (the ctpwa reader uses the same
    # SetBranchAddress(&TLorentzVector*) convention).
    uproot = pytest.importorskip("uproot")
    with uproot.open(str(path)) as f:
        tree = f["mc"]
        assert tree.num_entries == 10
        assert {"Kp", "Km", "eta"} <= set(tree.keys())
        for a, name in enumerate(["Kp", "Km", "eta"]):
            arr = tree[name].array()
            back = np.stack(
                [
                    np.asarray(arr["fE"], dtype=np.float64),
                    np.asarray(arr["fP"]["fX"], dtype=np.float64),
                    np.asarray(arr["fP"]["fY"], dtype=np.float64),
                    np.asarray(arr["fP"]["fZ"], dtype=np.float64),
                ],
                axis=1,
            )
            np.testing.assert_allclose(back, ev[:, a, :].numpy(), atol=1e-12, rtol=0)


def test_dat_rejects_weights(tmp_path):
    gen = _gen()
    ev = gen.generate(5, seed=1)
    w = torch.ones(5, dtype=torch.float64)
    with pytest.raises(RuntimeError, match="DAT output does not support"):
        gen.write(ev, str(tmp_path / "x.dat"), weights=w)


def test_threshold_error(tmp_path):
    cfg = tmp_path / "bad.yml"
    cfg.write_text(
        """
Particles:
  X: {J: 1, P: -1, mass: 0.9}
  A: {J: 0, P: -1, mass: 0.4937}
  B: {J: 0, P: -1, mass: 0.4937}
Data:
  order: [A, B]
"""
    )
    with pytest.raises(RuntimeError, match="below the final-state threshold"):
        ctpwa.PhaseSpaceGenerator(str(cfg), "X")


def test_mother_in_final_states_error(tmp_path):
    cfg = tmp_path / "bad2.yml"
    cfg.write_text(
        """
Particles:
  X: {J: 1, P: -1, mass: 2.0}
  A: {J: 0, P: -1, mass: 0.4}
Data:
  order: [X, A]
"""
    )
    with pytest.raises(RuntimeError, match="appears in Data.order"):
        ctpwa.PhaseSpaceGenerator(str(cfg), "X")
