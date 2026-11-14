"""Tests for redistricting_addons.sci (social ties kept together, Social Connectedness Index).

In short (for people): checks that the "ties kept" score is 100% when everyone is in one district, 0% when every
county is its own district, counts split counties fractionally, and that the ensemble records it.

For AI agents (detailed): toy three-county state with known SCI; ``matrices`` must give zero-diagonal weights
summing to 1 with w_ij proportional to SCI_ij * P_i * P_j; ``ties_kept`` must equal the hand-computed value for
a split county. The last test needs ``site/data/sci/PA.json`` (committed) and skips without it.
"""
import numpy as np
import pytest

from redistricting_addons import sci
from redistricting_addons.ensemble import ROOT, load_state, plan_stats

TOY = {"counties": ["1", "2", "3"], "raw": [10, 0, 5], "adjusted": [1.0, 0.0, 2.0]}  # pairs (0,1), (0,2), (1,2)
POP_BY_COUNTY = [100, 100, 200]


def test_matrices_normalised():
    w = sci.matrices(TOY, POP_BY_COUNTY)
    for m in w.values():
        assert np.allclose(np.diag(m), 0) and np.isclose(m.sum(), 1) and np.allclose(m, m.T)
    # raw: (0,1) = 10*100*100 = 1e5, (1,2) = 5*100*200 = 1e5 -> equal halves
    assert np.isclose(w["raw"][0, 1], w["raw"][1, 2])


def test_ties_kept_extremes_and_split():
    w = sci.matrices(TOY, POP_BY_COUNTY)
    county = np.array([0, 0, 1, 1, 2, 2])
    pop = np.array([50, 50, 50, 50, 100, 100], float)
    assert np.isclose(sci.ties_kept(w, county, pop, np.zeros(6, int), 2)["raw"], 100)
    assert np.isclose(sci.ties_kept(w, county, pop, np.array([0, 0, 1, 1, 2, 2]), 3)["raw"], 0)
    # county 1 split in half between district 0 (with county 0) and district 1 (with county 2):
    # pair (0,1) kept 0.5, pair (1,2) kept 0.5 -> raw 50%
    labels = np.array([0, 0, 0, 1, 1, 1])
    assert np.isclose(sci.ties_kept(w, county, pop, labels, 2)["raw"], 50)


@pytest.mark.skipif(not (ROOT / "site" / "data" / "sci" / "PA.json").exists(), reason="SCI file not built")
def test_ensemble_records_ties_kept():
    g = load_state(ROOT / "site" / "data" / "PA.json")
    s = plan_stats(g, g.plans["solver"], g.k)
    assert set(s["sci_ties_kept"]) == {"raw", "adjusted"}
    assert 0 < s["sci_ties_kept"]["adjusted"] < 100
    g2 = load_state(ROOT / "site" / "data" / "PA.json", sci=False)
    assert "sci_ties_kept" not in plan_stats(g2, g2.plans["solver"], g2.k)
