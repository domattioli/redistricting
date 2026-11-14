"""Tests for redistricting_addons.sci_states and sci_districts (social-connectedness states and districts).

In short (for people): on a small grid "country", checks that every state ends up with exactly one capital,
stays in one piece and within the population tolerance, and that optimising for social ties never makes the
score worse.

For AI agents (detailed): builds a synthetic ``U`` dict (grid units, four regions in quadrants, an SCI matrix with
strong ties inside a quadrant) so no downloads are needed. Checks ``fix_capitals`` (one capital per state, every
state within tol and contiguous), ``Chain`` moves with capitals (capitals never merge), ``short_bursts``
(score never decreases), ``Scorer`` (100% when one state holds everything).
"""
import numpy as np

from redistricting_addons import sci_states as S
from redistricting_addons.ensemble import Graph, contiguous, grid_graph, tree_seed


def toy(nx=16, ny=16):
    g = grid_graph(nx, ny)
    xs, ys = np.meshgrid(np.arange(nx), np.arange(ny))
    region = ((xs.ravel() >= nx // 2).astype(int) + 2 * (ys.ravel() >= ny // 2).astype(int))
    M = np.full((4, 4), 1.0) + 9 * np.eye(4)
    return {"pop": g.pop, "edges": g.edges, "region": region, "M_raw": M, "M_adjusted": M / 2,
            "lat": 40 + 0.1 * ys.ravel(), "lon": -90 + 0.1 * xs.ravel()}


def test_scorer_all_in_one():
    U = toy()
    sc = S.Scorer(U["M_raw"], U["region"], U["pop"], 1)
    assert np.isclose(sc.pct(sc.parts(np.zeros(len(U["pop"]), int))), 100)


def test_fix_capitals_and_bursts():
    U, k, tol = toy(), 4, 0.05
    g = Graph(pop=U["pop"], edges=U["edges"])
    start = tree_seed(g, k, tol, seed=3)
    caps = np.array([0, 15, 16 * 15, 16 * 16 - 1])  # the four corners
    ch = S.Chain(U, start, k, tol, "raw", seed=1)
    lab = S.fix_capitals(U, start, caps, k, tol, ch.wmax_all, seed=2, log=lambda *_: None)
    assert sorted(lab[caps].tolist()) == [0, 1, 2, 3]
    ch = S.Chain(U, lab, k, tol, "raw", capitals=caps, seed=4)
    before = ch.score()
    after = S.short_bursts(ch, 30, 5, log=lambda *_: None)
    assert after >= before - 1e-9
    assert sorted(ch.labels[caps].tolist()) == [0, 1, 2, 3]
    assert np.abs(ch.devs()).max() <= tol + 1e-9
    assert contiguous(g, ch.labels, k)


def test_voronoi_split_connected():
    from scipy import sparse
    from scipy.sparse.csgraph import connected_components
    g = grid_graph(20, 20)
    rng = np.random.default_rng(0)
    eu, ev = g.edges[:, 0], g.edges[:, 1]
    for _ in range(20):
        ca, cb = rng.choice(400, 2, replace=False)
        side = S.voronoi_split(400, eu, ev, rng.uniform(0.5, 1.5, len(eu)), g.pop, int(ca), int(cb),
                               0, np.inf, 0, np.inf, rng, target=200)
        assert side is not None and side[ca] and not side[cb]
        for part in (side, ~side):
            m = part[eu] & part[ev]
            a = sparse.coo_matrix((np.ones(m.sum()), (eu[m], ev[m])), shape=(400, 400))
            sub = np.flatnonzero(part)
            assert connected_components(a.tocsr()[sub][:, sub], directed=False)[0] == 1
