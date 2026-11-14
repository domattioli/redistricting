"""Tests for redistricting_addons.ensemble (ReCom). Run with pytest, or plainly: python tests/test_ensemble.py"""
import itertools
import sys
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from redistricting_addons.ensemble import (  # noqa: E402
    Graph, grid_graph, log_spanning_trees, plan_stats, recom_chain, summarize, tree_seed)


def _contiguous(g: Graph, labels, k) -> bool:
    for d in range(k):
        m = labels == d
        if not m.any():
            return False
        e = g.edges[m[g.edges[:, 0]] & m[g.edges[:, 1]]]
        idx = np.flatnonzero(m)
        loc = np.full(g.n, -1)
        loc[idx] = np.arange(idx.size)
        a = sparse.coo_matrix((np.ones(len(e)), (loc[e[:, 0]], loc[e[:, 1]])), shape=(idx.size, idx.size))
        if connected_components(a, directed=False)[0] != 1:
            return False
    return True


def _canon(labels):
    """Plan identity ignoring district numbering: the side containing node 0."""
    return tuple(np.flatnonzero(labels == labels[0]))


def _all_balanced_bipartitions(g: Graph):
    n = g.n
    out = set()
    for side in itertools.combinations(range(1, n), n // 2 - 1):
        labels = np.ones(n, int)
        labels[[0, *side]] = 0
        if _contiguous(g, labels, 2):
            out.add(_canon(labels))
    return out


def test_4x4_reaches_every_balanced_plan_and_favours_compact():
    g = grid_graph(4, 4)
    every = _all_balanced_bipartitions(g)
    start = np.repeat([0, 1], 8)  # top half / bottom half
    seen = {}
    # the snakiest plans (10 cut edges) come up ~1 in 10,000 moves, so the run must be long enough to reach them
    for labels in recom_chain(g, start, 2, steps=40000, tol=0.0, seed=1):
        assert np.bincount(labels, minlength=2).tolist() == [8, 8]
        assert _contiguous(g, labels, 2)
        c = _canon(labels)
        seen[c] = seen.get(c, 0) + 1
    assert set(seen) == every, f"reached {len(seen)} of {len(every)} plans"
    cut = {c: int(np.sum(np.isin(g.edges[:, 0], c) != np.isin(g.edges[:, 1], c))) for c in every}
    straight = [c for c in every if cut[c] == 4]  # the two straight halves
    snaky = [c for c in every if cut[c] == max(cut.values())]
    assert np.mean([seen[c] for c in straight]) > 3 * np.mean([seen[c] for c in snaky])


def test_chain_keeps_contiguity_and_tolerance():
    rng = np.random.default_rng(3)
    g = grid_graph(12, 10, pop=rng.integers(50, 150, 120))
    k, tol = 4, 0.05
    start = tree_seed(g, k, tol, seed=2)
    ideal = g.pop.sum() / k
    assert _contiguous(g, start, k)
    assert np.all(np.abs(np.bincount(start, weights=g.pop, minlength=k) / ideal - 1) <= tol + 1e-12)
    for labels in recom_chain(g, start, k, steps=300, tol=tol, seed=4):
        assert _contiguous(g, labels, k)
        assert np.all(np.abs(np.bincount(labels, weights=g.pop, minlength=k) / ideal - 1) <= tol + 1e-12)


def test_county_weight_reduces_splits():
    g = grid_graph(12, 12)
    g.county = (np.arange(144) // 12 // 3) * 4 + (np.arange(144) % 12) // 3  # 16 square counties of 3x3
    k = 4
    start = tree_seed(g, k, 0.05, seed=5)
    def mean_splits(w):
        s = [plan_stats(g, lab, k)["county_splits"] for i, lab in enumerate(recom_chain(g, start, k, 2000, 0.05, w, seed=6)) if i % 10 == 0]
        return np.mean(s)
    assert mean_splits(20.0) < mean_splits(1.0)


def test_log_spanning_trees_known_counts():
    # 3x3 grid: 192 spanning trees; 2x2 grid: 4; a path: 1 (Kirchhoff's theorem, known values)
    g = grid_graph(3, 3)
    assert abs(log_spanning_trees(g, np.zeros(9, int), 1) - np.log(192)) < 1e-9
    g = grid_graph(4, 2)  # left 2x2 block = district 0, right 2x2 block = district 1
    labels = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    assert abs(log_spanning_trees(g, labels, 2) - 2 * np.log(4)) < 1e-9
    g = grid_graph(5, 1)
    assert abs(log_spanning_trees(g, np.zeros(5, int), 1)) < 1e-9
    # a compact split has more trees than a snaky one of the same sizes
    g = grid_graph(4, 4)
    halves = np.repeat([0, 1], 8)
    snake = np.array([0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0])  # 8/8: an L round a block, both connected
    assert _contiguous(g, snake, 2)
    assert log_spanning_trees(g, halves, 2) > log_spanning_trees(g, snake, 2)


def test_summary_and_votes():
    g = grid_graph(6, 6)
    g.county = np.arange(36) // 6
    g.elections = {"e": {"label": "test", "d": np.where(np.arange(36) % 6 < 2, 3.0, 1.0), "r": np.full(36, 2.0)}}
    k = 3
    start = np.repeat([0, 1, 2], 12)
    samples = [[plan_stats(g, lab, k) for lab in recom_chain(g, start, k, 200, 0.0, seed=s)] for s in (7, 8)]
    out = summarize(g, samples, k)
    assert out["samples"] == 400
    el = out["elections"]["e"]
    assert sum(el["seats_d_histogram"].values()) == 400
    assert len(el["sorted_d_share_quantiles"]) == k
    assert "chain_agreement" in out


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
