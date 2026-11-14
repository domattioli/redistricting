"""Tests for the phase 3 add-ons (graphs, blocks, packaging fallback). Run with pytest or: python tests/test_phase3.py"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from redistricting_addons import blocks, cenpop, generate, graphs  # noqa: E402
from redistricting_addons.ensemble import contiguous, grid_graph, load_state  # noqa: E402


def _squares(nx, ny):
    """Unit squares as closed rings, keyed by y * nx + x."""
    return {y * nx + x: [np.array([[x, y], [x + 1, y], [x + 1, y + 1], [x, y + 1], [x, y]], float) * 1e-3]
            for y in range(ny) for x in range(nx)}


def test_shared_borders_rook_only():
    pairs, km = graphs.shared_borders(_squares(2, 2))
    assert sorted(map(tuple, pairs.tolist())) == [(0, 1), (0, 2), (1, 3), (2, 3)]  # no diagonal (corner) pairs
    assert np.all(km > 0)


def test_block_adjacency_matches_exact_matching():
    sq = _squares(4, 3)
    exact, _ = graphs.shared_borders(sq)
    hashed = blocks.block_adjacency(sq, 12)
    assert sorted(map(tuple, exact.tolist())) == sorted(map(tuple, hashed.tolist()))


def test_water_links_join_islands():
    xy = np.array([[0, 0], [1, 0], [10, 0], [11, 0]], float)
    links, km, _ = graphs.water_links(4, np.array([[0, 1], [2, 3]]), xy)
    assert links.tolist() == [[1, 2]] and abs(km[0] - 9) < 1e-9


def test_repair_and_balance_to_one_person():
    g = grid_graph(6, 6)
    rng = np.random.default_rng(0)
    pop = rng.integers(1, 9, 36)
    labels = np.repeat([0, 1], 18)
    labels[0] = 1  # a detached corner piece of district 1
    p = blocks.Plan(labels, pop, g.edges, 2)
    assert p.pieces()[0] == 3
    p.repair()
    assert p.pieces()[0] == 2
    p.balance()
    assert p.pops.max() - p.pops.min() <= 1
    assert p.pieces()[0] == 2


def test_contiguous():
    g = grid_graph(4, 1)
    assert contiguous(g, np.array([0, 0, 1, 1]), 2)
    assert not contiguous(g, np.array([0, 1, 0, 1]), 2)


def test_census_graph_loads_connected():
    g = load_state(ROOT / "site" / "data" / "RI.json", graph="census")
    assert g.graph == "census" and g.is_connected()
    island = load_state(ROOT / "site" / "data" / "RI.json", graph="census", water_links=False)
    assert not island.is_connected()  # Rhode Island's islands need water links
    site = load_state(ROOT / "site" / "data" / "RI.json")
    assert site.graph == "site" and site.is_connected()


def test_generate_adds_cenpop_only_when_needed():
    real_copy, real_path = cenpop.REPO_COPY, cenpop.path
    try:
        cenpop.path = lambda: Path("official.txt")
        cenpop.REPO_COPY = Path("does-not-exist.txt")
        assert generate.with_fallback(["--state", "PA"]) == ["--state", "PA", "--cenpop", "official.txt"]
        assert generate.with_fallback(["--state", "PA", "--cenpop", "x"]) == ["--state", "PA", "--cenpop", "x"]
        assert generate.with_fallback(["--state", "PA", "--pop-year", "2010"]) == ["--state", "PA", "--pop-year", "2010"]
        cenpop.REPO_COPY = Path(__file__)  # exists
        assert generate.with_fallback(["--state", "PA"]) == ["--state", "PA"]
    finally:
        cenpop.REPO_COPY, cenpop.path = real_copy, real_path


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
