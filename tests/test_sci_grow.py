"""Tests for redistricting_addons.sci_grow (no corner touching, stronger compactness, states grown from capitals).

In short (for people): on a small grid "country", checks that sliver borders are dropped without cutting the
country apart, that states grown from their capitals reach their own (unequal) populations in one piece each,
and that optimising never makes the score worse.

For AI agents (detailed): reuses the synthetic ``U`` of ``test_sci_states.toy``; border lengths are 1 per grid
edge except a few slivers (0.01). Checks ``strong_edges`` (slivers dropped, graph still connected), ``grow`` +
``balance_targets`` (one capital per state, each within tol of its own target, contiguous on the strong graph)
and ``StrongChain`` bursts (score never decreases, targets and capitals kept).
"""
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from redistricting_addons import sci_grow as G
from redistricting_addons import sci_states as S
from redistricting_addons.ensemble import Graph, contiguous

from test_sci_states import toy


def _km(U, slivers=(3, 40, 77)):
    km = np.ones(len(U["edges"]))
    km[list(slivers)] = 0.01
    return km


def test_strong_edges_drop_slivers_keep_connected():
    U = toy()
    km = _km(U)
    keep = G.strong_edges(U, km, 0.05)
    assert not keep[[3, 40, 77]].any() and keep.sum() == len(keep) - 3
    e = U["edges"][keep]
    n = len(U["pop"])
    assert connected_components(sparse.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)))[0] == 1


def test_grow_reaches_unequal_targets_and_bursts():
    U = toy()
    km = _km(U)
    U2 = G.subgraph(U, G.strong_edges(U, km))
    k, tol = 4, 0.05
    caps = np.array([0, 15, 16 * 15, 16 * 16 - 1])
    targets = np.array([40.0, 60.0, 70.0, 86.0])  # 256 units of population 1
    lab = G.grow(U2, caps, targets, tol)
    ch = G.StrongChain(U2, lab, k, tol, "raw", capitals=caps, targets=targets, full_edges=U["edges"],
                       border_km=km, max_border_km=40.0, floor=0.5, seed=3)
    G.balance_targets(ch, log=lambda *_: None)
    before = ch.score()
    after = S.short_bursts(ch, 20, 5, log=lambda *_: None)
    assert after >= before - 1e-9
    assert ch.labels[caps].tolist() == [0, 1, 2, 3]
    assert np.abs(ch.devs()).max() <= tol + 1e-9
    assert contiguous(Graph(pop=U2["pop"], edges=U2["edges"]), ch.labels, k)
