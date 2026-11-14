"""ReCom ensembles: a neutral baseline of districting plans for one state.

Recombination, or ReCom (DeFord, Duchin & Solomon 2019, arXiv:1911.05725):
merge two adjacent districts, draw a random spanning tree over their block
groups, and cut one tree edge whose two sides are both within the population
tolerance. Cutting a tree always leaves two connected pieces, and compact
regions have far more spanning trees than stringy ones, so the chain samples
contiguous, balanced, compact plans without any tuning.

Optional county weighting (Clelland, Colgate, DeFord, Malmskog &
Sancier-Barbosa 2021, arXiv:2011.06049): random edge weights inside a county
are drawn from [0, w] instead of [0, 1] and the *maximum* spanning tree is
kept, so county-internal edges tend to stay in the tree and are cut less.
``county_weight=1`` (the default) is plain ReCom.

The chain reads the site's own data (``site/data/<ST>.json``: block groups,
graph edges, population, counties, election votes), so its numbers match the
toys. It writes summaries only, never the sampled plans::

    python -m redistricting_addons.ensemble --state PA

writes ``site/data/ensembles/PA.json``. Nothing in ``redistricting/`` or the
pages is changed.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import breadth_first_order, connected_components, minimum_spanning_tree
from scipy.sparse.linalg import splu

__all__ = ["Graph", "load_state", "grid_graph", "recom_chain", "tree_seed", "plan_stats", "log_spanning_trees",
           "summarize", "main"]

ROOT = Path(__file__).resolve().parents[1]
QUANTILES = (5, 25, 50, 75, 95)


# --------------------------------------------------------------------------- data

@dataclass
class Graph:
    """Units (block groups) as nodes of an undirected graph."""

    pop: np.ndarray  # (n,) float
    edges: np.ndarray  # (m, 2) int64, unique, i < j
    county: np.ndarray | None = None  # (n,) int county index, or None
    elections: dict = field(default_factory=dict)  # name -> {"label", "d", "r"}
    plans: dict = field(default_factory=dict)  # name -> (n,) int labels, e.g. "solver", "official"
    k: int = 0
    abbr: str = ""
    graph: str = "site"  # "site" (the toys' Delaunay graph) or "census" (shared borders, redistricting_addons.graphs)
    water_links: bool = True
    sci: dict | None = None  # "raw"/"adjusted" -> (C, C) friendship weights (redistricting_addons.sci), or None

    @property
    def n(self) -> int:
        return self.pop.size

    def is_connected(self) -> bool:
        a = sparse.coo_matrix((np.ones(len(self.edges)), (self.edges[:, 0], self.edges[:, 1])), shape=(self.n, self.n))
        return connected_components(a, directed=False)[0] == 1


def _unique_edges(e) -> np.ndarray:
    e = np.asarray(e, np.int64).reshape(-1, 2)
    e = np.sort(e, axis=1)
    e = e[e[:, 0] != e[:, 1]]
    return np.unique(e, axis=0)


GRAPHS = {
    "site": "the site's block-group graph (Delaunay triangulation of centre points clipped to the state, "
            "edges and bridges in site/data/<ST>.json)",
    "census": "Census shared-border graph (site/data/graphs/<ST>.json: rook adjacency from the 2020 boundary file, "
              "shoreline-clipped)",
}


def load_state(path: str | Path, graph: str = "site", water_links: bool = True, sci: bool = True) -> Graph:
    """Load ``site/data/<ST>.json`` (the toys' data format) as a Graph.

    ``graph="site"`` (default) uses the page's own edges plus its water ``bridges``; ``graph="census"`` uses
    the shared-border graph built by ``redistricting_addons.graphs`` plus its ``water_links``. With
    ``water_links=False`` the water bridges / links are left out (pieces cut off by water then stay separate,
    and ReCom needs a connected graph, so states with islands cannot be sampled that way).
    With ``sci=True`` (default) and ``site/data/sci/<ST>.json`` present, the Social Connectedness Index weights are
    loaded too and every plan gets ``sci_ties_kept`` (redistricting_addons.sci); it adds statistics only.
    """
    path = Path(path)
    d = json.loads(path.read_text(encoding="utf-8"))
    if graph == "site":
        base = _unique_edges(d["edges"])
        bridges = _unique_edges(d.get("bridges", []))
        if water_links:
            edges = _unique_edges(np.r_[base, bridges])
        else:
            drop = {tuple(b) for b in bridges.tolist()}
            edges = np.array([e for e in base.tolist() if tuple(e) not in drop], np.int64).reshape(-1, 2)
    elif graph == "census":
        g = json.loads((path.parent / "graphs" / f"{d['abbr']}.json").read_text(encoding="utf-8"))
        if g["n"] != d["n"]:
            raise ValueError(f"{d['abbr']}: census graph has {g['n']} units, site data {d['n']}")
        edges = _unique_edges(np.r_[np.asarray(g["edges"], np.int64),
                                    np.asarray(g["water_links"] if water_links else [], np.int64)])
    else:
        raise ValueError(f"unknown graph {graph!r} (use 'site' or 'census')")
    elections = {name: {"label": e.get("label", name), "d": np.asarray(e["d"], float), "r": np.asarray(e["r"], float)}
                 for name, e in d.get("elections", {}).items()}
    plans = {}
    if "plan" in d:
        plans["solver"] = np.asarray(d["plan"], np.int64)
    if "current" in d:
        plans["official"] = np.asarray(d["current"], np.int64)
    county = np.asarray(d["cty"], np.int64)
    sci_w = None
    sci_p = path.parent / "sci" / f"{d.get('abbr', '')}.json"
    if sci and sci_p.exists():
        from .sci import matrices
        sci_w = matrices(json.loads(sci_p.read_text(encoding="utf-8")),
                         np.bincount(county, weights=np.asarray(d["pop"], float), minlength=len(d["counties"])))
    return Graph(pop=np.asarray(d["pop"], float), edges=edges, county=county,
                 elections=elections, plans=plans, k=int(d["seats"]), abbr=d.get("abbr", ""),
                 graph=graph, water_links=water_links, sci=sci_w)


def grid_graph(nx: int, ny: int, pop=None) -> Graph:
    """An nx-by-ny lattice (rook adjacency), for tests and examples. Node id = y * nx + x."""
    ids = np.arange(nx * ny).reshape(ny, nx)
    e = np.r_[np.column_stack([ids[:, :-1].ravel(), ids[:, 1:].ravel()]),
              np.column_stack([ids[:-1, :].ravel(), ids[1:, :].ravel()])]
    p = np.ones(nx * ny) if pop is None else np.asarray(pop, float)
    return Graph(pop=p, edges=_unique_edges(e))


# --------------------------------------------------------------------------- spanning-tree cuts

def _tree(n, eu, ev, wmax, rng):
    """Random maximum spanning tree (random weights in [0, wmax)), or None if the graph is disconnected."""
    w = rng.random(eu.size) * wmax
    cost = (float(wmax.max()) + 1.0) - w  # positive; minimum tree on cost = maximum tree on w
    t = minimum_spanning_tree(sparse.csr_matrix((cost, (eu, ev)), shape=(n, n)))
    return t if t.nnz == n - 1 else None


def _subtree_pops(t, pop, root=0):
    order, pred = breadth_first_order(t, root, directed=False, return_predecessors=True)
    sub = pop.tolist()
    pl = pred.tolist()
    for v in order[:0:-1].tolist():  # children before parents
        sub[pl[v]] += sub[v]
    return order, pred, np.asarray(sub)


def _descendants(order, pred, v, n):
    inside = [False] * n
    inside[v] = True
    pl = pred.tolist()
    for u in order.tolist():
        p = pl[u]
        if p >= 0 and inside[p]:
            inside[u] = True
    return np.asarray(inside)


def _bipartition(n, eu, ev, wmax, pop, ideal, tol, rng, tries=50):
    """Split a connected region into two pieces each within ``tol`` of ``ideal``: a boolean side mask, or None."""
    lo, hi = ideal * (1 - tol), ideal * (1 + tol)
    for _ in range(tries):
        t = _tree(n, eu, ev, wmax, rng)
        if t is None:
            return None
        order, pred, sub = _subtree_pops(t, pop)
        other = sub[order[0]] - sub
        ok = (sub >= lo) & (sub <= hi) & (other >= lo) & (other <= hi)
        ok[order[0]] = False
        cand = np.flatnonzero(ok)
        if cand.size:
            return _descendants(order, pred, int(cand[rng.integers(cand.size)]), n)
    return None


# --------------------------------------------------------------------------- chain

def _region(graph, mask):
    nodes = np.flatnonzero(mask)
    local = np.full(graph.n, -1, np.int64)
    local[nodes] = np.arange(nodes.size)
    e = graph.edges
    em = mask[e[:, 0]] & mask[e[:, 1]]
    return nodes, local[e[em, 0]], local[e[em, 1]], em


def _wmax(graph, em, county_weight):
    if county_weight == 1 or graph.county is None:
        return np.ones(int(em.sum()))
    e = graph.edges[em]
    return np.where(graph.county[e[:, 0]] == graph.county[e[:, 1]], float(county_weight), 1.0)


def recom_chain(graph: Graph, start, k: int, steps: int, tol: float = 0.01, county_weight: float = 1.0,
                seed: int = 0, max_proposals: int | None = None):
    """Yield the plan after each of ``steps`` accepted ReCom moves (the same array, updated in place).

    A proposal picks a uniformly random cut edge, merges the two districts it joins and re-splits them with a
    random spanning tree. A proposal with no balanced cut after a few trees is discarded and a new pair drawn.
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(start, np.int64).copy()
    ideal = graph.pop.sum() / k
    e0, e1 = graph.edges[:, 0], graph.edges[:, 1]
    budget = max_proposals if max_proposals is not None else 100 * steps + 1000
    done = proposals = 0
    while done < steps:
        if proposals >= budget:
            raise RuntimeError(f"ReCom stalled: {done} of {steps} moves after {proposals} proposals")
        proposals += 1
        cut = np.flatnonzero(labels[e0] != labels[e1])
        e = cut[rng.integers(cut.size)]
        a, b = labels[e0[e]], labels[e1[e]]
        mask = (labels == a) | (labels == b)
        nodes, eu, ev, em = _region(graph, mask)
        side = _bipartition(nodes.size, eu, ev, _wmax(graph, em, county_weight), graph.pop[nodes], ideal, tol, rng)
        if side is None:
            continue
        labels[nodes[side]] = a
        labels[nodes[~side]] = b
        done += 1
        yield labels


def tree_seed(graph: Graph, k: int, tol: float = 0.01, seed: int = 0, tries: int = 200) -> np.ndarray:
    """A random starting plan: cut ``k - 1`` balanced pieces off random spanning trees (recursive tree partition).

    Each piece is within ``tol`` of the ideal, and the pieces left keep an average within ``tol / 2`` so the last
    district is in tolerance too. Removing a subtree leaves the rest of the tree connected, so every district is
    contiguous.
    """
    rng = np.random.default_rng(seed)
    ideal = graph.pop.sum() / k
    for _ in range(tries):
        labels = np.full(graph.n, -1, np.int64)
        remaining = np.ones(graph.n, bool)
        ok = True
        for j in range(k - 1):
            nodes, eu, ev, _ = _region(graph, remaining)
            p = graph.pop[nodes]
            left = k - j - 1
            placed = False
            for _ in range(50):
                t = _tree(nodes.size, eu, ev, np.ones(eu.size), rng)
                if t is None:
                    raise ValueError("graph is not connected")
                order, pred, sub = _subtree_pops(t, p, root=int(rng.integers(nodes.size)))
                rest = (p.sum() - sub) / left
                good = (np.abs(sub / ideal - 1) <= tol) & (np.abs(rest / ideal - 1) <= tol / 2)
                good[order[0]] = False
                cand = np.flatnonzero(good)
                if cand.size:
                    inside = _descendants(order, pred, int(cand[rng.integers(cand.size)]), nodes.size)
                    labels[nodes[inside]] = j
                    remaining[nodes[inside]] = False
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if ok:
            labels[remaining] = k - 1
            return labels
    raise RuntimeError("could not build a balanced starting plan")


# --------------------------------------------------------------------------- statistics

def log_spanning_trees(graph: Graph, labels, k: int) -> float:
    """Sum over districts of the natural log of each district's number of spanning trees.

    Kirchhoff's matrix-tree theorem: the count is the determinant of the district's graph Laplacian with one row
    and column removed. Compact districts have far more spanning trees than stringy ones (Procaccia &
    Tucker-Foltz 2021), and ReCom samples plans roughly in proportion to this product, so higher = more compact.
    """
    labels = np.asarray(labels, np.int64)
    e0, e1 = graph.edges[:, 0], graph.edges[:, 1]
    inner = labels[e0] == labels[e1]
    order = np.argsort(labels, kind="stable")
    bounds = np.searchsorted(labels[order], np.arange(k + 1))
    eorder = np.argsort(labels[e0[inner]], kind="stable")
    ie0, ie1 = e0[inner][eorder], e1[inner][eorder]
    ebounds = np.searchsorted(labels[ie0], np.arange(k + 1))
    local = np.empty(graph.n, np.int64)
    total = 0.0
    for d in range(k):
        nodes = order[bounds[d]:bounds[d + 1]]
        n = nodes.size
        if n <= 1:
            continue
        local[nodes] = np.arange(n)
        u, v = local[ie0[ebounds[d]:ebounds[d + 1]]], local[ie1[ebounds[d]:ebounds[d + 1]]]
        deg = np.bincount(u, minlength=n) + np.bincount(v, minlength=n)
        rows = np.r_[u, v, np.arange(n)]
        cols = np.r_[v, u, np.arange(n)]
        vals = np.r_[-np.ones(u.size), -np.ones(v.size), deg.astype(float)]
        lap = sparse.csc_matrix((vals, (rows, cols)), shape=(n, n))
        ncomp, comp = connected_components(lap, directed=False)
        # a district in several pieces has no spanning tree: count trees piece by piece (a spanning forest)
        for c in range(ncomp):
            idx = np.flatnonzero(comp == c)
            if idx.size <= 1:
                continue
            sub = lap[idx][:, idx][1:, 1:]
            lu = splu(sparse.csc_matrix(sub))
            total += float(np.sum(np.log(np.abs(lu.U.diagonal()))))  # splu's L has a unit diagonal
    return total


def plan_stats(graph: Graph, labels, k: int, trees: bool = False) -> dict:
    """Neutral and partisan statistics of one plan (two-party shares; Democratic seats = districts with D > R)."""
    labels = np.asarray(labels, np.int64)
    e0, e1 = graph.edges[:, 0], graph.edges[:, 1]
    pops = np.bincount(labels, weights=graph.pop, minlength=k)
    out = {"cut_edges": int(np.sum(labels[e0] != labels[e1])),
           "max_dev_pct": float(100 * np.max(np.abs(pops / (graph.pop.sum() / k) - 1)))}
    if trees:
        out["log_spanning_trees"] = log_spanning_trees(graph, labels, k)
    if graph.sci is not None and graph.county is not None:
        from .sci import ties_kept
        out["sci_ties_kept"] = ties_kept(graph.sci, graph.county, graph.pop, labels, k)
    if graph.county is not None:
        live = graph.pop > 0
        pairs = np.unique(graph.county[live] * k + labels[live])
        out["county_splits"] = int(np.sum(np.bincount(pairs // k) > 1))
    for name, el in graph.elections.items():
        d = np.bincount(labels, weights=el["d"], minlength=k)
        r = np.bincount(labels, weights=el["r"], minlength=k)
        share = np.divide(d, d + r, out=np.full(k, 0.5), where=(d + r) > 0)
        out[name] = {"seats_d": int(np.sum(d > r)), "sorted_d_share": np.sort(share)}
    return out


def _hist(values):
    v, c = np.unique(np.asarray(values), return_counts=True)
    return {str(int(a)): int(b) for a, b in zip(v, c)}


def _q(values):
    return [round(float(x), 4) for x in np.percentile(np.asarray(values, float), QUANTILES)]


def _pct_at_or_below(values, x):
    return round(100 * float(np.mean(np.asarray(values) <= x)), 1)


def _tv(h1, h2):
    keys = set(h1) | set(h2)
    n1, n2 = sum(h1.values()), sum(h2.values())
    return round(0.5 * sum(abs(h1.get(s, 0) / n1 - h2.get(s, 0) / n2) for s in keys), 4)


def summarize(graph: Graph, samples: list[list[dict]], k: int) -> dict:
    """Summaries over all chains, plus per-chain agreement checks. ``samples[c]`` = plan_stats dicts of chain c."""
    flat = [s for chain in samples for s in chain]
    out = {"samples": len(flat),
           "cut_edges": {"quantiles": _q([s["cut_edges"] for s in flat]),
                         "mean": round(float(np.mean([s["cut_edges"] for s in flat])), 1)},
           "max_dev_pct": {"quantiles": _q([s["max_dev_pct"] for s in flat])}}
    if "log_spanning_trees" in flat[0]:
        v = [s["log_spanning_trees"] for s in flat]
        out["log_spanning_trees"] = {"quantiles": _q(v), "mean": round(float(np.mean(v)), 2)}
    if "sci_ties_kept" in flat[0]:
        out["sci_ties_kept"] = {name: {"quantiles": _q([s["sci_ties_kept"][name] for s in flat]),
                                       "mean": round(float(np.mean([s["sci_ties_kept"][name] for s in flat])), 3)}
                                for name in flat[0]["sci_ties_kept"]}
    if "county_splits" in flat[0]:
        out["county_splits"] = {"histogram": _hist([s["county_splits"] for s in flat]),
                                "quantiles": _q([s["county_splits"] for s in flat])}
    out["elections"] = {}
    for name, el in graph.elections.items():
        shares = np.array([s[name]["sorted_d_share"] for s in flat])
        out["elections"][name] = {
            "label": el["label"],
            "statewide_d_share": round(float(el["d"].sum() / (el["d"].sum() + el["r"].sum())), 4),
            "seats_d_histogram": _hist([s[name]["seats_d"] for s in flat]),
            "sorted_d_share_quantiles": [_q(shares[:, j]) for j in range(k)],
        }
    # agreement between chains started from different plans
    if len(samples) > 1:
        a, b = samples[0], samples[1]
        agree = {"cut_edges_mean": [round(float(np.mean([s["cut_edges"] for s in c])), 1) for c in (a, b)]}
        for name in graph.elections:
            ha = _hist([s[name]["seats_d"] for s in a])
            hb = _hist([s[name]["seats_d"] for s in b])
            qa = np.array([_q(np.array([s[name]["sorted_d_share"][j] for s in a])) for j in range(k)])
            qb = np.array([_q(np.array([s[name]["sorted_d_share"][j] for s in b])) for j in range(k)])
            agree[name] = {"seats_total_variation": _tv(ha, hb),
                           "median_share_max_diff": round(float(np.max(np.abs(qa[:, 2] - qb[:, 2]))), 4)}
        out["chain_agreement"] = agree
    return out


def reference(graph: Graph, labels, k: int, flat: list[dict]) -> dict:
    """Where an existing plan sits in the ensemble: its statistics and the share of sampled plans at or below it."""
    trees = "log_spanning_trees" in flat[0]
    s = plan_stats(graph, labels, k, trees)
    out = {"cut_edges": s["cut_edges"], "cut_edges_pct_at_or_below": _pct_at_or_below([f["cut_edges"] for f in flat], s["cut_edges"]),
           "max_dev_pct": round(s["max_dev_pct"], 3)}
    if trees:
        out["log_spanning_trees"] = round(s["log_spanning_trees"], 2)
        out["log_spanning_trees_pct_at_or_below"] = _pct_at_or_below([f["log_spanning_trees"] for f in flat],
                                                                      s["log_spanning_trees"])
    if "sci_ties_kept" in s and "sci_ties_kept" in flat[0]:
        out["sci_ties_kept"] = {name: {"value": round(v, 3),
                                       "pct_at_or_below": _pct_at_or_below([f["sci_ties_kept"][name] for f in flat], v)}
                                for name, v in s["sci_ties_kept"].items()}
    if "county_splits" in s:
        out["county_splits"] = s["county_splits"]
        out["county_splits_pct_at_or_below"] = _pct_at_or_below([f["county_splits"] for f in flat], s["county_splits"])
    for name in graph.elections:
        seats = s[name]["seats_d"]
        out[name] = {"seats_d": seats,
                     "seats_d_pct_at_or_below": _pct_at_or_below([f[name]["seats_d"] for f in flat], seats),
                     "sorted_d_share": [round(float(x), 4) for x in s[name]["sorted_d_share"]]}
    return out


# --------------------------------------------------------------------------- run

def contiguous(graph: Graph, labels, k: int) -> bool:
    """True when every district of ``labels`` is one connected piece of ``graph``."""
    labels = np.asarray(labels, np.int64)
    e = graph.edges[labels[graph.edges[:, 0]] == labels[graph.edges[:, 1]]]
    a = sparse.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(graph.n, graph.n))
    ncomp, _ = connected_components(a, directed=False)
    return ncomp == len(np.unique(labels))


def single_district(graph: Graph) -> dict:
    """Summary for an at-large state (one district): there is only one possible plan, so nothing is sampled."""
    labels = np.zeros(graph.n, np.int64)
    s = plan_stats(graph, labels, 1)
    return {"v": 1, "abbr": graph.abbr, "districts": 1, "samples": 0,
            "method": "none: one at-large district, so every plan is the same",
            "elections": {name: {"label": el["label"],
                                 "statewide_d_share": round(float(el["d"].sum() / (el["d"].sum() + el["r"].sum())), 4),
                                 "seats_d": s[name]["seats_d"]} for name, el in graph.elections.items()}}


def run(graph: Graph, k: int, steps: int, burn: int, every: int, tol: float, county_weight: float, seed: int,
        chains: int = 2, log=print, trees: bool = True) -> dict:
    if k == 1:
        return single_district(graph)
    starts = []
    default_graph = graph.graph == "site" and graph.water_links
    if "solver" in graph.plans and (default_graph or contiguous(graph, graph.plans["solver"], k)):
        # (on another graph the solver's plan can be in pieces, and ReCom must start from connected districts)
        starts.append(("solver plan", graph.plans["solver"]))
    while len(starts) < chains:
        starts.append((f"random spanning-tree plan (seed {seed + len(starts)})", tree_seed(graph, k, tol, seed + len(starts))))
    t0 = time.perf_counter()
    samples, chain_meta = [], []
    for c, (name, start) in enumerate(starts[:chains]):
        t1 = time.perf_counter()
        kept = []
        for i, labels in enumerate(recom_chain(graph, start, k, burn + steps, tol, county_weight, seed + 1000 * (c + 1)), 1):
            if i > burn and (i - burn) % every == 0:
                kept.append(plan_stats(graph, labels, k, trees))
            if log and i % 5000 == 0:
                log(f"  chain {c + 1}: {i}/{burn + steps} moves, {time.perf_counter() - t1:.0f}s")
        samples.append(kept)
        chain_meta.append({"start": name, "seed": seed + 1000 * (c + 1), "samples": len(kept),
                           "seconds": round(time.perf_counter() - t1, 1)})
    flat = [s for chain in samples for s in chain]
    out = {
        "v": 1,
        "abbr": graph.abbr,
        "districts": k,
        "method": "ReCom: merge two adjacent districts, cut a random maximum spanning tree at a balanced edge "
                  "(DeFord, Duchin & Solomon 2019, arXiv:1911.05725)"
                  + (f"; county-internal edge weights drawn from [0, {county_weight}] "
                     "(Clelland et al. 2021, arXiv:2011.06049)" if county_weight != 1 else ""),
        "graph": ("the site's block-group graph (edges and bridges in site/data/<ST>.json); cut edges are counted on it"
                  if default_graph else GRAPHS[graph.graph] + ("; water links included" if graph.water_links
                                                               else "; water links left out")
                  + "; cut edges are counted on it"),
        "params": {"moves_per_chain": steps, "burn_in": burn, "keep_every": every, "pop_tolerance": tol,
                   "county_weight": county_weight, "chains": chains, "spanning_trees": trees,
                   **({} if default_graph else {"graph": graph.graph, "water_links": graph.water_links})},
        "chains": chain_meta,
        "seconds": round(time.perf_counter() - t0, 1),
        "quantiles": list(QUANTILES),
        **summarize(graph, samples, k),
        "reference": {name: reference(graph, labels, k, flat) for name, labels in graph.plans.items()},
    }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m redistricting_addons.ensemble", description=__doc__.split("\n\n")[0])
    ap.add_argument("--state", required=True, help="two-letter code, e.g. PA")
    ap.add_argument("--data", help="input JSON (default site/data/<ST>.json)")
    ap.add_argument("--out", help="output JSON (default site/data/ensembles/<ST>.json)")
    ap.add_argument("--steps", type=int, default=20000, help="accepted ReCom moves per chain after burn-in")
    ap.add_argument("--burn", type=int, default=1000, help="moves discarded at the start of each chain")
    ap.add_argument("--every", type=int, default=10, help="keep every n-th plan")
    ap.add_argument("--tol", type=float, default=0.01, help="population tolerance, fraction of the ideal")
    ap.add_argument("--county-weight", type=float, default=1.0, help="1 = plain ReCom; e.g. 20 keeps counties whole more")
    ap.add_argument("--chains", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-trees", action="store_true", help="skip the spanning-tree compactness measure (faster)")
    ap.add_argument("--graph", choices=list(GRAPHS), default="site",
                    help="site = the toys' Delaunay graph (default); census = shared borders from the Census "
                         "boundary file (build with python -m redistricting_addons.graphs)")
    ap.add_argument("--no-water-links", action="store_true",
                    help="leave out water bridges (site) or water links (census); states with islands then fail")
    a = ap.parse_args(argv)
    st = a.state.upper()
    data = Path(a.data) if a.data else ROOT / "site" / "data" / f"{st}.json"
    out = Path(a.out) if a.out else ROOT / "site" / "data" / "ensembles" / f"{st}.json"
    g = load_state(data, graph=a.graph, water_links=not a.no_water_links)
    if not g.is_connected():
        raise SystemExit(f"{data}: graph is not connected")
    print(f"{st}: {g.n} units, {len(g.edges)} edges, {g.k} districts")
    res = run(g, g.k, a.steps, a.burn, a.every, a.tol, a.county_weight, a.seed, a.chains, trees=not a.no_trees)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1024:.1f} kB, {res['samples']} plans, {res.get('seconds', 0)}s)")
    for name, agr in res.get("chain_agreement", {}).items():
        if name != "cut_edges_mean":
            print(f"  {name}: chains differ by TV {agr['seats_total_variation']} in seats, "
                  f"{agr['median_share_max_diff']} in median district share")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
