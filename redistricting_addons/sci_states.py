"""Fifty new states drawn by social connectedness (Meta's Social Connectedness Index), equal population.

In short (for people): this redraws the map of the United States as 50 states of equal population, putting
together the places whose people are most connected on Facebook (the SCI). Two versions: **free** (the states
can form anywhere) and **capitals** (each of today's 50 capitals stays the capital of its own state; only the
borders move). Every state is one connected piece. Each result is scored by the share of friendships between
people in different counties that stay inside one state, and compared with today's states and with neutral
equal-population states drawn without the SCI. ``python -m redistricting_addons.sci_states`` builds all four
maps (free / capitals × raw / distance-adjusted SCI); maps go to ``docs/sci_states/`` and data to
``site/data/sci_states/``. SCI follows migration, party and race, so these are a lens on how the country is
connected, not a proposal that is neutral.

For AI agents (detailed):
- Units: 2020 counties, except counties above ``--split`` people (default 100,000), which are split into their
  census tracts so states can be balanced to within ``--tol`` (default 1%) of 1/50 of the population of the 50
  states plus DC (Puerto Rico left out). Population: 2020 CenPop block-group file (``cenpop``).
- Adjacency: within each state from ``site/data/graphs/<ST>.json`` (block-group rook adjacency plus water
  links); across state lines from the Census block-group boundary file (``graphs.shared_borders`` on block
  groups in the overlap of each pair of state bounding boxes); DC from its own polygons. Block-group edges are
  collapsed to unit edges, and ``graphs.water_links`` joins the remaining pieces to their nearest unit (Alaska
  to Washington); Hawaii is joined from its most populous unit to the nearest lower-48 unit (its nearest unit
  overall is in the Aleutians, which would make Juneau and Honolulu states impossible together).
- SCI region of a unit: its county, except Connecticut, where the SCI uses the 2022 planning regions: each
  block-group centre is placed in a planning region (point in polygon, ``cb_2022_us_county_500k``) and a unit
  takes its population's majority region. M_raw[r, s] = SCI; M_adj = SCI / exp(a + b log km) with the national
  fit of ``sci.distance_fit`` (centres = population-weighted block-group centres of each region).
- Score (percent): sum_s f_s' M0 f_s / P' M0 P, f_s = population of each region in state s, M0 = M with a zero
  diagonal (own-region pairs left out, as in ``sci.ties_kept``).
- Start: free = ``ensemble.tree_seed`` (recursive random spanning-tree split, 50 pieces within tol);
  capitals = ``grow_from_capitals`` (each state grows from its capital's unit; the smallest state takes the
  frontier unit whose SCI region it is most tied to), then ``balance_units`` (single border units move one hop
  at a time from surplus toward the nearest deficit, never disconnecting a state, capital units fixed; the
  idea of ``blocks.Plan.balance`` with a local connectivity test so it is fast at this size). (Kept but not used: ``balance``, ``capital_seed`` and ``fix_capitals``, which re-split pairs of
  states with ReCom cuts; they stall because a cut must fall on the short tree path between two nearby capitals.)
- Optimisation: short bursts (Cannon, Goldbloom-Helzner, Gupta, Matthews & Suwal 2023, "Voting rights,
  Markov chains, and optimization by short bursts", arXiv:2011.02288): bursts of ``--burst`` ReCom moves whose
  spanning trees favour strongly tied edges (random weights up to 1 + 4 × rank of log SCI); the best plan of a
  burst is kept when it scores at least as well. With capitals, a split is accepted only when the two capitals end
  up on different sides. Neutral comparison: plain ReCom from the free start, no SCI, 20 samples.
- Output: ``site/data/sci_states/<free|capitals>.<raw|adjusted>.json`` (states with name, capital, population,
  deviation, score contribution and unit GEOIDs; scores; today's states' score; neutral quantiles; parameters) and
  ``docs/sci_states/<same>.png``.
"""
from __future__ import annotations

import argparse
import json
import time
import zipfile
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import breadth_first_order, connected_components

from . import cenpop, graphs, sci
from .ensemble import Graph, _descendants, _region, _subtree_pops, _tree, tree_seed

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "site" / "data" / "sci_states"
DOCS = ROOT / "docs" / "sci_states"
K = 50
COUNTY22_URL = "https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_county_500k.zip"

# today's capitals (lat, lon)
CAPITALS = {
    "AL": ("Montgomery", 32.377, -86.300), "AK": ("Juneau", 58.302, -134.420), "AZ": ("Phoenix", 33.448, -112.097),
    "AR": ("Little Rock", 34.747, -92.289), "CA": ("Sacramento", 38.576, -121.494), "CO": ("Denver", 39.739, -104.985),
    "CT": ("Hartford", 41.764, -72.682), "DE": ("Dover", 39.157, -75.520), "FL": ("Tallahassee", 30.438, -84.281),
    "GA": ("Atlanta", 33.749, -84.388), "HI": ("Honolulu", 21.307, -157.857), "ID": ("Boise", 43.618, -116.215),
    "IL": ("Springfield", 39.798, -89.654), "IN": ("Indianapolis", 39.768, -86.163), "IA": ("Des Moines", 41.591, -93.604),
    "KS": ("Topeka", 39.048, -95.678), "KY": ("Frankfort", 38.187, -84.875), "LA": ("Baton Rouge", 30.457, -91.187),
    "ME": ("Augusta", 44.307, -69.782), "MD": ("Annapolis", 38.979, -76.491), "MA": ("Boston", 42.358, -71.064),
    "MI": ("Lansing", 42.733, -84.555), "MN": ("Saint Paul", 44.955, -93.102), "MS": ("Jackson", 32.303, -90.182),
    "MO": ("Jefferson City", 38.579, -92.173), "MT": ("Helena", 46.586, -112.018), "NE": ("Lincoln", 40.808, -96.700),
    "NV": ("Carson City", 39.164, -119.766), "NH": ("Concord", 43.207, -71.538), "NJ": ("Trenton", 40.220, -74.770),
    "NM": ("Santa Fe", 35.682, -105.940), "NY": ("Albany", 42.653, -73.757), "NC": ("Raleigh", 35.780, -78.639),
    "ND": ("Bismarck", 46.821, -100.783), "OH": ("Columbus", 39.961, -82.999), "OK": ("Oklahoma City", 35.492, -97.503),
    "OR": ("Salem", 44.938, -123.030), "PA": ("Harrisburg", 40.264, -76.884), "RI": ("Providence", 41.831, -71.415),
    "SC": ("Columbia", 34.000, -81.033), "SD": ("Pierre", 44.367, -100.346), "TN": ("Nashville", 36.166, -86.784),
    "TX": ("Austin", 30.275, -97.740), "UT": ("Salt Lake City", 40.777, -111.888), "VT": ("Montpelier", 44.262, -72.581),
    "VA": ("Richmond", 37.539, -77.434), "WA": ("Olympia", 47.035, -122.905), "WV": ("Charleston", 38.336, -81.612),
    "WI": ("Madison", 43.075, -89.384), "WY": ("Cheyenne", 41.140, -104.820),
}


def _state_files():
    out = {}
    for p in sorted((ROOT / "site" / "data").glob("[A-Z][A-Z].json")):
        if p.stem.isupper():
            d = json.loads(p.read_text(encoding="utf-8"))
            out[d["fips"]] = d
    return out


def _xy(lat, lon):
    return np.column_stack([np.asarray(lon) * 111.32 * np.cos(np.radians(38.0)), np.asarray(lat) * 110.57])


def _pip(px, py, rings) -> np.ndarray:
    """Points inside a polygon given as rings (even-odd rule)."""
    inside = np.zeros(px.size, bool)
    for r in rings:
        x1, y1, x2, y2 = r[:-1, 0], r[:-1, 1], r[1:, 0], r[1:, 1]
        for a, b, c, d in zip(x1, y1, x2, y2):
            cross = ((b > py) != (d > py)) & (px < (c - a) * (py - b) / ((d - b) or 1e-300) + a)
            inside ^= cross
    return inside


def _ct_regions(lat, lon) -> np.ndarray:
    """Connecticut 2022 planning-region FIPS (int) for each point, 0 when none."""
    z = cenpop.fetch(COUNTY22_URL, cenpop.cache_dir() / "cb_2022_us_county_500k.zip", None, timeout=600)
    d = cenpop.cache_dir() / "cb_county22"
    stem = d / "cb_2022_us_county_500k"
    if not stem.with_suffix(".shp").exists():
        d.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(d)
    dbf = graphs.read_dbf(stem.with_suffix(".dbf"), ["STATEFP", "GEOID"])
    recs = np.flatnonzero(dbf["STATEFP"] == "09")
    polys = graphs.read_polygons(stem.with_suffix(".shp"), stem.with_suffix(".shx"), recs)
    out = np.zeros(len(lat), np.int64)
    px, py = np.asarray(lon, float), np.asarray(lat, float)
    for r in recs:
        out[_pip(px, py, polys[int(r)]) & (out == 0)] = int(dbf["GEOID"][r])
    if (out == 0).any():  # a centre on the shoreline: nearest assigned point
        bad, good = np.flatnonzero(out == 0), np.flatnonzero(out > 0)
        for i in bad:
            out[i] = out[good[np.argmin((px[good] - px[i]) ** 2 + (py[good] - py[i]) ** 2)]]
    return out


# --------------------------------------------------------------------------- units and graph

def build_units(split: int = 100_000) -> dict:
    """Units, their graph, SCI regions and the SCI matrices. Cached in the download cache (not committed)."""
    cache = cenpop.cache_dir() / "sci" / f"units_{split}.npz"
    meta_p = cache.with_suffix(".json")
    if cache.exists() and meta_p.exists():
        z = np.load(cache)
        return {**{k: z[k] for k in z.files}, **json.loads(meta_p.read_text(encoding="utf-8"))}
    t0 = time.perf_counter()
    rows = [r for r in cenpop.rows() if r[0] != "72"]
    geo = np.array([r[0] + r[1] + r[2] + r[3] for r in rows])
    st = np.array([r[0] for r in rows])
    pop = np.array([float(r[4]) for r in rows])
    lat = np.array([float(r[5]) for r in rows]); lon = np.array([float(r[6]) for r in rows])
    county = np.array([g[:5] for g in geo])
    cpop = {}
    for c, p in zip(county.tolist(), pop.tolist()):
        cpop[c] = cpop.get(c, 0.0) + p
    ukey = np.array([g[:11] if cpop[c] > split else c for g, c in zip(geo.tolist(), county.tolist())])
    keys, unit = np.unique(ukey, return_inverse=True)
    n = len(keys)
    print(f"{len(geo):,} block groups -> {n:,} units ({time.perf_counter() - t0:.0f}s)")

    # block-group edges: within states from site/data/graphs, across states and DC from polygons
    gi = {g: i for i, g in enumerate(geo.tolist())}
    files = _state_files()
    bg_edges = []
    for fips, d in files.items():
        g = json.loads((ROOT / "site" / "data" / "graphs" / f"{d['abbr']}.json").read_text(encoding="utf-8"))
        off = gi[geo[st == fips][0]]
        e = np.r_[np.asarray(g["edges"], np.int64), np.asarray(g["water_links"], np.int64)].reshape(-1, 2) + off
        bg_edges.append(e)
    shp, shx, dbf_p = graphs.boundary_files()
    dbf = graphs.read_dbf(dbf_p, ["STATEFP", "GEOID"])
    rec_bg = {int(r): gi.get(g) for r, g in enumerate(dbf["GEOID"].tolist()) if g in gi}
    polys = graphs.read_polygons(shp, shx, list(rec_bg))
    box = {}
    for r, rings in polys.items():
        if rings:
            a = np.vstack(rings)
            box[rec_bg[r]] = (a[:, 0].min(), a[:, 1].min(), a[:, 0].max(), a[:, 1].max())
    bgs = np.array(sorted(box)); bx = np.array([box[i] for i in bgs])
    sbox = {}
    for s_ in np.unique(st[bgs]):
        m = st[bgs] == s_
        sbox[s_] = (bx[m, 0].min(), bx[m, 1].min(), bx[m, 2].max(), bx[m, 3].max())
    by_rec = {v: k for k, v in rec_bg.items()}
    S = sorted(sbox)
    eps = 0.01
    for i, a in enumerate(S):
        for b in S[i + 1:]:
            A, B = sbox[a], sbox[b]
            x0, y0, x1, y1 = max(A[0], B[0]) - eps, max(A[1], B[1]) - eps, min(A[2], B[2]) + eps, min(A[3], B[3]) + eps
            if x0 > x1 or y0 > y1:
                continue
            m = (np.isin(st[bgs], [a, b]) & (bx[:, 2] >= x0) & (bx[:, 0] <= x1) & (bx[:, 3] >= y0) & (bx[:, 1] <= y1))
            cand = bgs[m]
            if cand.size < 2:
                continue
            pairs, _ = graphs.shared_borders({int(c): polys[by_rec[int(c)]] for c in cand})
            if len(pairs):
                bg_edges.append(pairs[st[pairs[:, 0]] != st[pairs[:, 1]]])
    dc = np.flatnonzero(st == "11")
    pairs, _ = graphs.shared_borders({int(c): polys[by_rec[int(c)]] for c in dc if int(c) in by_rec})
    bg_edges.append(pairs)
    e = np.vstack(bg_edges)
    ue = np.column_stack([unit[e[:, 0]], unit[e[:, 1]]])
    ue = ue[ue[:, 0] != ue[:, 1]]
    ue = np.unique(np.sort(ue, axis=1), axis=0)
    upop = np.bincount(unit, weights=pop, minlength=n)
    ulat = np.bincount(unit, weights=pop * lat, minlength=n) / np.maximum(upop, 1)
    ulon = np.bincount(unit, weights=pop * lon, minlength=n) / np.maximum(upop, 1)
    empty = upop == 0
    ulat[empty] = np.bincount(unit, weights=lat, minlength=n)[empty] / np.bincount(unit, minlength=n)[empty]
    ulon[empty] = np.bincount(unit, weights=lon, minlength=n)[empty] / np.bincount(unit, minlength=n)[empty]
    ustate = np.array([k_[:2] for k_ in keys])
    links, link_km, ncomp = graphs.water_links(n, ue, _xy(ulat, ulon))
    # Hawaii's nearest unit by distance is in the Aleutians, which would route Hawaii through Alaska (and make a
    # Honolulu state and a Juneau state impossible together): join Hawaii to the nearest lower-48 unit instead.
    hi = (ustate[links[:, 0]] == "15") | (ustate[links[:, 1]] == "15")
    if hi.any():
        lower = np.flatnonzero(~np.isin(ustate, ["02", "15"]))
        hon = np.flatnonzero(ustate == "15")
        h = int(hon[np.argmax(upop[hon])])
        d = (ulat[lower] - ulat[h]) ** 2 + ((ulon[lower] - ulon[h]) * np.cos(np.radians(30))) ** 2
        links = np.r_[links[~hi], [[min(h, int(lower[np.argmin(d)])), max(h, int(lower[np.argmin(d)]))]]]
    edges = np.unique(np.sort(np.r_[ue, links], axis=1), axis=0)
    print(f"{len(ue):,} unit edges, {len(links)} water links ({time.perf_counter() - t0:.0f}s)")

    # SCI regions: county, or Connecticut planning region by majority population
    region = np.array([int(c) for c in county.tolist()])
    ct = st == "09"
    region[ct] = _ct_regions(lat[ct], lon[ct])
    upick = np.zeros(n, np.int64)
    for u_ in np.unique(unit[ct]):
        m = ct & (unit == u_)
        r, w = np.unique(region[m], return_inverse=True)
        upick[u_] = r[np.argmax(np.bincount(w, weights=pop[m] + 1e-9))]
    ureg = np.array([int(k[:5]) for k in keys])
    ureg[upick > 0] = upick[upick > 0]
    regs, ur = np.unique(ureg, return_inverse=True)
    R = len(regs)
    su, sf, ss = sci.table()
    pos = {int(r): i for i, r in enumerate(regs)}
    m = np.isin(su, regs) & np.isin(sf, regs)
    M = np.zeros((R, R))
    M[np.searchsorted(regs, su[m]), np.searchsorted(regs, sf[m])] = ss[m]
    M = np.maximum(M, M.T)
    rpop = np.bincount(ur, weights=upop, minlength=R)
    rlat = np.bincount(ur, weights=upop * ulat, minlength=R) / np.maximum(rpop, 1)
    rlon = np.bincount(ur, weights=upop * ulon, minlength=R) / np.maximum(rpop, 1)
    fit = sci.distance_fit(su, sf, ss, sci.county_centres())
    la, lo = np.radians(rlat), np.radians(rlon)
    h = (np.sin((la[:, None] - la[None, :]) / 2) ** 2
         + np.cos(la)[:, None] * np.cos(la)[None, :] * np.sin((lo[:, None] - lo[None, :]) / 2) ** 2)
    km = 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))
    Madj = M / np.exp(fit["a"] + fit["b"] * np.log(np.maximum(km, 1.0)))
    missing = [int(r) for r in regs if not (su == r).any()]
    out = {"keys": keys, "unit_state": np.array([k[:2] for k in keys]), "pop": upop, "lat": ulat, "lon": ulon,
           "edges": edges, "region": ur, "regions": regs, "M_raw": M, "M_adjusted": Madj,
           "bg_unit": unit, "bg_lat": lat, "bg_lon": lon, "bg_pop": pop}
    np.savez(cache, **out)
    meta = {"split": split, "water_links": int(len(links)), "fit": fit, "regions_missing_from_sci": missing,
            "pos_unused": len(pos)}
    meta_p.write_text(json.dumps(meta), encoding="utf-8")
    return {**out, **meta}


# --------------------------------------------------------------------------- scoring

class Scorer:
    """Ties-kept score with cheap updates when two states change."""

    def __init__(self, M, region, pop, k):
        self.M0 = M.copy(); np.fill_diagonal(self.M0, 0)
        self.region, self.pop, self.k, self.R = region, pop, k, M.shape[0]
        P = np.bincount(region, weights=pop, minlength=self.R)
        self.total = float(P @ self.M0 @ P)

    def f(self, labels, s):
        m = labels == s
        return np.bincount(self.region[m], weights=self.pop[m], minlength=self.R)

    def parts(self, labels):
        return np.array([float((fs := self.f(labels, s)) @ self.M0 @ fs) for s in range(self.k)])

    def pct(self, parts):
        return 100 * float(parts.sum()) / self.total


def score_labels(U, labels, k) -> dict:
    out = {}
    for name in ("raw", "adjusted"):
        sc = Scorer(U[f"M_{name}"], U["region"], U["pop"], k)
        out[name] = round(sc.pct(sc.parts(labels)), 3)
    return out


# --------------------------------------------------------------------------- moves

def _split(nodes, eu, ev, wmax, pop, lo_a, hi_a, lo_b, hi_b, rng, caps=None, tries=40):
    """Cut a random max spanning tree so one side is in [lo_a, hi_a] and the other in [lo_b, hi_b].
    With ``caps`` (local indices of the two capitals) each side must hold exactly one. Returns a side mask."""
    n = nodes.size
    for _ in range(tries):
        t = _tree(n, eu, ev, wmax, rng)
        if t is None:
            return None
        order, pred, sub = _subtree_pops(t, pop)
        other = sub[order[0]] - sub
        ok = (((sub >= lo_a) & (sub <= hi_a) & (other >= lo_b) & (other <= hi_b))
              | ((sub >= lo_b) & (sub <= hi_b) & (other >= lo_a) & (other <= hi_a)))
        ok[order[0]] = False
        if caps is not None:
            c = np.zeros(n); c[caps[0]] = 1; c[caps[1]] = 2
            _, _, csub = _subtree_pops(t, c, root=int(order[0]))
            ok &= (csub == 1) | (csub == 2)  # the subtree holds exactly one of the two capitals
        cand = np.flatnonzero(ok)
        if cand.size:
            return _descendants(order, pred, int(cand[rng.integers(cand.size)]), n)
    return None


def _edge_km(U):
    e = U["edges"]
    la, lo = np.radians(U["lat"]), np.radians(U["lon"])
    h = (np.sin((la[e[:, 1]] - la[e[:, 0]]) / 2) ** 2
         + np.cos(la[e[:, 0]]) * np.cos(la[e[:, 1]]) * np.sin((lo[e[:, 1]] - lo[e[:, 0]]) / 2) ** 2)
    return np.maximum(2 * 6371.0 * np.arcsin(np.sqrt(np.clip(h, 0, 1))), 0.01)


def voronoi_split(n, eu, ev, length, pop, ca, cb, lo_a, hi_a, lo_b, hi_b, rng, target=None):
    """Split a connected region between two capitals (local indices ``ca``, ``cb``): order units by
    d(u, ca) - d(u, cb) (shortest paths with the given edge lengths) and give ``ca`` a prefix. Every prefix is a
    connected set holding ``ca`` and its complement a connected set holding ``cb`` (a unit on the shortest path
    from u to its capital has a smaller or equal key), so any population split is reachable up to one unit.
    Returns the side mask of ``ca`` (a random balanced prefix, or the one nearest ``target``), or None."""
    from scipy.sparse.csgraph import dijkstra
    G = sparse.csr_matrix((np.r_[length, length], (np.r_[eu, ev], np.r_[ev, eu])), shape=(n, n))
    D = dijkstra(G, indices=[ca, cb])
    if not np.isfinite(D).all():
        return None
    # units behind a chokepoint share one key; a cut inside such a tie group would break a side, so cuts are
    # allowed only between different keys (rounded to absorb floating-point noise)
    key = np.round(D[0] - D[1], 6)
    order = np.lexsort((D[0], key))
    ks = key[order]
    cum = np.cumsum(pop[order])
    tot = cum[-1]
    can = np.r_[ks[1:] > ks[:-1], False]
    ok = can & (cum >= lo_a) & (cum <= hi_a) & (tot - cum >= lo_b) & (tot - cum <= hi_b)
    if target is not None:
        c = np.flatnonzero(can)
        if not c.size:
            return None
        i = int(c[np.argmin(np.abs(cum[c] - target))])
    else:
        c = np.flatnonzero(ok)
        if not c.size:
            return None
        i = int(c[rng.integers(c.size)])
    side = np.zeros(n, bool)
    side[order[:i + 1]] = True
    if not side[ca] or side[cb]:
        return None
    return side


class Chain:
    def __init__(self, U, labels, k, tol, objective, capitals=None, beta=4.0, seed=0):
        self.U, self.k, self.tol = U, k, tol
        self.g = Graph(pop=U["pop"], edges=U["edges"])
        self.labels = np.asarray(labels, np.int64).copy()
        self.ideal = U["pop"].sum() / k
        self.rng = np.random.default_rng(seed)
        self.caps = capitals  # unit index of each state's capital (state s <-> caps[s]) or None
        self.sc = Scorer(U[f"M_{objective}"], U["region"], U["pop"], k)
        self.parts = self.sc.parts(self.labels)
        Mf = U[f"M_{objective}"]
        e, r = U["edges"], U["region"]
        s = np.log(np.maximum(Mf[r[e[:, 0]], r[e[:, 1]]], 1e-12))
        rank = np.argsort(np.argsort(s)) / max(1, len(s) - 1)
        self.wmax_all = (1.0 + beta * rank) if beta else np.ones(len(e))
        self.km = _edge_km(U)

    def score(self):
        return self.sc.pct(self.parts)

    def move(self, a=None, b=None, targets=None, sci_trees=True):
        L, e = self.labels, self.g.edges
        if a is None:
            cut = np.flatnonzero(L[e[:, 0]] != L[e[:, 1]])
            x = cut[self.rng.integers(cut.size)]
            a, b = L[e[x, 0]], L[e[x, 1]]
        mask = (L == a) | (L == b)
        nodes, eu, ev, em = _region(self.g, mask)
        p = self.g.pop[nodes]
        if targets is None:
            ta = tb = self.ideal
        else:
            ta, tb = targets
        d = self.tol * self.ideal
        caps = None
        if self.caps is not None:
            loc = {int(u): i for i, u in enumerate(nodes)}
            caps = (loc[int(self.caps[a])], loc[int(self.caps[b])])
        w = self.wmax_all[em] if sci_trees else np.ones(int(em.sum()))
        if caps is not None:
            # two-capital Voronoi split; random edge lengths for variety, strongly tied edges shorter
            length = self.km[em] * self.rng.uniform(0.3, 1.7, int(em.sum())) / w
            side = voronoi_split(nodes.size, eu, ev, length, p, caps[0], caps[1], ta - d, ta + d, tb - d, tb + d, self.rng)
        else:
            side = _split(nodes, eu, ev, w, p, ta - d, ta + d, tb - d, tb + d, self.rng, caps)
        if side is None:
            return False
        # side = subtree; give it to the state whose capital it holds (or whose target it matches)
        if caps is not None:
            sa = side if side[caps[0]] else ~side
        else:
            ps = p[side].sum()
            sa = side if abs(ps - ta) <= d else ~side
        L[nodes[sa]] = a
        L[nodes[~sa]] = b
        for s_ in (a, b):
            fs = self.sc.f(L, s_)
            self.parts[s_] = float(fs @ self.sc.M0 @ fs)
        return True

    def devs(self):
        return np.bincount(self.labels, weights=self.g.pop, minlength=self.k) / self.ideal - 1

    def neighbours(self, s):
        L, e = self.labels, self.g.edges
        m = (L[e[:, 0]] == s) ^ (L[e[:, 1]] == s)
        x = e[m]
        return np.unique(np.where(L[x[:, 0]] == s, L[x[:, 1]], L[x[:, 0]]))


def _improving_split(ch: Chain, a, b, tries=30):
    """Re-split states a and b (capitals kept apart) at the cut closest to their average population, when that
    lowers the sum of their squared deviations (pairwise averaging: repeated, it evens out every state)."""
    L = ch.labels
    mask = (L == a) | (L == b)
    nodes, eu, ev, em = _region(ch.g, mask)
    p = ch.g.pop[nodes]
    tot, I = p.sum(), ch.ideal
    old = (p[L[nodes] == a].sum() / I - 1) ** 2 + (p[L[nodes] == b].sum() / I - 1) ** 2
    if old <= 2 * ch.tol ** 2 and max(abs(p[L[nodes] == a].sum() / I - 1), abs(p[L[nodes] == b].sum() / I - 1)) <= ch.tol:
        return False
    loc = {int(u): i for i, u in enumerate(nodes)}
    c0, c1 = loc[int(ch.caps[a])], loc[int(ch.caps[b])]
    c = np.zeros(nodes.size); c[c0] = 1; c[c1] = 2
    best, best_side = old - 1e-5, None
    for _ in range(tries):
        t = _tree(nodes.size, eu, ev, ch.wmax_all[em], ch.rng)
        order, pred, sub = _subtree_pops(t, p)
        _, _, cs = _subtree_pops(t, c, root=int(order[0]))
        ok = (cs == 1) | (cs == 2)
        ok[order[0]] = False
        cand = np.flatnonzero(ok)
        if not cand.size:
            continue
        dev = (sub[cand] / I - 1) ** 2 + ((tot - sub[cand]) / I - 1) ** 2
        x = int(np.argmin(dev))
        if dev[x] < best:
            best = dev[x]
            side = _descendants(order, pred, int(cand[x]), nodes.size)
            best_side = side if side[c0] else ~side
            if abs(sub[cand[x]] - tot / 2) <= ch.tol * I / 2:
                break
    if best_side is None:
        return False
    L[nodes[best_side]] = a
    L[nodes[~best_side]] = b
    for s_ in (a, b):
        fs = ch.sc.f(L, s_)
        ch.parts[s_] = float(fs @ ch.sc.M0 @ fs)
    return True


def balance(ch: Chain, log=print, limit=100_000):
    """Pairwise averaging until every state is within tol: adjacent states with the largest population gap are
    re-split nearer their average (capitals kept apart). Each step lowers the sum of squared deviations."""
    e = ch.g.edges
    for it in range(limit):
        dv = ch.devs()
        if np.abs(dv).max() <= ch.tol:
            log(f"  balanced after {it} rounds; max deviation {100 * np.abs(dv).max():.2f}%")
            return
        L = ch.labels
        x = L[e[:, 0]] != L[e[:, 1]]
        pairs = np.unique(np.sort(np.column_stack([L[e[x, 0]], L[e[x, 1]]]), axis=1), axis=0)
        gap = np.abs(dv[pairs[:, 0]] - dv[pairs[:, 1]])
        # favour pairs that include a state outside the tolerance
        gap = gap + (np.maximum(np.abs(dv[pairs[:, 0]]), np.abs(dv[pairs[:, 1]])) > ch.tol)
        moved = False
        for q in np.argsort(-gap)[:25]:
            if _improving_split(ch, int(pairs[q, 0]), int(pairs[q, 1])):
                moved = True
                break
        if not moved:
            old = ch.tol; ch.tol = 1.0; ch.move(); ch.tol = old
        if it % 200 == 0:
            log(f"  balancing: round {it}, max deviation {100 * np.abs(dv).max():.1f}%")
    raise RuntimeError("balancing did not converge")


def capital_seed(U, caps, k, tol, wmax, seed=0, tries=12, log=print) -> np.ndarray:
    """Recursive spanning-tree split with backtracking: a region holding m capitals is cut along a random spanning
    tree (weights up to ``wmax``, favouring strongly tied edges) into two connected pieces holding m1 and m - m1
    capitals and, within 3 tol of a state (absolute), m1 and m - m1 states' worth of people. When a piece cannot be
    split further, the cut above it is drawn again. Ends with one capital per state; ``balance`` then brings every
    state within tol."""
    import sys
    sys.setrecursionlimit(10000)
    rng = np.random.default_rng(seed)
    g = Graph(pop=U["pop"], edges=U["edges"])
    I = U["pop"].sum() / k
    capmark = np.full(g.n, -1, np.int64)
    capmark[caps] = np.arange(k)
    labels = np.full(g.n, -1, np.int64)
    delta = 3 * tol * I
    calls = [0]

    def rec(mask) -> bool:
        calls[0] += 1
        if calls[0] > 20000:
            return False
        nodes, eu, ev, em = _region(g, mask)
        cm = capmark[nodes]
        m = int((cm >= 0).sum())
        if m == 1:
            labels[nodes] = cm[cm >= 0][0]
            return True
        p = g.pop[nodes]
        c = (cm >= 0).astype(float)
        for _ in range(tries if m < k else 50):
            t = _tree(nodes.size, eu, ev, wmax[em], rng)
            order, pred, sub = _subtree_pops(t, p)
            _, _, cs = _subtree_pops(t, c, root=int(order[0]))
            cs = np.rint(cs).astype(int)
            ok = ((cs >= 1) & (cs <= m - 1) & (np.abs(sub - cs * I) <= delta)
                  & (np.abs((p.sum() - sub) - (m - cs) * I) <= delta))
            ok[order[0]] = False
            cand = np.flatnonzero(ok)
            if not cand.size:
                continue
            side = _descendants(order, pred, int(cand[rng.integers(cand.size)]), nodes.size)
            a = np.zeros(g.n, bool); a[nodes[side]] = True
            b = np.zeros(g.n, bool); b[nodes[~side]] = True
            if rec(a) and rec(b):
                return True
        return False

    if not rec(np.ones(g.n, bool)):
        raise RuntimeError(f"no capital-respecting split found ({calls[0]} region splits tried)")
    log(f"  start: {calls[0]} region splits")
    return labels


def fix_capitals(U, labels, caps, k, tol, wmax, seed=0, max_moves=400_000, log=print) -> np.ndarray:
    """From a balanced plan, move capitals between neighbouring states until each state holds exactly one.

    A state with two or more capitals re-splits with the neighbour nearest (in steps between states) to a state
    with none, at a balanced cut (every state stays within tol) that hands that neighbour one more capital. Surplus
    capitals thus travel toward the states that lack one. Every move is a ReCom move, so states stay connected."""
    rng = np.random.default_rng(seed)
    g = Graph(pop=U["pop"], edges=U["edges"])
    L = np.asarray(labels, np.int64).copy()
    I = U["pop"].sum() / k
    lo, hi = I * (1 - tol), I * (1 + tol)
    isc = np.zeros(g.n); isc[caps] = 1
    e = g.edges
    for it in range(max_moves):
        cnt = np.bincount(L[caps], minlength=k)
        pen = int(np.abs(cnt - 1).sum())
        if pen == 0:
            log(f"  capitals placed after {it} moves")
            perm = np.empty(k, np.int64)
            perm[L[caps]] = np.arange(k)  # state s is the state of capital s (Chain relies on it)
            return perm[L]
        x = L[e[:, 0]] != L[e[:, 1]]
        sp = np.unique(np.sort(np.column_stack([L[e[x, 0]], L[e[x, 1]]]), axis=1), axis=0)
        adj = [[] for _ in range(k)]
        for a_, b_ in sp.tolist():
            adj[a_].append(b_); adj[b_].append(a_)
        dist = np.full(k, 10 ** 6)
        frontier = [s for s in range(k) if cnt[s] == 0]
        for s in frontier:
            dist[s] = 0
        while frontier:
            nxt = []
            for s in frontier:
                for t2 in adj[s]:
                    if dist[t2] > dist[s] + 1:
                        dist[t2] = dist[s] + 1; nxt.append(t2)
            frontier = nxt
        surplus = np.array([s for s in range(k) if cnt[s] >= 2])
        wts = np.exp(-dist[surplus].astype(float)); wts /= wts.sum()
        s = int(rng.choice(surplus, p=wts))  # surplus capitals nearest to a state without one move first
        nbs = sorted(adj[s], key=lambda t2: (dist[t2], rng.random()))
        if rng.random() < 0.2:
            rng.shuffle(nbs)
        moved = False
        for t2 in nbs[:3]:
            mask = (L == s) | (L == t2)
            nodes, eu, ev, em = _region(g, mask)
            p = g.pop[nodes]; c = isc[nodes]
            want_t = cnt[t2] + 1
            for _ in range(20):
                tr = _tree(nodes.size, eu, ev, wmax[em], rng)
                order, pred, sub = _subtree_pops(tr, p)
                _, _, cs = _subtree_pops(tr, c, root=int(order[0]))
                cs = np.rint(cs).astype(int)
                other = p.sum() - sub
                ok = (sub >= lo) & (sub <= hi) & (other >= lo) & (other <= hi) & (cs == want_t)
                ok[order[0]] = False
                cand = np.flatnonzero(ok)
                if cand.size:
                    side = _descendants(order, pred, int(cand[rng.integers(cand.size)]), nodes.size)
                    L[nodes[side]] = t2
                    L[nodes[~side]] = s
                    moved = True
                    break
            if moved:
                break
        if it % 500 == 0:
            log(f"  placing capitals: move {it}, {pen} off")
    raise RuntimeError("could not give every state one capital")


def balance_units(U, labels, caps, k, tol=0.01, log=print, max_rounds=100_000) -> np.ndarray:
    """Even out state populations by moving single border units, never moving a capital and never cutting a state.

    Each round the most over-populated state that can make progress hands border units to the neighbouring state
    nearest (in steps between states) to an under-populated one, up to its surplus (or that state's deficit);
    the receiving state passes it on in later rounds. A unit may leave a state only when its neighbours in that
    state stay linked among themselves through its other neighbours (a cheap local test that is sufficient for
    the state to stay in one piece). Like ``blocks.Plan.balance`` but local and fast enough for 67,000 units.
    Stops when every state is within ``tol``."""
    from collections import deque
    L = np.asarray(labels, np.int64).copy()
    pop = U["pop"]
    n = len(pop)
    e = U["edges"]
    A = sparse.csr_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n))
    nb = [A.indices[A.indptr[u]:A.indptr[u + 1]].tolist() for u in range(n)]
    nbs = [set(x) for x in nb]
    frozen = set(int(c) for c in caps)
    I = pop.sum() / k
    P = np.bincount(L, weights=pop, minlength=k)

    def removable(u):
        if u in frozen:
            return False
        s = L[u]
        same = [v for v in nb[u] if L[v] == s]
        if len(same) <= 1:
            return len(same) == 1
        # are u's same-state neighbours linked among themselves without u (through each other or a shared neighbour)?
        seen, dq = {same[0]}, deque([same[0]])
        goal = set(same[1:])
        local = set(same)
        for v in same:
            local |= {w for w in nb[v] if L[w] == s and w != u and (nbs[w] & nbs[u])}
        while dq and goal:
            x = dq.popleft()
            for w in nb[x]:
                if w in local and w not in seen and w != u:
                    seen.add(w); goal.discard(w); dq.append(w)
        return not goal

    blocked = set()
    moved_units = clears = 0
    for rnd in range(max_rounds):
        dv = P / I - 1
        if np.abs(dv).max() <= tol:
            log(f"  balanced in {rnd} rounds, {moved_units:,} units moved; max deviation {100 * np.abs(dv).max():.3f}%")
            return L
        cut = L[e[:, 0]] != L[e[:, 1]]
        sp = np.unique(np.sort(np.column_stack([L[e[cut, 0]], L[e[cut, 1]]]), axis=1), axis=0)
        adj = [set() for _ in range(k)]
        for a_, b_ in sp.tolist():
            adj[a_].add(b_); adj[b_].add(a_)
        dist = np.full(k, 1 << 20)
        q = deque(int(s) for s in np.flatnonzero(dv < 0))
        for s in q:
            dist[s] = 0
        while q:
            x = q.popleft()
            for y in adj[x]:
                if dist[y] > dist[x] + 1:
                    dist[y] = dist[x] + 1; q.append(y)
        progress = False
        for a in np.argsort(-dv):
            a = int(a)
            if dv[a] <= 0:
                break
            for y in sorted(adj[a], key=lambda y: (dist[y], P[y])):
                if dist[y] > dist[a] or (a, y) in blocked:
                    continue
                amount = P[a] - I  # pass the whole surplus on; the receiving state passes on what it cannot keep
                slack = 0.3 * tol * I
                border = {u for u in np.flatnonzero(L == a).tolist() if any(L[v] == y for v in nb[u])}
                sent = 0.0
                while border and sent < amount - slack:
                    # most edges into y first (keeps the new border short)
                    cand = sorted(border, key=lambda u: -sum(L[v] == y for v in nb[u]))
                    took = False
                    for u in cand[:200]:
                        if pop[u] > amount - sent + slack or not removable(u):
                            continue
                        L[u] = y; P[a] -= pop[u]; P[y] += pop[u]; sent += pop[u]; moved_units += 1
                        border.discard(u)
                        border |= {v for v in nb[u] if L[v] == a}
                        took = True
                        break
                    if not took:
                        break
                if sent > 0:
                    progress = True
                    break
                blocked.add((a, y))
            if progress:
                break
        if not progress:
            if blocked and clears < 3:
                blocked.clear(); clears += 1
                continue
            raise RuntimeError(f"balancing stuck at max deviation {100 * np.abs(dv).max():.2f}%")
        clears = 0
        if rnd % 50 == 0:
            log(f"  balancing round {rnd}: max deviation {100 * np.abs(dv).max():.1f}%, {moved_units:,} units moved")
    raise RuntimeError("balancing did not converge")


def capital_assignment(U, caps, k, iters=4000, alpha=1.0, log=print) -> np.ndarray:
    """Each unit joins the capital its people are most tied to, with a price per capital adjusted until every
    state has about the same population (a balanced assignment, solved by dual ascent).

    Cost of unit u for capital s = -log SCI(region of u, region of capital s) + alpha log(km to the capital) minus
    the capital's price (raw SCI falls with distance, and the distance term separates tracts of one county, so the
    pieces are mostly contiguous); prices of under-populated states rise. Capital
    units always stay with their own capital. Returns labels (state s = capital s); pieces cut off from their
    capital are fixed afterwards by ``attach_pieces``."""
    M, r, pop = U["M_raw"], U["region"], U["pop"]
    I = pop.sum() / k
    rc = r[caps]
    C = -np.log(np.maximum(M[r][:, rc], 1e-9))  # (n, k)
    # plus distance to the capital, so tracts of one county (same SCI row) are not all-or-nothing
    la, lo = np.radians(U["lat"]), np.radians(U["lon"])
    h = (np.sin((la[:, None] - la[caps][None, :]) / 2) ** 2
         + np.cos(la)[:, None] * np.cos(la[caps])[None, :] * np.sin((lo[:, None] - lo[caps][None, :]) / 2) ** 2)
    C += alpha * np.log(np.maximum(2 * 6371.0 * np.arcsin(np.sqrt(np.clip(h, 0, 1))), 1.0))
    lam = np.zeros(k)
    best, best_lab = np.inf, None
    for it in range(iters):
        lab = np.argmin(C - lam[None, :], axis=1)
        lab[caps] = np.arange(k)
        P = np.bincount(lab, weights=pop, minlength=k)
        dev = np.abs(P / I - 1).max()
        if dev < best:
            best, best_lab = dev, lab.copy()
        if dev < 0.005:
            break
        lam += (0.5 / np.sqrt(1 + it)) * (1 - P / I)
    log(f"  capital assignment: max deviation {100 * best:.2f}% after {it + 1} price updates")
    return best_lab


def attach_pieces(U, labels, caps, k) -> np.ndarray:
    """Make every state one piece: pieces not holding the state's capital join the neighbouring state they share
    most edges with (smallest pieces first)."""
    L = np.asarray(labels, np.int64).copy()
    n, e = len(L), U["edges"]
    capset = np.zeros(n, bool); capset[caps] = True
    for _ in range(100_000):
        same = e[L[e[:, 0]] == L[e[:, 1]]]
        a = sparse.coo_matrix((np.ones(len(same)), (same[:, 0], same[:, 1])), shape=(n, n))
        nc, comp = connected_components(a, directed=False)
        if nc == k:
            return L
        has_cap = np.zeros(nc, bool); has_cap[comp[caps]] = True
        orphans = np.flatnonzero(~has_cap)
        size = np.bincount(comp, weights=U["pop"] + 1e-6, minlength=nc)
        c = int(orphans[np.argmin(size[orphans])])
        members = comp == c
        x = e[members[e[:, 0]] ^ members[e[:, 1]]]
        outside = np.where(members[x[:, 0]], x[:, 1], x[:, 0])
        if not outside.size:
            raise RuntimeError("an isolated piece")
        L[members] = np.bincount(L[outside], minlength=k).argmax()
    raise RuntimeError("could not join pieces")


def capital_voronoi(U, caps, k, iters=20000, log=print) -> np.ndarray:
    """Additively weighted graph Voronoi from the capitals, with weights tuned for equal population.

    Unit u joins the capital s minimising d(u, s) - w_s, where d is the shortest-path length through the unit
    graph (edges weighted by km between unit centres). Each such cell is connected: every unit on u's shortest
    path to s also prefers s. Weights of under-populated states rise until all are within about one unit's
    population. Capital units stay with their capital."""
    from scipy.sparse.csgraph import dijkstra
    e, n, pop = U["edges"], len(U["pop"]), U["pop"]
    la, lo = np.radians(U["lat"]), np.radians(U["lon"])
    h = (np.sin((la[e[:, 1]] - la[e[:, 0]]) / 2) ** 2
         + np.cos(la[e[:, 0]]) * np.cos(la[e[:, 1]]) * np.sin((lo[e[:, 1]] - lo[e[:, 0]]) / 2) ** 2)
    km = np.maximum(2 * 6371.0 * np.arcsin(np.sqrt(np.clip(h, 0, 1))), 0.01)
    G = sparse.csr_matrix((np.r_[km, km], (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n))
    D = dijkstra(G, indices=caps).T  # (n, k)
    I = pop.sum() / k
    w = np.zeros(k)
    best, best_lab = np.inf, None
    step = 100.0
    for it in range(iters):
        lab = np.argmin(D - w[None, :], axis=1)
        lab[caps] = np.arange(k)
        P = np.bincount(lab, weights=pop, minlength=k)
        dev = np.abs(P / I - 1).max()
        if dev < best:
            best, best_lab = dev, lab.copy()
        if dev < 0.01:
            break
        w += step * (1 - P / I)
        if it % 500 == 499:
            step *= 0.9
    log(f"  capital Voronoi: max deviation {100 * best:.2f}% after {it + 1} weight updates")
    return best_lab


def balance_pairs(ch: Chain, log=print, limit=200_000):
    """Pairwise averaging with two-capital Voronoi splits until every state is within tol: adjacent pairs, largest
    population gap first (states outside tol first), are re-split near their average; a split is kept only when it
    lowers the pair's sum of squared deviations (some pairs cannot be split finely, e.g. when both reach part of
    their region only through one chokepoint). When no pair improves, a random capital-respecting move shakes
    things up. Each split keeps both states connected around their capitals."""
    e = ch.g.edges
    I = ch.ideal
    stuck = 0
    for it in range(limit):
        dv = ch.devs()
        if np.abs(dv).max() <= ch.tol:
            log(f"  balanced after {it} rounds; max deviation {100 * np.abs(dv).max():.2f}%")
            return
        L = ch.labels
        x = L[e[:, 0]] != L[e[:, 1]]
        pairs = np.unique(np.sort(np.column_stack([L[e[x, 0]], L[e[x, 1]]]), axis=1), axis=0)
        gap = np.abs(dv[pairs[:, 0]] - dv[pairs[:, 1]])
        gap = gap + (np.maximum(np.abs(dv[pairs[:, 0]]), np.abs(dv[pairs[:, 1]])) > ch.tol)
        moved = False
        for q in np.argsort(-gap)[:40]:
            a, b = int(pairs[q, 0]), int(pairs[q, 1])
            if gap[q] < 1e-4:
                break
            mask = (L == a) | (L == b)
            nodes, eu, ev, em = _region(ch.g, mask)
            p = ch.g.pop[nodes]
            loc = {int(u): i for i, u in enumerate(nodes)}
            length = ch.km[em] * ch.rng.uniform(0.7, 1.3, int(em.sum()))
            side = voronoi_split(nodes.size, eu, ev, length, p, loc[int(ch.caps[a])], loc[int(ch.caps[b])],
                                 0, np.inf, 0, np.inf, ch.rng, target=p.sum() / 2)
            if side is None:
                continue
            new = (p[side].sum() / I - 1) ** 2 + (p[~side].sum() / I - 1) ** 2
            if new >= dv[a] ** 2 + dv[b] ** 2 - 1e-9:
                continue
            L[nodes[side]] = a
            L[nodes[~side]] = b
            for s_ in (a, b):
                fs = ch.sc.f(L, s_)
                ch.parts[s_] = float(fs @ ch.sc.M0 @ fs)
            moved = True
            break
        if not moved:
            stuck += 1
            old = ch.tol; ch.tol = 0.5
            ch.move()
            ch.tol = old
        if it % 500 == 0:
            log(f"  balancing: round {it}, max deviation {100 * np.abs(dv).max():.1f}%, {stuck} shakes")
    raise RuntimeError("balancing did not converge")


def grow_from_capitals(U, caps, k, objective) -> np.ndarray:
    """Each state grows from its capital; the smallest state takes the frontier unit it is most tied to."""
    n = len(U["pop"])
    e = U["edges"]
    A = sparse.csr_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n))
    ind, ptr = A.indices, A.indptr
    M, r, pop = U[f"M_{objective}"], U["region"], U["pop"]
    labels = np.full(n, -1, np.int64)
    aff = np.zeros((k, M.shape[0]))
    P = np.zeros(k)
    frontier = [set() for _ in range(k)]
    for s, u in enumerate(caps):
        labels[u] = s; P[s] += pop[u]; aff[s] += pop[u] * M[:, r[u]]
        frontier[s].update(ind[ptr[u]:ptr[u + 1]].tolist())
    left = n - k
    active = set(range(k))
    while left:
        s = min(active, key=lambda x: P[x])
        fr = [v for v in frontier[s] if labels[v] < 0]
        frontier[s] = set(fr)
        if not fr:
            active.discard(s)
            if not active:
                raise RuntimeError("growth stalled")
            continue
        fr = np.array(fr)
        u = int(fr[np.argmax(aff[s, r[fr]])])
        labels[u] = s; P[s] += pop[u]; aff[s] += pop[u] * M[:, r[u]]
        frontier[s].discard(u)
        frontier[s].update(v for v in ind[ptr[u]:ptr[u + 1]].tolist() if labels[v] < 0)
        left -= 1
    return labels


def short_bursts(ch: Chain, bursts: int, length: int, log=print):
    best, best_labels, best_parts = ch.score(), ch.labels.copy(), ch.parts.copy()
    t0 = time.perf_counter()
    for i in range(bursts):
        ch.labels, ch.parts = best_labels.copy(), best_parts.copy()
        bb, bl, bp = -1.0, None, None
        done = tries = 0
        while done < length and tries < 20 * length:
            tries += 1
            if ch.move():
                done += 1
                s = ch.score()
                if s > bb:
                    bb, bl, bp = s, ch.labels.copy(), ch.parts.copy()
        if bl is not None and bb >= best:
            best, best_labels, best_parts = bb, bl, bp
        if (i + 1) % 100 == 0:
            log(f"  burst {i + 1}/{bursts}: {best:.3f}% ({time.perf_counter() - t0:.0f}s)")
    ch.labels, ch.parts = best_labels, best_parts
    return best


# --------------------------------------------------------------------------- output

def _names(U):
    files = _state_files()
    cn = {c["fips"]: (c["name"], d["abbr"]) for d in files.values() for c in d["counties"]}
    cn["11001"] = ("District of Columbia", "DC")
    return cn, {d["fips"]: d["abbr"] for d in files.values()}


def write(U, labels, name, info, current):
    k = int(labels.max()) + 1
    cn, abbr = _names(U)
    keys, pop = U["keys"], U["pop"]
    ideal = pop.sum() / k
    states = []
    for s in range(k):
        m = labels == s
        cp = {}
        for key, p in zip(keys[m].tolist(), pop[m].tolist()):
            cp[key[:5]] = cp.get(key[:5], 0) + p
        top = max(cp, key=cp.get)
        nm, ab = cn.get(top, (top, abbr.get(top[:2], "")))
        cap = info.get("capital_names", {}).get(s)
        states.append({"id": s, "name": cap[0] if cap else f"{nm} ({ab})", **({"capital": cap[1]} if cap else {}),
                       "largest_county": f"{nm}, {ab}", "pop": int(pop[m].sum()),
                       "dev_pct": round(100 * (pop[m].sum() / ideal - 1), 3),
                       "today_states": sorted({abbr.get(x[:2], x[:2]) for x in keys[m].tolist()}),
                       "units": ",".join(keys[m].tolist())})
    res = {"v": 1, "name": name, **{k_: v for k_, v in info.items() if k_ != "capital_names"},
           "ties_kept_pct": score_labels(U, labels, k), "today": current, "states": states}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(res, separators=(",", ":")), encoding="utf-8")
    draw(U, labels, DOCS / f"{name}.png", info.get("title", name), res)
    return res


_POLY_CACHE = {}


def _bg_polygons(U):
    """Block-group outlines in ``U`` order (lon, lat rings), from the Census boundary file."""
    if "p" in _POLY_CACHE:
        return _POLY_CACHE["p"]
    rows = [r for r in cenpop.rows() if r[0] != "72"]
    geo = [r[0] + r[1] + r[2] + r[3] for r in rows]
    shp, shx, dbf_p = graphs.boundary_files()
    dbf = graphs.read_dbf(dbf_p, ["GEOID"])
    rec = {g: i for i, g in enumerate(dbf["GEOID"].tolist())}
    idx = [rec.get(g) for g in geo]
    polys = graphs.read_polygons(shp, shx, [i for i in idx if i is not None])
    out = [polys[i] if i is not None else [] for i in idx]
    _POLY_CACHE["p"] = out
    return out


def _colours(labels, edges, k):
    """Colour states so neighbours differ (greedy, largest degree first), from a 10-colour palette."""
    pal = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac"]
    x = labels[edges[:, 0]] != labels[edges[:, 1]]
    adj = [set() for _ in range(k)]
    for a_, b_ in np.unique(np.sort(np.column_stack([labels[edges[x, 0]], labels[edges[x, 1]]]), axis=1), axis=0).tolist():
        adj[a_].add(b_); adj[b_].add(a_)
    col = [-1] * k
    for s in sorted(range(k), key=lambda s: -len(adj[s])):
        used = {col[t] for t in adj[s]}
        col[s] = next(c for c in range(len(pal)) if c not in used)
    return [pal[c] for c in col]


def draw(U, labels, path, title, res):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.collections import PolyCollection
    except ImportError:
        return
    k = int(labels.max()) + 1
    lab = labels[U["bg_unit"]]
    cols = _colours(labels, U["edges"], k)
    polys = _bg_polygons(U)
    lat, lon, bpop = U["bg_lat"], U["bg_lon"], U["bg_pop"]
    fig = plt.figure(figsize=(16, 10), dpi=120)
    fig.patch.set_facecolor("white")

    def panel(ax, sel, lat0, xform=None):
        verts, fc = [], []
        c0 = np.cos(np.radians(lat0))
        for i in np.flatnonzero(sel):
            for r in polys[i]:
                x = np.where(r[:, 0] > 0, r[:, 0] - 360, r[:, 0])  # Aleutians across the 180th meridian
                v = np.column_stack([x * c0, r[:, 1]])
                if xform is not None:
                    v = xform(v)
                verts.append(v); fc.append(cols[lab[i]])
        ax.add_collection(PolyCollection(verts, facecolors=fc, edgecolors=fc, linewidths=0.15))
        ax.autoscale_view(); ax.set_aspect("equal"); ax.axis("off")

    main = (lon > -130) & (lat > 24) & (lat < 50)
    ax = fig.add_axes([0.0, 0.05, 1.0, 0.85])
    panel(ax, main, 38)
    c0 = np.cos(np.radians(38))
    for s in range(k):
        m = main & (lab == s)
        nm = res["states"][s]["name"].split(" (")[0]
        ab = res["states"][s]["name"].split("(")[-1].rstrip(")")
        if res.get("capitals_fixed") and ab in CAPITALS:
            # capitals version: label each state at its capital (its people may live far from it)
            _, la_, lo_ = CAPITALS[ab]
            if 24 < la_ < 50:
                ax.plot(lo_ * c0, la_, "o", ms=3, color="#111")
                ax.text(lo_ * c0, la_ + 0.35, nm, fontsize=6.5, ha="center", va="bottom", weight="bold", color="#111",
                        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75))
            continue
        if m.any():
            w = bpop[m] + 1
            ax.text(np.average(lon[m] * c0, weights=w), np.average(lat[m], weights=w), nm, fontsize=6.5, ha="center",
                    va="center", weight="bold", color="#111",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75))
    ak = fig.add_axes([0.01, 0.04, 0.2, 0.22]); panel(ak, (lat > 50), 60)
    hi = fig.add_axes([0.22, 0.05, 0.1, 0.1]); panel(hi, (lat < 24) & (lon < -150), 20)
    tk, td = res["ties_kept_pct"], res["today"]
    nq = res.get("neutral_quantiles_5_50_95", {})
    sub = (f"Cross-county friendships kept inside one state: {tk['raw']:.1f}% raw, {tk['adjusted']:.2f}% distance-adjusted. "
           f"Neutral equal-population states: {nq.get('raw', [0, 0, 0])[1]:.1f}% raw. "
           f"Today's states (very unequal populations): {td['raw']:.1f}%.")
    fig.suptitle(title + "\n" + sub, fontsize=11)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def report() -> None:
    """docs/SCI_STATES.md from the saved variants."""
    names = {"free.raw": "Free, raw SCI", "free.adjusted": "Free, distance-adjusted SCI",
             "capitals.raw": "Today's capitals, raw SCI", "capitals.adjusted": "Today's capitals, distance-adjusted SCI"}
    res = {v: json.loads((OUT / f"{v}.json").read_text(encoding="utf-8")) for v in names if (OUT / f"{v}.json").exists()}
    if not res:
        return
    any_ = next(iter(res.values()))
    nq, today = any_["neutral_quantiles_5_50_95"], any_["today"]
    fr = res.get("free.raw", any_)
    lines = ["# Fifty states drawn by social connectedness", "", "## In short (for people)", "",
             "The country redrawn as 50 states of equal population (within 1%), each in one piece, putting together "
             "the places whose people are most connected on Facebook (Meta's Social Connectedness Index). Two versions: "
             "states that form anywhere, and states that keep today's 50 capitals (one each; only the borders move).",
             "",
             f"- The best map keeps **{fr['ties_kept_pct']['raw']:.1f}%** of friendships between people in different "
             f"counties inside one state, against {nq['raw'][1]:.1f}% for typical equal-population states drawn without "
             f"the index (5–95%: {nq['raw'][0]:.1f}–{nq['raw'][2]:.1f}%). Today's states keep {today['raw']:.1f}%, but "
             "they are very unequal in size (California is 68 times Wyoming), and big states keep more ties by being big.",
             *([f"- Keeping today's capitals costs ties: the best capitals map keeps "
                f"{res['capitals.raw']['ties_kept_pct']['raw']:.1f}%, about what neutral states keep. Equal population "
                "forces capitals far from big populations to reach them: in this run, interior western capitals' states "
                "run into California and four southeastern capitals' states run down the Florida peninsula in strips."]
               if "capitals.raw" in res else []),
             "- The raw index mostly follows distance, so the raw maps are compact regions around big metro areas. The "
             "distance-adjusted index rewards ties that distance does not explain (migration, family, history); as a "
             "rule for whole states it gives sprawling shapes, because most adjusted weight joins places too far apart to "
             "share any state.",
             "- These maps show how the country is connected; the index follows migration, party and race, so they are "
             "a lens, not a neutral proposal.", "",
             "| Version | Ties kept, raw | Ties kept, distance-adjusted | Largest population gap | Map |",
             "| --- | --: | --: | --: | --- |"]
    for v, r in res.items():
        lines.append(f"| {names[v]} | {r['ties_kept_pct']['raw']:.1f}% | {r['ties_kept_pct']['adjusted']:.2f}% | "
                     f"{r['max_dev_pct']:.2f}% | [{v}.png](sci_states/{v}.png) |")
    lines += ["", f"Comparison: neutral equal-population states {nq['raw'][1]:.1f}% raw / {nq['adjusted'][1]:.2f}% adjusted; "
              f"today's states {today['raw']:.1f}% / {today['adjusted']:.2f}%.", ""]
    for v in res:
        lines += [f"![{names[v]}](sci_states/{v}.png)", ""]
    lines += ["---", "", "## For AI agents (detailed)", "",
              "Generated by `python -m redistricting_addons.sci_states` (method in its docstring). Data per version: "
              "`site/data/sci_states/<version>.json` (states with name, capital, population, deviation, today's states they "
              "draw from and unit GEOIDs: counties, or tracts in counties above 100,000 people). Score = share of "
              "cross-county friendship weight (SCI × both populations, own-county pairs left out) inside one state. "
              "Optimiser: short bursts of ReCom moves (free) or two-capital Voronoi splits (capitals), favouring strongly "
              "tied edges.", ""]
    for v, r in res.items():
        lines += [f"### {names[v]}", "", "| State | Population | Gap | Draws from today's |", "| --- | --: | --: | --- |"]
        for s in sorted(r["states"], key=lambda s: s["name"]):
            lines.append(f"| {s['name']} | {s['pop']:,} | {s['dev_pct']:+.2f}% | {', '.join(s['today_states'])} |")
        lines.append("")
    (ROOT / "docs" / "SCI_STATES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- main

def main(argv=None) -> int:
    import functools, builtins
    globals()["print"] = functools.partial(builtins.print, flush=True)
    ap = argparse.ArgumentParser(prog="python -m redistricting_addons.sci_states", description=__doc__.split("\n\n")[0])
    ap.add_argument("--split", type=int, default=100_000, help="counties above this population are split into tracts")
    ap.add_argument("--tol", type=float, default=0.01, help="population tolerance, fraction of 1/50 of the total")
    ap.add_argument("--bursts", type=int, default=1500)
    ap.add_argument("--burst", type=int, default=10, help="ReCom moves per burst")
    ap.add_argument("--variants", nargs="*", default=["free.adjusted", "free.raw", "capitals.adjusted", "capitals.raw"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--redraw", action="store_true", help="only redraw docs/sci_states/<variant>.png from the saved files")
    ap.add_argument("--resume", action="store_true",
                    help="start from the saved site/data/sci_states/<variant>.json and keep optimising")
    a = ap.parse_args(argv)
    U = build_units(a.split)
    n, k = len(U["pop"]), K
    if a.redraw:
        pos = {key: i for i, key in enumerate(U["keys"].tolist())}
        for v in a.variants:
            prev = OUT / f"{v}.json"
            if not prev.exists():
                continue
            res = json.loads(prev.read_text(encoding="utf-8"))
            lab = np.full(n, -1, np.int64)
            for s_ in res["states"]:
                lab[[pos[x] for x in s_["units"].split(",")]] = s_["id"]
            draw(U, lab, DOCS / f"{v}.png", res["title"], res)
            print(f"drew docs/sci_states/{v}.png")
        report()
        return 0
    files = _state_files()
    fips_of = {d["abbr"]: f for f, d in files.items()}
    order = sorted(fips_of)
    today = np.array([order.index(files[s_]["abbr"]) if s_ in files else -1 for s_ in U["unit_state"].tolist()])
    today[today < 0] = order.index("MD")  # DC with Maryland, for today's score
    current = score_labels(U, today, k)
    print(f"today's 50 states keep {current['adjusted']}% (distance-adjusted) / {current['raw']}% (raw) of cross-county ties")
    g = Graph(pop=U["pop"], edges=U["edges"])
    free0 = tree_seed(g, k, a.tol, seed=a.seed)
    # neutral comparison: plain ReCom from the free start, no SCI
    ch = Chain(U, free0, k, a.tol, "adjusted", beta=0, seed=a.seed + 5)
    neutral = {"raw": [], "adjusted": []}
    for i in range(400):
        while not ch.move(sci_trees=False):
            pass
        if (i + 1) % 20 == 0:
            for nm, v in score_labels(U, ch.labels, k).items():
                neutral[nm].append(v)
    nq = {nm: [round(float(x), 3) for x in np.percentile(v, [5, 50, 95])] for nm, v in neutral.items()}
    print(f"neutral equal-population states (5/50/95%): {nq}")
    # capitals: unit nearest each capital within its state
    caps, cap_names = [], {}
    for s, ab in enumerate(order):
        cname, la, lo = CAPITALS[ab]
        m = np.flatnonzero(U["unit_state"] == fips_of[ab])
        d = (U["lat"][m] - la) ** 2 + ((U["lon"][m] - lo) * np.cos(np.radians(la))) ** 2
        caps.append(int(m[np.argmin(d)])); cap_names[s] = (ab, cname)
    caps = np.array(caps)
    for v in a.variants:
        kind, obj = v.split(".")
        t0 = time.perf_counter()
        print(f"{v}:")
        prev = OUT / f"{v}.json"
        if a.resume and prev.exists():
            old = json.loads(prev.read_text(encoding="utf-8"))
            pos = {key: i for i, key in enumerate(U["keys"].tolist())}
            lab = np.full(n, -1, np.int64)
            for s_ in old["states"]:
                lab[[pos[x] for x in s_["units"].split(",")]] = s_["id"]
            ch = Chain(U, lab, k, a.tol, obj, capitals=caps if kind == "capitals" else None,
                       seed=a.seed + 11 + 1000 * old.get("resumed", 0) + 1)
            print(f"  resumed from {old['ties_kept_pct'][obj]}% ({old['params']['bursts']} bursts before)")
        elif kind == "free":
            ch = Chain(U, free0, k, a.tol, obj, seed=a.seed + 11)
        else:
            ch = Chain(U, capital_voronoi(U, caps, k, iters=3000), k, a.tol, obj, capitals=caps, seed=a.seed + 13)
            balance_pairs(ch)
        start = ch.score()
        best = short_bursts(ch, a.bursts, a.burst)
        dv = ch.devs()
        assert np.abs(dv).max() <= a.tol + 1e-9
        info = {"title": ("50 states drawn by social connectedness" if kind == "free"
                          else "Today's 50 capitals, borders drawn by social connectedness")
                + f" ({'distance-adjusted' if obj == 'adjusted' else 'raw'} SCI, equal population)",
                "kind": kind, "objective": obj, "capitals_fixed": kind == "capitals",
                "resumed": (old.get("resumed", 0) + 1) if a.resume and prev.exists() else 0,
                "params": {"split": a.split, "tol": a.tol, "bursts": a.bursts + (old["params"]["bursts"] if a.resume and prev.exists() else 0),
                           "burst": a.burst, "seed": a.seed,
                           "units": n, "water_links": U["water_links"]},
                "start_score": round(start, 3), "neutral_quantiles_5_50_95": nq,
                "max_dev_pct": round(100 * float(np.abs(dv).max()), 3),
                "seconds": round(time.perf_counter() - t0, 1)}
        if kind == "capitals":
            info["capital_names"] = {s: (f"{CAPITALS[ab][0]} ({ab})", CAPITALS[ab][0]) for s, (ab, _) in cap_names.items()}
        res = write(U, ch.labels, v, info, current)
        print(f"  {v}: {res['ties_kept_pct']} (start {start:.2f}%), {info['seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
