"""Optional block-level refinement: turn a block-group plan into a census-block plan equal to within one person.

In short (for people): congressional districts must be "as nearly equal as practicable", which in practice
means within a person or two. Block groups (about 1,300 people each) cannot get closer than about 0.1%. This
tool takes a block-group plan, hands each of its census blocks (about 40 people each) to the same district, and
then moves a few hundred blocks along district borders until every district has the same population to within
one person, keeping every district in one piece. ``python -m redistricting_addons.blocks --state RI`` refines the
solver's plan for Rhode Island and writes a report and a block-equivalency file to ``site/data/blocks/``.

For AI agents (detailed):
- Data: TIGER/Line 2020 ``tl_2020_<FIPS>_tabblock20`` (one download per state, cached in
  ``redistricting_addons.cenpop.cache_dir()/blocks``; sha256 recorded in the report). Its dBase table carries
  ``POP20`` (2020 PL 94-171 population, with the Census Bureau's disclosure-avoidance noise; Cohen, Duchin et al.
  2022 found that noise does not stop equal-population districting). No Census API key is needed.
- Adjacency: rook, by exact matching of boundary segments (TIGER/Line shares vertices along common edges),
  keyed by a 64-bit hash of the quantised segment to keep memory down on large states. Land is not separated
  from water: water blocks (population 0) are ordinary nodes, so contiguity may run through water blocks, as
  enacted block-equivalency files allow. Components of the block graph (offshore islands) are joined to their
  nearest block (by internal point) with recorded ``water_links``.
- Steps: (1) each block takes its block group's district (block GEOID[:12] = block-group GEOID; the site's unit
  order comes from ``cenpop.geoids``); (2) contiguity repair: every district keeps its most populous piece and
  smaller pieces join the neighbouring district they share most edges with; (3) balancing: integer targets
  (``total // k``, with ``total % k`` districts one higher, given to the districts currently largest), then, while
  some district is off target, the district with the largest surplus sends population along the shortest path
  of adjacent districts to the nearest district with a deficit, one hop at a time, moving border blocks largest
  first without going past the amount and never disconnecting the giving district.
- Output: ``site/data/blocks/<ST>.<plan>.json`` (report: deviation before and after, pieces repaired, blocks and
  people moved, cut block edges, timings) and ``site/data/blocks/<ST>.<plan>.bef.csv.gz`` (block GEOID, district
  1..k). Not touched by the sync script. Only the pilot states are committed; any state can be built.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import time
import zipfile
from collections import deque
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from . import cenpop
from .graphs import read_dbf, read_polygons

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "site" / "data" / "blocks"
URL = "https://www2.census.gov/geo/tiger/TIGER2020/TABBLOCK20/tl_2020_{fips}_tabblock20.zip"
Q = 1e7


# --------------------------------------------------------------------------- data

def block_files(fips: str) -> tuple[Path, Path, Path, str]:
    d = cenpop.cache_dir() / "blocks"
    z = cenpop.fetch(URL.format(fips=fips), d / f"tl_2020_{fips}_tabblock20.zip", None, timeout=1800)
    sha = hashlib.sha256(z.read_bytes()).hexdigest()
    stem = d / f"t{fips}" / f"tl_2020_{fips}_tabblock20"
    if not all(stem.with_suffix(s).exists() for s in (".shp", ".shx", ".dbf")):
        with zipfile.ZipFile(z) as zf:
            for name in zf.namelist():
                if name.endswith((".shp", ".shx", ".dbf")):
                    zf.extract(name, stem.parent)
    return stem.with_suffix(".shp"), stem.with_suffix(".shx"), stem.with_suffix(".dbf"), sha


def _mix(x):
    """splitmix64 finaliser on uint64 arrays."""
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def block_adjacency(polys: dict[int, list[np.ndarray]], n: int) -> np.ndarray:
    """Unique (i, j), i < j, pairs of blocks sharing a boundary segment (hashed segment keys)."""
    keys, owners = [], []
    with np.errstate(over="ignore"):
        for key, rings in polys.items():
            for r in rings:
                q = np.round(r * Q).astype(np.int64).view(np.uint64)
                a, b = q[:-1], q[1:]
                ha = _mix(a[:, 0] * np.uint64(0x9E3779B97F4A7C15) + a[:, 1])
                hb = _mix(b[:, 0] * np.uint64(0x9E3779B97F4A7C15) + b[:, 1])
                lo, hi = np.minimum(ha, hb), np.maximum(ha, hb)
                ok = ha != hb
                keys.append(_mix(lo[ok] * np.uint64(0xC2B2AE3D27D4EB4F) + hi[ok]))
                owners.append(np.full(int(ok.sum()), key, np.int32))
    k = np.concatenate(keys)
    o = np.concatenate(owners)
    del keys, owners
    order = np.lexsort((o, k))
    k, o = k[order], o[order]
    same = (k[1:] == k[:-1]) & (o[1:] != o[:-1])
    i = np.flatnonzero(same)
    pairs = np.column_stack([np.minimum(o[i], o[i + 1]), np.maximum(o[i], o[i + 1])]).astype(np.int64)
    return np.unique(pairs, axis=0) if len(pairs) else np.zeros((0, 2), np.int64)


def join_components(n: int, edges: np.ndarray, xy: np.ndarray):
    links = []
    e = edges
    while True:
        a = sparse.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
        nc, comp = connected_components(a, directed=False)
        if nc == 1:
            return e, np.array(links, np.int64).reshape(-1, 2)
        small = int(np.argmin(np.bincount(comp)))
        inside, outside = np.flatnonzero(comp == small), np.flatnonzero(comp != small)
        d, j = cKDTree(xy[outside]).query(xy[inside])
        t = int(np.argmin(d))
        u, v = sorted((int(inside[t]), int(outside[j[t]])))
        links.append((u, v))
        e = np.vstack([e, [[u, v]]])


# --------------------------------------------------------------------------- refinement

class Plan:
    def __init__(self, labels, pop, edges, k):
        self.lab = np.asarray(labels, np.int64).copy()
        self.pop = np.asarray(pop, np.int64)
        self.k = k
        n = len(self.lab)
        a = sparse.coo_matrix((np.ones(2 * len(edges)), (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
                              shape=(n, n)).tocsr()
        self.ptr, self.idx = a.indptr, a.indices
        self.edges = edges
        self.pops = np.bincount(self.lab, weights=self.pop, minlength=k).astype(np.int64)

    def nbrs(self, u):
        return self.idx[self.ptr[u]:self.ptr[u + 1]]

    def cut(self):
        return int(np.sum(self.lab[self.edges[:, 0]] != self.lab[self.edges[:, 1]]))

    def pieces(self):
        e = self.edges[self.lab[self.edges[:, 0]] == self.lab[self.edges[:, 1]]]
        n = len(self.lab)
        a = sparse.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
        return connected_components(a, directed=False)

    def repair(self, rounds=10000):
        """Give each district one piece: smaller pieces join the neighbour sharing most edges with them."""
        moved_blocks = moved_people = 0
        for _ in range(rounds):
            nc, comp = self.pieces()
            if nc == self.k:
                break
            cpop = np.bincount(comp, weights=self.pop + 1e-6)
            cd = np.zeros(nc, np.int64); cd[comp] = self.lab
            main = {}
            for c in range(nc):
                d = int(cd[c])
                if d not in main or cpop[c] > cpop[main[d]]:
                    main[d] = c
            orphans = [c for c in range(nc) if main.get(int(cd[c])) != c]
            c = min(orphans, key=lambda x: cpop[x])
            members = np.flatnonzero(comp == c)
            own = int(cd[c])
            votes = {}
            for u in members:
                for v in self.nbrs(u):
                    if comp[v] != c and self.lab[v] != own:
                        votes[int(self.lab[v])] = votes.get(int(self.lab[v]), 0) + 1
            if not votes:
                break
            to = max(votes, key=votes.get)
            people = int(self.pop[members].sum())
            self.lab[members] = to
            self.pops[own] -= people; self.pops[to] += people
            moved_blocks += len(members); moved_people += people
        return moved_blocks, moved_people

    def connected_without(self, u):
        d = self.lab[u]
        nb = [v for v in self.nbrs(u) if self.lab[v] == d]
        if len(nb) <= 1:
            return True
        goal, seen, dq = set(nb[1:]), {u, nb[0]}, deque([nb[0]])
        while dq and goal:
            a = dq.popleft()
            for v in self.nbrs(a):
                if v not in seen and self.lab[v] == d:
                    seen.add(v); goal.discard(v); dq.append(v)
        return not goal

    def district_graph(self):
        a = self.lab[self.edges[:, 0]]; b = self.lab[self.edges[:, 1]]
        m = a != b
        g = {d: set() for d in range(self.k)}
        for x, y in zip(a[m].tolist(), b[m].tolist()):
            g[x].add(y); g[y].add(x)
        return g

    def transfer(self, a, b, amount):
        """Move border blocks from district a to adjacent district b, largest first, up to ``amount`` people."""
        moved_b = moved_p = 0
        border = set()
        for x, y in self.edges[(self.lab[self.edges[:, 0]] != self.lab[self.edges[:, 1]])].tolist():
            if self.lab[x] == a and self.lab[y] == b: border.add(x)
            elif self.lab[y] == a and self.lab[x] == b: border.add(y)
        remaining = amount
        while remaining > 0:
            cand = sorted((u for u in border if self.lab[u] == a and 0 < self.pop[u] <= remaining),
                          key=lambda u: -self.pop[u])
            done = False
            for u in cand:
                if self.connected_without(u):
                    self.lab[u] = b
                    p = int(self.pop[u])
                    self.pops[a] -= p; self.pops[b] += p
                    remaining -= p; moved_b += 1; moved_p += p
                    border.discard(u)
                    border.update(int(v) for v in self.nbrs(u) if self.lab[v] == a)
                    done = True
                    break
            if not done:
                break
        return moved_b, moved_p, amount - remaining

    def border(self, a, b):
        out = []
        for x, y in self.edges[(self.lab[self.edges[:, 0]] != self.lab[self.edges[:, 1]])].tolist():
            if self.lab[x] == a and self.lab[y] == b: out.append(x)
            elif self.lab[y] == a and self.lab[x] == b: out.append(y)
        return sorted(set(out))

    def swap(self, a, b, top):
        """Net-move between 1 and ``top`` people from a to b (largest possible first) by swapping one border
        block each way; returns (blocks moved, people moved, net amount)."""
        A, B = self.border(a, b), self.border(b, a)
        by_pop = {}
        for v in B:
            by_pop.setdefault(int(self.pop[v]), []).append(v)
        ok_u, ok_v = {}, {}
        for need in range(top, 0, -1):
            for u in A:
                pu = int(self.pop[u])
                vs = by_pop.get(pu - need)
                if not vs:
                    continue
                if u not in ok_u:
                    ok_u[u] = self.connected_without(u)
                if not ok_u[u]:
                    continue
                self.lab[u] = b
                for v in vs:
                    if any(self.lab[w] == a for w in self.nbrs(v)) and self.connected_without(v):
                        self.lab[v] = a
                        pv = int(self.pop[v])
                        self.pops[a] += pv - pu; self.pops[b] += pu - pv
                        return 2, pu + pv, need
                self.lab[u] = a
        return 0, 0, 0

    def balance(self, max_iter=20000, log=None, every=200):
        """Diffuse population toward integer targets, one hop at a time, until no district can move anything.
        ``log`` (optional) gets a progress line every ``every`` rounds."""
        total, k = int(self.pop.sum()), self.k
        q, r = divmod(total, k)
        moved_b = moved_p = 0
        for it in range(max_iter):
            if log and it % every == 0:
                log(f"  balance round {it}: largest - smallest = {int(self.pops.max() - self.pops.min()):,} people, "
                    f"{moved_b:,} units moved")
            if self.pops.max() - self.pops.min() <= 1:
                break
            # integer targets: q each, plus one for the r districts that are currently largest
            target = np.full(k, q, np.int64)
            target[np.argsort(-self.pops, kind="stable")[:r]] += 1
            surplus = self.pops - target
            g = self.district_graph()
            # distance (in districts) from every district to the nearest one with a deficit
            dist = {int(d): 0 for d in np.flatnonzero(surplus < 0)}
            dq = deque(dist)
            while dq:
                x = dq.popleft()
                for y in g[x]:
                    if y not in dist:
                        dist[y] = dist[x] + 1; dq.append(y)
            moved = False
            for a in [int(x) for x in np.argsort(-surplus, kind="stable") if surplus[x] > 0]:
                steps = sorted((y for y in g[a] if y in dist and dist[y] < dist.get(a, 1 << 30)), key=lambda y: (dist[y], surplus[y]))
                for y in steps:  # first hop toward the nearest deficit; the next step carries it on
                    amount = int(surplus[a]) if surplus[y] >= 0 else int(min(surplus[a], -surplus[y]))
                    mb, mp, sent = self.transfer(a, y, amount)
                    if sent > 0:
                        moved_b += mb; moved_p += mp; moved = True
                        break
                if moved:
                    break
            if not moved:  # fine tuning: an exact net amount by swapping one border block each way, toward a deficit
                for a in [int(x) for x in np.argsort(-surplus, kind="stable") if surplus[x] > 0]:
                    for y in sorted((y for y in g[a] if y in dist and dist[y] < dist.get(a, 1 << 30)),
                                    key=lambda y: (dist[y], surplus[y])):
                        top = int(surplus[a]) if surplus[y] >= 0 else int(min(surplus[a], -surplus[y]))
                        mb, mp, _ = self.swap(a, y, top)
                        if mb:
                            moved_b += mb; moved_p += mp; moved = True
                            break
                    if moved:
                        break
            if not moved:
                break
        return moved_b, moved_p


# --------------------------------------------------------------------------- run

def refine_state(st: str, plan: str = "solver") -> dict:
    t0 = time.perf_counter()
    site = json.loads((ROOT / "site" / "data" / f"{st}.json").read_text(encoding="utf-8"))
    fips, k = site["fips"], int(site["seats"])
    bg_labels = np.asarray(site["plan"] if plan == "solver" else site["current"], np.int64)
    geo_index = {g: i for i, g in enumerate(cenpop.geoids(fips))}
    shp, shx, dbf_path, sha = block_files(fips)
    t_dl = time.perf_counter()
    dbf = read_dbf(dbf_path, ["GEOID20", "POP20", "INTPTLAT20", "INTPTLON20"])
    n = len(dbf["GEOID20"])
    pop = dbf["POP20"].astype(np.int64)
    bg = np.array([geo_index[g[:12]] for g in dbf["GEOID20"]])
    polys = read_polygons(shp, shx, range(n))
    edges = block_adjacency(polys, n)
    del polys
    lat = dbf["INTPTLAT20"].astype(float); lon = dbf["INTPTLON20"].astype(float)
    xy = np.column_stack([111.195 * lon * np.cos(np.radians(lat.mean())), 111.195 * lat])
    edges, links = join_components(n, edges, xy)
    t_graph = time.perf_counter()
    p = Plan(bg_labels[bg], pop, edges, k)
    ideal = pop.sum() / k
    dev = lambda: {"max_abs_persons": int(np.max(np.abs(p.pops - ideal)) + 0.5), "range_persons": int(p.pops.max() - p.pops.min()),
                   "max_abs_pct": round(float(100 * np.max(np.abs(p.pops / ideal - 1))), 4)}
    before = {**dev(), "pieces": int(p.pieces()[0]), "cut_block_edges": p.cut()}
    rb, rp = p.repair()
    after_repair = {**dev(), "pieces": int(p.pieces()[0])}
    bb, bp = p.balance()
    after = {**dev(), "pieces": int(p.pieces()[0]), "cut_block_edges": p.cut()}
    changed = p.lab != bg_labels[bg]
    OUT.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["GEOID20", "district"])
    for g, d in zip(dbf["GEOID20"], p.lab + 1):
        w.writerow([g, int(d)])
    bef = OUT / f"{st}.{plan}.bef.csv.gz"
    with gzip.GzipFile(bef, "wb", mtime=0) as fh:
        fh.write(buf.getvalue().encode("ascii"))
    report = {
        "v": 1, "abbr": st, "plan": plan, "districts": k, "blocks": int(n), "people": int(pop.sum()),
        "ideal": round(float(ideal), 2),
        "source": f"TIGER/Line 2020 tl_2020_{fips}_tabblock20 (sha256 {sha}); POP20; rook adjacency; "
                  f"{len(links)} water links join islands to their nearest block",
        "block_edges": int(len(edges)), "water_links": int(len(links)),
        "before": before, "after_repair": after_repair, "after": after,
        "repair": {"blocks_moved": int(rb), "people_moved": int(rp)},
        "balance": {"blocks_moved": int(bb), "people_moved": int(bp)},
        "blocks_changed_from_block_group_plan": int(changed.sum()),
        "people_changed_from_block_group_plan": int(pop[changed].sum()),
        "district_populations": p.pops.tolist(),
        "seconds": {"download_and_read": round(t_dl - t0, 1), "graph": round(t_graph - t_dl, 1),
                    "refine": round(time.perf_counter() - t_graph, 1)},
        "bef": bef.name,
    }
    (OUT / f"{st}.{plan}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m redistricting_addons.blocks", description=__doc__.split("\n\n")[0])
    ap.add_argument("--state", nargs="+", required=True, help="two-letter codes")
    ap.add_argument("--plan", choices=["solver", "official"], default="solver",
                    help="block-group plan to refine: the solver's (default) or the official map")
    a = ap.parse_args(argv)
    for st in a.state:
        r = refine_state(st.upper(), a.plan)
        print(f"{r['abbr']} {r['plan']}: {r['blocks']} blocks; before {r['before']['range_persons']} people apart "
              f"({r['before']['pieces']} pieces for {r['districts']} districts), after {r['after']['range_persons']} "
              f"({r['after']['pieces']} pieces); {r['blocks_changed_from_block_group_plan']} blocks / "
              f"{r['people_changed_from_block_group_plan']} people changed; {r['seconds']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
