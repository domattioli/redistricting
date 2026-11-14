"""Stronger compactness for the SCI states, and states grown from today's capitals to today's populations.

In short (for people): three additions to the 50-state SCI maps (``sci_states``). (1) **No corner touching**:
two places count as neighbours only when they share a real stretch of border, not a point or a sliver, so no
state can hang together by a corner. (2) **Stronger compactness**: borders are measured in kilometres (a thin neck
of a state costs a lot of border), the total may be at most 85% of today's state lines, and the least cohesive
state counts extra, so the worst state cannot be sacrificed for the average. (3) **Grown from the capitals**:
every capital keeps its state's 2020 population (Idaho about 1.84 million, California 39.5 million) and gathers
the people most connected to it on Facebook, spreading only to neighbouring places, until it has that many;
then, optionally, the borders are smoothed for friendship compactness. ``python -m redistricting_addons.sci_grow``
writes ``site/data/sci_states/<variant>.json`` and ``docs/sci_states/<variant>.png`` for ``free.strong``,
``grown`` and ``grown.strong``.

For AI agents (detailed):
- Additive: nothing in ``sci_states`` is changed; this module reuses its units (``build_units``), scorer,
  ``Chain`` (subclassed as ``StrongChain``), ``voronoi_split``, ``short_bursts``, ``write`` and ``report``.
- Border length per unit edge (``unit_borders``, cached as ``sci/unit_border_km_<split>.npy``): the sum of the
  shared block-group border km behind it (``site/data/graphs/<ST>.json`` ``border_km`` within states,
  ``graphs.shared_borders`` across state lines and inside DC); water links have 0.
- No corner touching (``strong_edges``): an edge is kept when its shared border is at least ``--frac`` (default
  5%) of the shorter land perimeter of its two units (perimeter = sum of the unit's shared borders); water links
  are kept; if dropping weak edges cuts the graph, the longest dropped edges joining the pieces are put back.
  Block-group adjacency is already rook (a shared segment of positive length); this also removes slivers.
- ``StrongChain``: optional per-state population ``targets`` (tolerance is a fraction of each state's own
  target), score = mean state cohesion (raw SCI, own-county pairs included) + ``floor`` x the least cohesive
  state's cohesion, minus 100 / cap points per km of border over ``max_border_km`` (border = shared km of every
  unit edge between two states, on the full edge list).
- ``grow``: one priority queue for all 50 states; a unit next to state s has priority log SCI(unit's county,
  capital's county), ties by distance to the capital; a state takes the best unit on its frontier while that keeps
  it within (1 + tol) of its target. Units no state could take are joined to the bordering state whose capital
  they are most tied to; ``balance_targets`` then re-splits neighbouring pairs between their two capitals
  (``voronoi_split`` at the cut nearest the pair's target share) until each state is within tol of its target.
  Every state stays connected throughout (growth and joins only along edges; Voronoi splits keep both sides
  connected). Targets: 2020 population of each state, DC added to Maryland (DC has no capital here).
- ``grown.strong``: short bursts of capital-respecting moves from ``grown`` with the ``StrongChain`` score,
  ``max_border_km`` = today's state lines' km (same populations as today, so today's border is the yardstick).
  ``free.strong``: equal populations, start ``ensemble.tree_seed`` on the strong graph, cap 0.85 x today's km.
"""
from __future__ import annotations

import argparse
import heapq
import json
import time

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from . import cenpop, graphs
from . import sci_states as S
from .ensemble import Graph, _region
from .ensemble import tree_seed


def unit_borders(U, split: int = 100_000) -> np.ndarray:
    """Shared border km per row of ``U['edges']`` (0 for water links). Cached."""
    cache = cenpop.cache_dir() / "sci" / f"unit_border_km_{split}.npy"
    if cache.exists():
        km = np.load(cache)
        if len(km) == len(U["edges"]):
            return km
    t0 = time.perf_counter()
    rows = [r for r in cenpop.rows() if r[0] != "72"]
    geo = np.array([r[0] + r[1] + r[2] + r[3] for r in rows])
    st = np.array([r[0] for r in rows])
    unit = U["bg_unit"]
    gi = {g: i for i, g in enumerate(geo.tolist())}
    files = S._state_files()
    pe, pk = [], []
    for fips, d in files.items():
        g = json.loads((S.ROOT / "site" / "data" / "graphs" / f"{d['abbr']}.json").read_text(encoding="utf-8"))
        off = gi[geo[st == fips][0]]
        pe.append(np.asarray(g["edges"], np.int64).reshape(-1, 2) + off)
        pk.append(np.asarray(g["border_km"], float))
    shp, shx, dbf_p = graphs.boundary_files()
    dbf = graphs.read_dbf(dbf_p, ["GEOID"])
    rec_bg = {int(r): gi.get(g) for r, g in enumerate(dbf["GEOID"].tolist()) if g in gi}
    polys = graphs.read_polygons(shp, shx, list(rec_bg))
    by_bg = {v: polys[r] for r, v in rec_bg.items()}
    box = {i: (np.vstack(p)[:, 0].min(), np.vstack(p)[:, 1].min(), np.vstack(p)[:, 0].max(), np.vstack(p)[:, 1].max())
           for i, p in by_bg.items() if p}
    bgs = np.array(sorted(box)); bx = np.array([box[i] for i in bgs])
    sbox = {}
    for s_ in np.unique(st[bgs]):
        m = st[bgs] == s_
        sbox[s_] = (bx[m, 0].min(), bx[m, 1].min(), bx[m, 2].max(), bx[m, 3].max())
    Sl = sorted(sbox)
    eps = 0.01
    for i, a in enumerate(Sl):  # the same candidate search as sci_states.build_units
        for b in Sl[i + 1:]:
            A, B = sbox[a], sbox[b]
            x0, y0, x1, y1 = max(A[0], B[0]) - eps, max(A[1], B[1]) - eps, min(A[2], B[2]) + eps, min(A[3], B[3]) + eps
            if x0 > x1 or y0 > y1:
                continue
            m = np.isin(st[bgs], [a, b]) & (bx[:, 2] >= x0) & (bx[:, 0] <= x1) & (bx[:, 3] >= y0) & (bx[:, 1] <= y1)
            cand = bgs[m]
            if cand.size < 2:
                continue
            pairs, km = graphs.shared_borders({int(c): by_bg[int(c)] for c in cand})
            if len(pairs):
                x = st[pairs[:, 0]] != st[pairs[:, 1]]
                pe.append(pairs[x]); pk.append(km[x])
    dc = [int(c) for c in np.flatnonzero(st == "11") if int(c) in by_bg]
    pairs, km = graphs.shared_borders({c: by_bg[c] for c in dc})
    pe.append(pairs); pk.append(km)
    e, k_ = np.vstack(pe), np.concatenate(pk)
    ue = np.sort(np.column_stack([unit[e[:, 0]], unit[e[:, 1]]]), axis=1)
    keep = ue[:, 0] != ue[:, 1]
    n = len(U["pop"])
    code = ue[keep, 0].astype(np.int64) * n + ue[keep, 1]
    E = U["edges"]
    ecode = E[:, 0].astype(np.int64) * n + E[:, 1]
    o = np.argsort(ecode)
    pos = np.searchsorted(ecode[o], code)
    ok = (pos < len(o)) & (ecode[o][np.minimum(pos, len(o) - 1)] == code)
    out = np.bincount(o[pos[ok]], weights=k_[keep][ok], minlength=len(E))
    np.save(cache, out)
    print(f"unit border lengths: {np.count_nonzero(out):,} of {len(E):,} edges with land border "
          f"({time.perf_counter() - t0:.0f}s)")
    return out


def strong_edges(U, km, frac: float = 0.05) -> np.ndarray:
    """Mask of edges kept: shared border >= frac x the shorter perimeter of the two units, water links kept,
    weak edges put back (longest first) only where needed to keep the graph connected."""
    E, n = U["edges"], len(U["pop"])
    per = np.bincount(E[:, 0], weights=km, minlength=n) + np.bincount(E[:, 1], weights=km, minlength=n)
    keep = (km == 0) | (km >= frac * np.minimum(per[E[:, 0]], per[E[:, 1]]))
    while True:
        e = E[keep]
        ncomp, comp = connected_components(
            sparse.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)), directed=False)
        if ncomp == 1:
            return keep
        join = np.flatnonzero(~keep & (comp[E[:, 0]] != comp[E[:, 1]]))
        # put back the longest joining edge of each piece pair
        seen = set()
        for x in join[np.argsort(-km[join])]:
            pr = (min(comp[E[x, 0]], comp[E[x, 1]]), max(comp[E[x, 0]], comp[E[x, 1]]))
            if pr not in seen:
                seen.add(pr); keep[x] = True


def subgraph(U, keep) -> dict:
    """``U`` with only the kept edges (the Chain and its moves see only real borders)."""
    return {**U, "edges": U["edges"][keep]}


class StrongChain(S.Chain):
    """``sci_states.Chain`` plus per-state targets, a border-km cap and a cohesion floor (see module docstring)."""

    def __init__(self, U, labels, k, tol, objective, *, targets=None, full_edges=None, border_km=None,
                 max_border_km=None, floor=0.0, **kw):
        super().__init__(U, labels, k, tol, objective, cohesion=True, **kw)
        self.T = np.full(k, self.ideal) if targets is None else np.asarray(targets, float)
        self.fe, self.fkm = full_edges, border_km
        self.max_km, self.floor = max_border_km, floor

    def border(self):
        L = self.labels
        return float(self.fkm[L[self.fe[:, 0]] != L[self.fe[:, 1]]].sum())

    def cohesions(self):
        return self.parts / np.maximum(self.den, 1e-300)

    def score(self):
        c = self.cohesions()
        s = 100 * float(c.mean()) + self.floor * 100 * float(c.min())
        if self.max_km is not None:
            s -= 100 * max(0.0, self.border() - self.max_km) / self.max_km
        if self.max_cut is not None:
            s -= 100 * max(0, self.cut_edges() - self.max_cut) / self.max_cut
        return s

    def devs(self):
        return np.bincount(self.labels, weights=self.g.pop, minlength=self.k) / self.T - 1

    def move(self, a=None, b=None, targets=None, sci_trees=True):
        L, e = self.labels, self.g.edges
        if a is None:
            cut = np.flatnonzero(L[e[:, 0]] != L[e[:, 1]])
            x = cut[self.rng.integers(cut.size)]
            a, b = int(L[e[x, 0]]), int(L[e[x, 1]])
        ta, tb = (self.T[a], self.T[b]) if targets is None else targets
        da, db = self.tol * ta, self.tol * tb
        mask = (L == a) | (L == b)
        nodes, eu, ev, em = _region(self.g, mask)
        p = self.g.pop[nodes]
        w = self.wmax_all[em] if sci_trees else np.ones(int(em.sum()))
        if self.caps is not None:
            loc = {int(u): i for i, u in enumerate(nodes)}
            ca, cb = loc[int(self.caps[a])], loc[int(self.caps[b])]
            length = self.km[em] * self.rng.uniform(0.85, 1.15, int(em.sum())) / w
            side = S.voronoi_split(nodes.size, eu, ev, length, p, ca, cb, ta - da, ta + da, tb - db, tb + db, self.rng)
            if side is None:
                return False
            sa = side if side[ca] else ~side
        else:
            side = S._split(nodes, eu, ev, w, p, ta - da, ta + da, tb - db, tb + db, self.rng)
            if side is None:
                return False
            ps = p[side].sum()
            sa = side if abs(ps - ta) <= da else ~side
        L[nodes[sa]] = a
        L[nodes[~sa]] = b
        for s_ in (a, b):
            self.update(s_)
        return True


def grow(U, caps, targets, tol=0.01, objective="raw") -> np.ndarray:
    """Capitals gather the units most tied to them (SCI to the capital's county), spreading only along edges, until
    each reaches its target; leftovers join the bordering state whose capital they are most tied to."""
    E, n, pop, r = U["edges"], len(U["pop"]), U["pop"], U["region"]
    A = sparse.csr_matrix((np.ones(2 * len(E)), (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])), shape=(n, n))
    ind, ptr = A.indices, A.indptr
    M = U[f"M_{objective}"]
    k = len(caps)
    cr = r[caps]
    la, lo = np.radians(U["lat"]), np.radians(U["lon"])

    def km(u, s):
        c = caps[s]
        return 6371.0 * np.hypot((la[u] - la[c]), (lo[u] - lo[c]) * np.cos(0.5 * (la[u] + la[c])))

    def pri(u, s):
        return -np.log(max(M[r[u], cr[s]], 1e-30))

    labels = np.full(n, -1, np.int64)
    P = np.zeros(k)
    heap = []
    for s, u in enumerate(caps):
        labels[u] = s; P[s] += pop[u]
    for s, u in enumerate(caps):
        for v in ind[ptr[u]:ptr[u + 1]]:
            if labels[v] < 0:
                heapq.heappush(heap, (pri(v, s), km(v, s), int(v), s))
    hi = np.asarray(targets) * (1 + tol)
    while heap:
        _, _, v, s = heapq.heappop(heap)
        if labels[v] >= 0 or P[s] + pop[v] > hi[s]:
            continue
        labels[v] = s; P[s] += pop[v]
        for w in ind[ptr[v]:ptr[v + 1]]:
            if labels[w] < 0:
                heapq.heappush(heap, (pri(w, s), km(w, s), int(w), s))
    left = int((labels < 0).sum())
    while (labels < 0).any():
        und = np.flatnonzero(labels < 0)
        changed = False
        for v in und:
            nb = ind[ptr[v]:ptr[v + 1]]
            st = np.unique(labels[nb][labels[nb] >= 0])
            if st.size:
                labels[v] = int(st[np.argmin([pri(v, s) for s in st])]); changed = True
        if not changed:
            raise RuntimeError("units unreachable from any capital")
    print(f"  grown: {left:,} units joined afterwards; largest gap "
          f"{100 * np.abs(np.bincount(labels, weights=pop, minlength=k) / targets - 1).max():.1f}%")
    return labels


def balance_targets(ch: StrongChain, log=print, limit=200_000):
    """Re-split neighbouring pairs between their two capitals at the cut nearest the pair's target share until
    every state is within tol of its own target (kept only when the pair's squared relative gaps shrink)."""
    e, T = ch.g.edges, ch.T
    stuck = 0
    for it in range(limit):
        dv = ch.devs()
        if np.abs(dv).max() <= ch.tol:
            log(f"  balanced after {it} rounds; max gap {100 * np.abs(dv).max():.2f}%")
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
            want = p.sum() * T[a] / (T[a] + T[b])
            side = S.voronoi_split(nodes.size, eu, ev, length, p, loc[int(ch.caps[a])], loc[int(ch.caps[b])],
                                   0, np.inf, 0, np.inf, ch.rng, target=want)
            if side is None:
                continue
            new = (p[side].sum() / T[a] - 1) ** 2 + (p[~side].sum() / T[b] - 1) ** 2
            if new >= dv[a] ** 2 + dv[b] ** 2 - 1e-9:
                continue
            L[nodes[side]] = a
            L[nodes[~side]] = b
            for s_ in (a, b):
                ch.update(s_)
            moved = True
            break
        if not moved:
            stuck += 1
            old = ch.tol; ch.tol = 0.5
            ch.move()
            ch.tol = old
        if it % 500 == 0:
            log(f"  balancing: round {it}, max gap {100 * np.abs(dv).max():.1f}%, {stuck} shakes")
    raise RuntimeError("balancing did not converge")


def _finish(U, ch, name, info, current, km, today_km, targets=None):
    """Write through ``sci_states.write`` and add border km, the least cohesive state and target gaps."""
    res = S.write(U, ch.labels, name, info, current)
    L = ch.labels
    res["border_km"] = round(float(km[L[U["edges"][:, 0]] != L[U["edges"][:, 1]]].sum()), 1)
    res["today_border_km"] = round(today_km, 1)
    if targets is not None:
        P = np.bincount(L, weights=U["pop"], minlength=len(targets))
        for s_ in res["states"]:
            s_["target"] = int(targets[s_["id"]])
            s_["dev_pct"] = round(100 * (P[s_["id"]] / targets[s_["id"]] - 1), 3)
        res["max_dev_pct"] = round(100 * float(np.abs(P / targets - 1).max()), 3)
    (S.OUT / f"{name}.json").write_text(json.dumps(res, separators=(",", ":")), encoding="utf-8")
    S.draw(U, L, S.DOCS / f"{name}.png", info["title"], res)
    return res


def report() -> None:
    """docs/SCI_GROW.md from the saved files (with free.compact and today's states for comparison)."""
    names = {"free.compact": "Equal states, friendship-compact (earlier)", "free.strong": "Equal states, strongly compact",
             "grown": "Grown from capitals", "grown.strong": "Grown from capitals, then compact"}
    res = {v: json.loads((S.OUT / f"{v}.json").read_text(encoding="utf-8")) for v in names if (S.OUT / f"{v}.json").exists()}
    if not res:
        return
    U = S.build_units()
    km = unit_borders(U)
    E = U["edges"]
    pos = {key: i for i, key in enumerate(U["keys"].tolist())}
    for v, r in res.items():
        if "border_km" not in r:
            lab = np.full(len(U["pop"]), -1, np.int64)
            for s_ in r["states"]:
                lab[[pos[x] for x in s_["units"].split(",")]] = s_["id"]
            r["border_km"] = round(float(km[lab[E[:, 0]] != lab[E[:, 1]]].sum()), 1)
    tk = next((r["today_border_km"] for r in res.values() if "today_border_km" in r), float("nan"))
    today = next(iter(res.values()))["today"]
    lines = ["# SCI states: stronger compactness, and states grown from today's capitals", "", "## In short (for people)", "",
             "- **No corner touching.** Two places count as neighbours only when they share at least 5% of the shorter "
             "one's border, so no state can hang together by a corner or a sliver.",
             "- **Stronger compactness.** Border is measured in kilometres (a thin neck costs a lot of it), and the least "
             "cohesive state counts extra, so the worst state is not sacrificed for the average.",
             "- **Grown from the capitals.** Each of today's capitals keeps its state's 2020 population (DC counted with "
             "Maryland) and gathers the people most connected to it on Facebook, spreading only into neighbouring "
             "places, until it has that many (Boise gathers about 1.84 million). Then, optionally, the borders are "
             "smoothed for friendship compactness without moving any capital.",
             "- *Cohesion* of a state = the share of its people's friendships that stay inside it. The index follows "
             "migration, party and race, so these maps are a lens, not a neutral proposal.", "",
             "| Version | Mean cohesion | Least cohesive state | Border (km) | Ties kept, raw | Map |",
             "| --- | --: | --: | --: | --: | --- |"]
    for v, r in res.items():
        c = r["cohesion_pct"]
        lines.append(f"| {names[v]} | {c['mean']:.1f}% | {c['min']:.1f}% | {r['border_km']:,.0f} | "
                     f"{r['ties_kept_pct']['raw']:.1f}% | [{v}.png](sci_states/{v}.png) |")
    lines += [f"| Today's states | 56.6% | | {tk:,.0f} | {today['raw']:.1f}% | |", ""]
    for v in res:
        if v != "free.compact":
            lines += [f"![{names[v]}](sci_states/{v}.png)", ""]
    lines += ["---", "", "## For AI agents (detailed)", "",
              "Generated by `python -m redistricting_addons.sci_grow` (method in its docstring); data in "
              "`site/data/sci_states/<version>.json` (as `sci_states`, plus `border_km`, `today_border_km` and, for the "
              "grown versions, each state's `target` with `dev_pct` measured against it).", ""]
    for v in ("grown", "grown.strong"):
        if v in res:
            lines += [f"### {names[v]}", "", "| Capital | Population | Target | Gap | Cohesion | Draws from today's |",
                      "| --- | --: | --: | --: | --: | --- |"]
            for s_ in sorted(res[v]["states"], key=lambda s: s["name"]):
                lines.append(f"| {s_['name']} | {s_['pop']:,} | {s_.get('target', 0):,} | {s_['dev_pct']:+.2f}% | "
                             f"{s_['cohesion_pct']:.1f}% | {', '.join(s_['today_states'])} |")
            lines.append("")
    (S.ROOT / "docs" / "SCI_GROW.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    import builtins, functools
    globals()["print"] = functools.partial(builtins.print, flush=True)
    ap = argparse.ArgumentParser(prog="python -m redistricting_addons.sci_grow", description=__doc__.split("\n\n")[0])
    ap.add_argument("--variants", nargs="*", default=["grown", "grown.strong", "free.strong"])
    ap.add_argument("--frac", type=float, default=0.05, help="shortest shared border kept, fraction of perimeter")
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--bursts", type=int, default=3000)
    ap.add_argument("--burst", type=int, default=10)
    ap.add_argument("--floor", type=float, default=0.5, help="weight of the least cohesive state")
    ap.add_argument("--cap", type=float, default=0.85, help="free.strong border cap, fraction of today's km")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args(argv)
    U = S.build_units()
    n, k = len(U["pop"]), S.K
    km = unit_borders(U)
    keep = strong_edges(U, km, a.frac)
    print(f"no corner touching: {int((~keep).sum()):,} of {len(keep):,} unit edges dropped")
    U2 = subgraph(U, keep)
    files = S._state_files()
    fips_of = {d["abbr"]: f for f, d in files.items()}
    order = sorted(fips_of)
    today = np.array([order.index(files[s_]["abbr"]) if s_ in files else order.index("MD")
                      for s_ in U["unit_state"].tolist()])
    current = S.score_labels(U, today, k)
    E = U["edges"]
    today_km = float(km[today[E[:, 0]] != today[E[:, 1]]].sum())
    targets = np.bincount(today, weights=U["pop"], minlength=k)
    print(f"today's state lines: {today_km:,.0f} km of unit borders")
    caps = []
    for s, ab in enumerate(order):
        _, la, lo = S.CAPITALS[ab]
        m = np.flatnonzero(U["unit_state"] == fips_of[ab])
        d = (U["lat"][m] - la) ** 2 + ((U["lon"][m] - lo) * np.cos(np.radians(la))) ** 2
        caps.append(int(m[np.argmin(d)]))
    caps = np.array(caps)
    cap_names = {s: (f"{S.CAPITALS[ab][0]} ({ab})", S.CAPITALS[ab][0]) for s, ab in enumerate(order)}
    pos = {key: i for i, key in enumerate(U["keys"].tolist())}

    def load(v):
        r = json.loads((S.OUT / f"{v}.json").read_text(encoding="utf-8"))
        lab = np.full(n, -1, np.int64)
        for s_ in r["states"]:
            lab[[pos[x] for x in s_["units"].split(",")]] = s_["id"]
        return lab, r

    common = dict(full_edges=E, border_km=km, floor=a.floor)
    fc = S.OUT / "free.compact.json"  # neutral equal-population quantiles, computed by sci_states
    nq = json.loads(fc.read_text(encoding="utf-8")).get("neutral_quantiles_5_50_95", {}) if fc.exists() else {}
    for v in a.variants:
        t0 = time.perf_counter()
        print(f"{v}:")
        prev = S.OUT / f"{v}.json"
        resumed = a.resume and prev.exists()
        base = {"params": {"frac": a.frac, "tol": a.tol, "seed": a.seed, "units": n,
                           "edges_dropped": int((~keep).sum())}, "neutral_quantiles_5_50_95": nq}
        if v == "grown":
            lab = grow(U2, caps, targets, a.tol)
            ch = StrongChain(U2, lab, k, a.tol, "raw", capitals=caps, targets=targets, seed=a.seed + 21, **common)
            balance_targets(ch)
            info = {**base, "title": "Today's capitals and populations, each grown through its strongest friendships",
                    "kind": "grown", "objective": "sci-to-capital", "capitals_fixed": True}
        elif v == "grown.strong":
            lab = load(v)[0] if resumed else load("grown")[0]
            ch = StrongChain(U2, lab, k, a.tol, "raw", capitals=caps, targets=targets, max_border_km=today_km,
                             seed=a.seed + 23 + 1000 * resumed, **common)
            short_bursts_ = S.short_bursts(ch, a.bursts, a.burst)
            info = {**base, "title": "Today's capitals and populations, grown by friendships, then made friendship-compact"
                    " (no corner touching, no more border than today)", "kind": "grown", "objective": "compact",
                    "capitals_fixed": True}
            info["params"].update(bursts=a.bursts, burst=a.burst, floor=a.floor, max_border_km=round(today_km, 1))
        elif v == "free.strong":
            lab = load(v)[0] if resumed else tree_seed(Graph(pop=U2["pop"], edges=U2["edges"]), k, a.tol, seed=a.seed)
            ch = StrongChain(U2, lab, k, a.tol, "raw", max_border_km=a.cap * today_km,
                             seed=a.seed + 25 + 1000 * resumed, **common)
            S.short_bursts(ch, a.bursts, a.burst)
            info = {**base, "title": f"50 equal states by social connectedness, strongly friendship-compact (no corner "
                    f"touching, border at most {round(100 * a.cap)}% of today's)", "kind": "free", "objective": "strong",
                    "capitals_fixed": False}
            info["params"].update(bursts=a.bursts, burst=a.burst, floor=a.floor, max_border_km=round(a.cap * today_km, 1))
        else:
            raise SystemExit(f"unknown variant {v}")
        dv = ch.devs()
        assert np.abs(dv).max() <= a.tol + 1e-9, dv
        if ch.caps is not None:
            info["capital_names"] = cap_names
        info["max_dev_pct"] = round(100 * float(np.abs(dv).max()), 3)
        info["seconds"] = round(time.perf_counter() - t0, 1)
        res = _finish(U, ch, v, info, current, km, today_km, targets if v.startswith("grown") else None)
        print(f"  {v}: cohesion {res['cohesion_pct']}, border {res['border_km']:,.0f} km "
              f"(today {today_km:,.0f}), ties kept {res['ties_kept_pct']}, {info['seconds']}s")
    report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
