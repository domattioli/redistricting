"""Social connectedness at ZIP-code level (ZCTAs): finer ties for districts, and per-link weights for the Draw toy.

In short (for people): the county-level friendship index cannot tell neighbourhoods apart, and most districts lie
inside a few counties. Meta also publishes the index between ZIP-code areas. This reads that file (9.4 GB), places
every block group in its ZIP-code area, and (1) gives each link between neighbouring block groups the strength of
the friendship tie between their ZIP areas, which the Draw toy's new "friendship ties" slider uses, and (2) draws,
for each state, a compact district map that keeps as many ZIP-to-ZIP friendships inside districts as it can, and
compares it with the neutral maps. ``python -m redistricting_addons.sci_zcta``.

For AI agents (detailed):
- Data: HDX "Social Connectedness Index" ``us_zcta.zip`` (Google Drive link on the HDX page; 1.95 GB, ten CSV
  shards, 9.4 GB; columns user_country, friend_country, user_region, friend_region, scaled_sci), cached as
  ``$REDISTRICTING_CACHE/sci/us_zcta.zip`` (sha256 pinned below). ZCTA shapes: Census ``cb_2020_us_zcta520_500k``.
- Block group -> ZCTA: the block group's population centre (CenPop 2020) in a ZCTA polygon (shapely STRtree);
  centres in no ZCTA (water, parks) take the nearest one.
- Reading: each shard is parsed with numpy (fixed layout "US,US,ddddd,ddddd,n\\r\\n", checked per block), in
  parallel processes; a pair is kept when the two ZCTAs share a state (each ZCTA carries a 64-bit mask of the
  states of its block groups). Kept pairs are cached as ``sci/zcta_pairs.npz``.
- Per state: Z ZCTAs, M[z, z'] = SCI (symmetric, own pairs included), distance-adjusted M / exp(a + b log km) with
  a fit over the state's own pairs (ZCTA centres = population-weighted block-group centres).
- Site output ``site/data/sci/zcta/<ST>.json``: ``zcta`` (codes), ``bg_zcta`` (index per site unit), and per site
  graph edge (``edges`` order of ``site/data/<ST>.json``) ``edge_raw`` / ``edge_adj`` = rank of log SCI between the
  endpoints' ZCTAs among the state's edges, 0-255 (same-ZCTA edges use the ZCTA's own SCI). The Draw add-on
  ``site/addons/sci-slider.js`` turns them into link-length penalties.
- Districts: ``ties`` = share of cross-ZCTA friendship weight (SCI x both populations, own pairs left out) whose
  two ZCTAs' people share a district (``sci.ties_kept`` with ZCTAs as regions). Short bursts of plain ReCom from
  the solver plan maximising it (distance-adjusted), with the compactness cap of ``sci_districts`` (cut edges at
  most the county-weighted neutral median). Output ``site/data/sci/districts/<ST>.zcta.json`` (labels, seats per
  election, cut edges, ties kept both ways, neutral placement) and ``docs/SCI_ZCTA.md``.
"""
from __future__ import annotations

import argparse
import json
import time
import zipfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from . import cenpop
from .ensemble import ROOT, load_state, plan_stats, recom_chain

ZIP_URL = "https://drive.usercontent.google.com/download?id=13gbCdgHD-xfkogDoSzcAGKhNLHMPy8tR&export=download&confirm=t"
ZIP_SHA256 = None  # 2026 file: c78a9694b70e742cbf879c0b7b4f309093e9e8a9cd22b65b246736c6ca25e951 (not checked on each run: hashing 1.95 GB through cenpop.fetch reads it all into memory)
SHAPES_URL = "https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_zcta520_500k.zip"
OUT = ROOT / "site" / "data" / "sci" / "zcta"
DIST = ROOT / "site" / "data" / "sci" / "districts"
ENS = ROOT / "site" / "data" / "ensembles"


def _cache():
    d = cenpop.cache_dir() / "sci"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _zip() -> Path:
    return cenpop.fetch(ZIP_URL, _cache() / "us_zcta.zip", ZIP_SHA256, timeout=7200)


def _states_order():
    return sorted({r[0] for r in cenpop.rows() if r[0] != "72"})


def bg_zcta():
    """(rows' GEOIDs, ZCTA code per block group) for every block group outside Puerto Rico. Cached."""
    p = _cache() / "bg_zcta.npy"
    rows = [r for r in cenpop.rows() if r[0] != "72"]
    if p.exists():
        z = np.load(p)
        if len(z) == len(rows):
            return rows, z
    import shapely
    from shapely.geometry import Polygon, MultiPolygon
    from . import graphs
    t0 = time.perf_counter()
    zf = cenpop.fetch(SHAPES_URL, cenpop.cache_dir() / "cb_2020_us_zcta520_500k.zip", None, timeout=1800)
    d = cenpop.cache_dir() / "cb_zcta20"
    stem = d / "cb_2020_us_zcta520_500k"
    if not stem.with_suffix(".shp").exists():
        d.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zf) as z:
            z.extractall(d)
    dbf = graphs.read_dbf(stem.with_suffix(".dbf"), ["ZCTA5CE20"])
    codes = np.array([int(c) for c in dbf["ZCTA5CE20"].tolist()])
    polys = graphs.read_polygons(stem.with_suffix(".shp"), stem.with_suffix(".shx"), list(range(len(codes))))
    geoms, gcode = [], []
    for i in range(len(codes)):
        rings = polys.get(i) or []
        parts = [Polygon(r) for r in rings if len(r) >= 4]
        if parts:
            g = MultiPolygon(parts).buffer(0) if len(parts) > 1 else parts[0].buffer(0)
            geoms.append(g); gcode.append(codes[i])
    gcode = np.array(gcode)
    tree = shapely.STRtree(geoms)
    lat = np.array([float(r[5]) for r in rows]); lon = np.array([float(r[6]) for r in rows])
    pts = shapely.points(lon, lat)
    pi, gi = tree.query(pts, predicate="within")
    out = np.full(len(rows), -1, np.int64)
    out[pi] = gcode[gi]
    miss = np.flatnonzero(out < 0)
    if miss.size:
        near = tree.query_nearest(pts[miss], return_distance=False)
        out[miss[near[0]]] = gcode[near[1]]
    np.save(p, out)
    print(f"block groups -> ZCTAs: {len(rows):,} placed, {miss.size:,} by nearest ({time.perf_counter() - t0:.0f}s)")
    return rows, out


def _parse_block(buf: bytes):
    """Rows of one block of whole lines -> (user zcta, friend zcta, sci) arrays."""
    a = np.frombuffer(buf, np.uint8)
    nl = np.flatnonzero(a == 10)
    if not nl.size:
        return np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros(0, np.int64)
    st = np.r_[0, nl[:-1] + 1]
    ok = (a[st] == 85) & (a[st + 5] == 44) & (a[st + 11] == 44) & (a[st + 17] == 44)  # "U", commas at fixed places
    if not ok.all():  # header or an odd line: drop it (checked: only the header in practice)
        st, nl = st[ok], nl[ok]
    d = a.astype(np.int64) - 48
    u = d[st + 6] * 10000 + d[st + 7] * 1000 + d[st + 8] * 100 + d[st + 9] * 10 + d[st + 10]
    f = d[st + 12] * 10000 + d[st + 13] * 1000 + d[st + 14] * 100 + d[st + 15] * 10 + d[st + 16]
    end = nl - (a[nl - 1] == 13)  # one past the last digit
    v = np.zeros(len(st), np.int64)
    mul = np.ones(len(st), np.int64)
    for k in range(1, 12):
        pos = end - k
        m = pos >= st + 18
        if not m.any():
            break
        v[m] += d[pos[m]] * mul[m]
        mul[m] *= 10
    return u.astype(np.int32), f.astype(np.int32), v


def _read_shard(args):
    zpath, name, maskfile = args
    import os, ctypes
    try:
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x40)
    except Exception:
        pass
    mask = np.load(maskfile)
    U, F, V = [], [], []
    rows = 0
    with zipfile.ZipFile(zpath) as z, z.open(name) as fh:
        rest = b""
        while True:
            chunk = fh.read(64 << 20)
            if not chunk:
                break
            buf = rest + chunk
            cut = buf.rfind(b"\n") + 1
            rest = buf[cut:]
            u, f, v = _parse_block(buf[:cut])
            rows += len(u)
            keep = (mask[u] & mask[f]) != 0
            U.append(u[keep]); F.append(f[keep]); V.append(v[keep])
    return name, rows, np.concatenate(U), np.concatenate(F), np.concatenate(V)


def pairs(workers: int = 8):
    """Kept (user, friend, sci) ZCTA pairs: both ZCTAs share a state. Cached."""
    p = _cache() / "zcta_pairs.npz"
    if p.exists():
        z = np.load(p)
        return z["u"], z["f"], z["s"]
    rows, bz = bg_zcta()
    states = _states_order()
    sbit = {s: i for i, s in enumerate(states)}
    mask = np.zeros(100000, np.uint64)
    for r, zc in zip(rows, bz.tolist()):
        mask[zc] |= np.uint64(1) << np.uint64(sbit[r[0]])
    mf = _cache() / "zcta_state_mask.npy"
    np.save(mf, mask)
    zp = _zip()
    with zipfile.ZipFile(zp) as z:
        names = [i.filename for i in z.infolist() if i.filename.endswith(".csv") and not i.filename.startswith("__")]
    t0 = time.perf_counter()
    U, F, V = [], [], []
    total = 0
    with Pool(min(workers, len(names))) as pool:
        for name, n, u, f, v in pool.imap_unordered(_read_shard, [(str(zp), nm, str(mf)) for nm in names]):
            total += n
            U.append(u); F.append(f); V.append(v)
            print(f"  {name}: {n:,} rows, {len(u):,} kept ({time.perf_counter() - t0:.0f}s)")
    u, f, s = np.concatenate(U), np.concatenate(F), np.concatenate(V)
    np.savez(p, u=u, f=f, s=s)
    print(f"ZCTA pairs: {total:,} rows read, {len(u):,} within a state")
    return u, f, s


def state_matrices(fips: str, rows, bz, u, f, s):
    """For one state: ZCTA codes, block-group -> local ZCTA index (state rows order), M_raw, M_adj, ZCTA pops."""
    idx = np.array([i for i, r in enumerate(rows) if r[0] == fips])
    zs = bz[idx]
    codes, local = np.unique(zs, return_inverse=True)
    Z = len(codes)
    pop = np.array([float(rows[i][4]) for i in idx])
    lat = np.array([float(rows[i][5]) for i in idx]); lon = np.array([float(rows[i][6]) for i in idx])
    zpop = np.bincount(local, weights=pop, minlength=Z)
    w = np.maximum(zpop, 1e-9)
    zlat = np.bincount(local, weights=pop * lat + 1e-9 * lat, minlength=Z) / np.bincount(local, weights=pop + 1e-9, minlength=Z)
    zlon = np.bincount(local, weights=pop * lon + 1e-9 * lon, minlength=Z) / np.bincount(local, weights=pop + 1e-9, minlength=Z)
    m = np.isin(u, codes) & np.isin(f, codes)
    M = np.zeros((Z, Z))
    M[np.searchsorted(codes, u[m]), np.searchsorted(codes, f[m])] = s[m]
    M = np.maximum(M, M.T)
    la, lo = np.radians(zlat), np.radians(zlon)
    h = (np.sin((la[:, None] - la[None, :]) / 2) ** 2
         + np.cos(la)[:, None] * np.cos(la)[None, :] * np.sin((lo[:, None] - lo[None, :]) / 2) ** 2)
    km = np.maximum(2 * 6371.0 * np.arcsin(np.sqrt(np.clip(h, 0, 1))), 1.0)
    iu = np.triu_indices(Z, 1)
    good = M[iu] > 0
    if good.sum() > 10:
        b, a = np.polyfit(np.log(km[iu][good]), np.log(M[iu][good]), 1)
    else:
        a, b = 0.0, 0.0
    Madj = M / np.exp(a + b * np.log(km))
    return {"codes": codes, "local": local, "M_raw": M, "M_adj": Madj, "zpop": zpop, "fit": [float(a), float(b)],
            "missing": int((M.sum(1) == 0).sum())}


def _rank255(x):
    r = np.argsort(np.argsort(x, kind="stable"), kind="stable")
    return np.round(255 * r / max(1, len(x) - 1)).astype(int)


def site_files(states=None):
    rows, bz = bg_zcta()
    u, f, s = pairs()
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in (ROOT / "site" / "data").glob("[A-Z][A-Z].json") if p.stem.isupper())
    for p in files:
        d = json.loads(p.read_text(encoding="utf-8"))
        if states and d["abbr"] not in states:
            continue
        sm = state_matrices(d["fips"], rows, bz, u, f, s)
        if len(sm["local"]) != d["n"]:
            print(f"  {d['abbr']}: unit count differs, skipped"); continue
        e = np.asarray(d["edges"], np.int64).reshape(-1, 2)
        a, b = sm["local"][e[:, 0]], sm["local"][e[:, 1]]
        raw = np.log(np.maximum(sm["M_raw"][a, b], 1e-9))
        adj = np.log(np.maximum(sm["M_adj"][a, b], 1e-300))
        out = {"v": 1, "abbr": d["abbr"], "zcta": [f"{c:05d}" for c in sm["codes"].tolist()],
               "bg_zcta": sm["local"].tolist(), "edge_raw": _rank255(raw).tolist(), "edge_adj": _rank255(adj).tolist(),
               "fit": sm["fit"], "zcta_without_sci": sm["missing"],
               "note": "edge_* = rank (0-255) of log SCI between the endpoints' ZIP areas among this state's site edges"}
        (OUT / f"{d['abbr']}.json").write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
        sd = _cache() / "zcta_state"
        sd.mkdir(exist_ok=True)  # per-state matrices for the district runs (so workers need not load every pair)
        np.savez(sd / f"{d['abbr']}.npz", local=sm["local"], M_raw=sm["M_raw"], M_adj=sm["M_adj"], zpop=sm["zpop"])
    print(f"wrote site/data/sci/zcta/ ({len(list(OUT.glob('*.json')))} states)")


# ---------------------------------------------------------------------------------------------- districts

def _weights(sm):
    out = {}
    P = sm["zpop"]
    for name, M in (("raw", sm["M_raw"]), ("adjusted", sm["M_adj"])):
        w = M.copy()
        np.fill_diagonal(w, 0)
        w = w * P[:, None] * P[None, :]
        t = w.sum()
        out[name] = w / t if t > 0 else w
    return out


def optimise(args):
    st, bursts, burst, seed = args
    from .sci import ties_kept
    try:
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x40)
    except Exception:
        pass
    g = load_state(ROOT / "site" / "data" / f"{st}.json", sci=False)
    if g.k < 2 or "solver" not in g.plans:
        return st, None
    sm = dict(np.load(_cache() / "zcta_state" / f"{st}.npz"))
    W = _weights(sm)
    zl = sm["local"]
    ens = json.loads((ENS / f"{st}.county20.json").read_text(encoding="utf-8"))
    cap = ens["cut_edges"]["quantiles"][2]
    e = g.edges

    def cut(lab):
        return int(np.count_nonzero(lab[e[:, 0]] != lab[e[:, 1]]))

    def score(lab):
        return -1.0 if cut(lab) > cap else ties_kept({"adjusted": W["adjusted"]}, zl, g.pop, lab, g.k)["adjusted"]

    best = g.plans["solver"].copy()
    best_s = start = score(best)
    t0 = time.perf_counter()
    for b in range(bursts):
        bb, bl = -1.0, None
        for lab in recom_chain(g, best, g.k, burst, 0.01, 1.0, seed + 7919 * b):
            sc = score(lab)
            if sc > bb:
                bb, bl = sc, lab.copy()
        if bl is not None and bb >= best_s:
            best, best_s = bl, bb
    ps = plan_stats(g, best, g.k)
    tk = ties_kept(W, zl, g.pop, best, g.k)
    ref = {}
    for nm, lab in (("official", g.plans.get("official")), ("solver", g.plans["solver"])):
        if lab is not None:
            ref[nm] = {k_: round(v, 3) for k_, v in ties_kept(W, zl, g.pop, lab, g.k).items()}
    seats = {el: ps[el]["seats_d"] for el in ("pres2016", "pres2020", "pres2024") if el in ps}
    place = {}
    for el, n_ in seats.items():
        h = {int(k_): v for k_, v in ens["elections"][el]["seats_d_histogram"].items()}
        t = sum(h.values())
        place[el] = {"seats_d": n_, "neutral_fewer": round(sum(v for k_, v in h.items() if k_ < n_) / t, 4),
                     "neutral_more": round(sum(v for k_, v in h.items() if k_ > n_) / t, 4)}
    out = {"v": 1, "abbr": st, "objective": "zcta ties kept (distance-adjusted), compact", "labels": best.tolist(),
           "ties_kept_zcta": {k_: round(v, 3) for k_, v in tk.items()}, "reference_ties_kept_zcta": ref,
           "cut_edges": cut(best), "cut_cap": cap, "seats": place,
           "params": {"bursts": bursts, "burst": burst, "seed": seed, "tol": 0.01},
           "start_score": round(start, 3), "seconds": round(time.perf_counter() - t0, 1)}
    DIST.mkdir(parents=True, exist_ok=True)
    (DIST / f"{st}.zcta.json").write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    return st, out


def report():
    res = {}
    for p in sorted(DIST.glob("*.zcta.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        res[r["abbr"]] = r
    if not res:
        return
    tot = {el: sum(r["seats"][el]["seats_d"] for r in res.values()) for el in ("pres2016", "pres2020", "pres2024")}
    med = {el: 0 for el in tot}
    for st in res:  # neutral (county-weighted) median Democratic seats over the same states
        ens = json.loads((ENS / f"{st}.county20.json").read_text(encoding="utf-8"))
        for el in med:
            h = {int(k_): v for k_, v in ens["elections"][el]["seats_d_histogram"].items()}
            ks = sorted(h)
            c = np.cumsum([h[k_] for k_ in ks]) / sum(h.values())
            med[el] += ks[int(np.searchsorted(c, 0.5))]
    county = {}
    for st in res:
        q = DIST / f"{st}.adjusted.compact.json"
        if q.exists():
            county[st] = json.loads(q.read_text(encoding="utf-8"))
    outl = [st for st, r in res.items() if any(min(v["neutral_fewer"], v["neutral_more"]) >= 0 and
                                                (v["neutral_fewer"] >= 0.95 or v["neutral_more"] >= 0.95)
                                                for v in r["seats"].values())]
    lines = ["# Districts that keep ZIP-code friendships together", "", "## In short (for people)", "",
             f"For {len(res)} states, a compact district map that keeps as many friendships between ZIP-code areas "
             "inside districts as it can (Meta's Social Connectedness Index at ZIP level, which, unlike the county "
             "level, sees neighbourhoods inside a county). Each map is placed among the 10,000 neutral maps of its state.",
             "",
             f"- Democratic seats over these states: {tot['pres2016']} (2016 votes), {tot['pres2020']} (2020), "
             f"{tot['pres2024']} (2024); the neutral medians add up to {med['pres2016']}, {med['pres2020']} and {med['pres2024']}.",
             f"- Maps outside the middle 90% of neutral maps in some election: {', '.join(outl) if outl else 'none'}.",
             "- Social ties follow migration, party and race; these maps are a lens, not a neutral proposal.", "",
             "| State | Ties kept (ZIP, adjusted) | Official | Solver | D seats 2016 / 2020 / 2024 | Neutral maps with fewer / more D seats (2020) |",
             "| --- | --: | --: | --: | --- | --- |"]
    for st, r in res.items():
        ref = r["reference_ties_kept_zcta"]
        s = r["seats"]
        lines.append(f"| {st} | {r['ties_kept_zcta']['adjusted']:.2f}% | {ref.get('official', {}).get('adjusted', float('nan')):.2f}% | "
                     f"{ref.get('solver', {}).get('adjusted', float('nan')):.2f}% | {s['pres2016']['seats_d']} / {s['pres2020']['seats_d']} / "
                     f"{s['pres2024']['seats_d']} | {100 * s['pres2020']['neutral_fewer']:.0f}% / {100 * s['pres2020']['neutral_more']:.0f}% |")
    lines += ["", "---", "", "## For AI agents (detailed)", "",
              "Generated by `python -m redistricting_addons.sci_zcta` (method in its docstring). Plans in "
              "`site/data/sci/districts/<ST>.zcta.json`. Ties kept = share of cross-ZCTA friendship weight (SCI x both "
              "ZCTA populations, own pairs left out) inside one district, distance-adjusted with each state's own fit; "
              "compactness cap = the county-weighted neutral median of cut edges.", ""]
    (ROOT / "docs" / "SCI_ZCTA.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    import builtins, functools
    globals()["print"] = functools.partial(builtins.print, flush=True)
    ap = argparse.ArgumentParser(prog="python -m redistricting_addons.sci_zcta", description=__doc__.split("\n\n")[0])
    ap.add_argument("--states", nargs="*")
    ap.add_argument("--bursts", type=int, default=600)
    ap.add_argument("--burst", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip-districts", action="store_true")
    a = ap.parse_args(argv)
    pairs(a.workers)
    site_files(a.states)
    if not a.skip_districts:
        sts = a.states or sorted(p.stem for p in (ROOT / "site" / "data").glob("[A-Z][A-Z].json") if p.stem.isupper())
        with Pool(a.workers) as pool:
            for st, r in pool.imap_unordered(optimise, [(st, a.bursts, a.burst, a.seed) for st in sts]):
                if r:
                    print(f"  {st}: ties {r['ties_kept_zcta']['adjusted']:.2f}% (start {r['start_score']:.2f}), "
                          f"2020 D {r['seats']['pres2020']['seats_d']}, {r['seconds']}s")
        report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
