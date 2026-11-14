"""Side-by-side sample maps: today's map, a neutral ReCom map and a social-connectedness (SCI) map.

In short (for people): pictures that put three maps of the same place next to each other, coloured by how each
district (or state) voted in 2020: the map in use today, one map drawn by the neutral ReCom sampler without party
data, and the map drawn to keep friendships together. ``python -m redistricting_addons.samples`` writes
``docs/samples/<ST>.png`` for a few states and ``docs/samples/states.png`` for the 50 states.

For AI agents (detailed):
- Districts: block-group polygons from ``graphs.boundary_files`` in ``cenpop.geoids`` order (the site's unit
  order). Official = ``current`` in ``site/data/<ST>.json``; ReCom = 3,000 accepted county-weighted (w = 20) moves of
  ``ensemble.recom_chain`` from the solver plan (seed 1, 1% tolerance), the same chain as the ensembles; SCI = the
  compact, distance-adjusted plan in ``site/data/sci/districts/<ST>.adjusted.compact.json``. Each district is filled by
  its 2020 two-party Democratic share (red-white-blue, 30-70%); titles give Democratic seats with 2016/2020/2024 votes.
- States: units from ``sci_states.build_units``; today = the 50 states (DC with Maryland); neutral = the saved plain
  ReCom sample (``neutral_states_<seed>.npy`` in the cache, written by ``sci_states``); SCI = the given variant's file.
  Colour per state = 2020 two-party Democratic share from ``site/data/<ST>.json`` block-group votes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import cenpop, graphs
from .ensemble import ROOT, load_state, recom_chain

OUT = ROOT / "docs" / "samples"


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("rb", ["#b8433a", "#e9b7ae", "#f4f2ee", "#b3c4e6", "#2f5ba8"])
    return plt, PolyCollection, cmap


def _state_polys(fips: str):
    geo = cenpop.geoids(fips)
    shp, shx, dbf_p = graphs.boundary_files()
    dbf = graphs.read_dbf(dbf_p, ["GEOID"])
    rec = {g: i for i, g in enumerate(dbf["GEOID"].tolist())}
    idx = [rec.get(g) for g in geo]
    polys = graphs.read_polygons(shp, shx, [i for i in idx if i is not None])
    return [polys[i] if i is not None else [] for i in idx]


def _outlines(ax, polys, lab, k, c0, lw=0.9):
    """District (or state) borders: the outline of each district's block groups merged (needs shapely; skipped
    without it)."""
    try:
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
    except ImportError:
        return
    for dist in range(k):
        geoms = []
        for i in np.flatnonzero(np.asarray(lab) == dist):
            for r in polys[i]:
                if len(r) >= 4:
                    pg = Polygon(np.column_stack([np.where(r[:, 0] > 0, r[:, 0] - 360, r[:, 0]) * c0, r[:, 1]]))
                    geoms.append(pg if pg.is_valid else pg.buffer(0))
        if not geoms:
            continue
        u = unary_union(geoms).buffer(0)
        for part in getattr(u, "geoms", [u]):
            if part.geom_type != "Polygon":
                continue
            xs, ys = part.exterior.xy
            if part.area > 1e-4:
                ax.plot(xs, ys, color="#18211f", lw=lw)


def _shares(d, labels, k, el="pres2020"):
    e = d["elections"][el]
    D = np.bincount(labels, weights=np.asarray(e["d"], float), minlength=k)
    R = np.bincount(labels, weights=np.asarray(e["r"], float), minlength=k)
    return D / np.maximum(D + R, 1)


def _seats(d, labels, k):
    return [int(np.sum(_shares(d, labels, k, el) > 0.5)) for el in ("pres2016", "pres2020", "pres2024")]


def districts(st: str, steps: int = 3000) -> Path:
    plt, PolyCollection, cmap = _mpl()
    d = json.loads((ROOT / "site" / "data" / f"{st}.json").read_text(encoding="utf-8"))
    k = int(d["seats"])
    g = load_state(ROOT / "site" / "data" / f"{st}.json", sci=False)
    rc = None
    for lab in recom_chain(g, g.plans["solver"], k, steps, 0.01, 20.0, seed=1):
        rc = lab
    sci_p = ROOT / "site" / "data" / "sci" / "districts" / f"{st}.adjusted.compact.json"
    plans = [("Official map", np.asarray(d["current"]))]
    plans.append(("Neutral ReCom map", rc.copy()))
    if sci_p.exists():
        plans.append(("Social-ties map (compact)", np.asarray(json.loads(sci_p.read_text())["labels"])))
    polys = _state_polys(d["fips"])
    lat0 = float(np.mean([r[:, 1].mean() for p in polys[:200] for r in p])) if polys else 40
    c0 = np.cos(np.radians(lat0))
    fig, axes = plt.subplots(1, len(plans), figsize=(6 * len(plans), 5.6))
    for ax, (name, lab) in zip(np.atleast_1d(axes), plans):
        sh = _shares(d, lab, k)
        verts, fc = [], []
        for i, rings in enumerate(polys):
            for r in rings:
                verts.append(np.column_stack([r[:, 0] * c0, r[:, 1]])); fc.append(cmap(np.clip((sh[lab[i]] - .3) / .4, 0, 1)))
        ax.add_collection(PolyCollection(verts, facecolors=fc, edgecolors=fc, linewidths=.2))
        _outlines(ax, polys, lab, k, c0)
        ax.autoscale_view(); ax.set_aspect("equal"); ax.axis("off")
        s16, s20, s24 = _seats(d, lab, k)
        ax.set_title(f"{name}\nDemocratic seats of {k}: {s16} (2016), {s20} (2020), {s24} (2024)", fontsize=11)
    fig.suptitle(f"{d['name']}: three maps, districts coloured by 2020 vote (red = Republican, blue = Democratic)",
                 fontsize=13, x=.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))  # leave room for the overall title
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{st}.png"
    fig.savefig(out, dpi=130, facecolor="white")
    plt.close(fig)
    return out


def states(variant: str = "free.compact", seed: int = 0) -> Path:
    from . import sci_states as S
    plt, PolyCollection, cmap = _mpl()
    U = S.build_units()
    k = 50
    files = S._state_files()
    order = sorted(d["abbr"] for d in files.values())
    fo = {f: d["abbr"] for f, d in files.items()}
    today = np.array([order.index(fo[s]) if s in fo else order.index("MD") for s in U["unit_state"].tolist()])
    neutral = np.load(cenpop.cache_dir() / "sci" / f"neutral_states_{seed}.npy")
    pos = {x: i for i, x in enumerate(U["keys"].tolist())}
    res = json.loads((S.OUT / f"{variant}.json").read_text(encoding="utf-8"))
    sci = np.full(len(U["pop"]), -1, np.int64)
    for s_ in res["states"]:
        sci[[pos[x] for x in s_["units"].split(",")]] = s_["id"]
    # 2020 votes per block group (U order) from the state files; DC (no state file) counted as 92% D of its people
    rows = [r for r in cenpop.rows() if r[0] != "72"]
    bg_d = np.zeros(len(rows)); bg_r = np.zeros(len(rows))
    start = 0
    st_of = [r[0] for r in rows]
    i = 0
    while i < len(rows):
        j = i
        while j < len(rows) and st_of[j] == st_of[i]:
            j += 1
        f = st_of[i]
        if f in files:
            e = files[f]["elections"]["pres2020"]
            bg_d[i:j] = e["d"]; bg_r[i:j] = e["r"]
        else:  # DC
            pp = np.array([float(r[4]) for r in rows[i:j]])
            bg_d[i:j] = 0.45 * 0.92 * pp; bg_r[i:j] = 0.45 * 0.08 * pp
        i = j
    bu = U["bg_unit"]
    ud = np.bincount(bu, weights=bg_d, minlength=len(U["pop"])); ur = np.bincount(bu, weights=bg_r, minlength=len(U["pop"]))
    polys = S._bg_polygons(U)
    lat, lon = U["bg_lat"], U["bg_lon"]
    main = (lon > -130) & (lat > 24) & (lat < 50)
    c0 = np.cos(np.radians(38))
    plans = [("Today's 50 states", today), ("Neutral ReCom states (equal population)", neutral),
             ("Social-ties states, friendship-compact", sci)]
    fig, axes = plt.subplots(3, 1, figsize=(12, 22))
    for ax, (name, lab) in zip(axes, plans):
        D = np.bincount(lab, weights=ud, minlength=k); R = np.bincount(lab, weights=ur, minlength=k)
        sh = D / np.maximum(D + R, 1)
        bl = lab[bu]
        verts, fc = [], []
        for i in np.flatnonzero(main):
            for r in polys[i]:
                x = np.where(r[:, 0] > 0, r[:, 0] - 360, r[:, 0])
                verts.append(np.column_stack([x * c0, r[:, 1]])); fc.append(cmap(np.clip((sh[bl[i]] - .3) / .4, 0, 1)))
        ax.add_collection(PolyCollection(verts, facecolors=fc, edgecolors=fc, linewidths=.15))
        _outlines(ax, [polys[i] if main[i] else [] for i in range(len(polys))], bl, k, c0, lw=.6)
        ax.autoscale_view(); ax.set_aspect("equal"); ax.axis("off")
        sen = 2 * int(np.sum(sh > .5))
        ax.set_title(f"{name}: {int(np.sum(sh > .5))} of 50 states vote Democratic in 2020 ({sen} senators)", fontsize=12)
    fig.suptitle("Fifty states three ways, coloured by 2020 presidential vote (lower 48 shown)", fontsize=14, x=.01, ha="left")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "states.png"
    fig.savefig(out, dpi=110, facecolor="white")
    plt.close(fig)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m redistricting_addons.samples", description=__doc__.split("\n\n")[0])
    ap.add_argument("--states", nargs="*", default=["PA", "NC", "TX", "WI"])
    ap.add_argument("--fifty", default="free.compact", help="sci_states variant for the 50-state picture ('' to skip)")
    a = ap.parse_args(argv)
    for st in a.states:
        print("wrote", districts(st))
    if a.fifty:
        print("wrote", states(a.fifty))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
