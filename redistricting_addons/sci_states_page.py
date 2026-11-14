"""Shapes for the interactive 50-state page (``site/sci-states/``): every saved SCI-states map as simplified outlines.

In short (for people): the redrawn 50-state maps were only pictures and data files. This writes, for each saved
version (and for today's states), one small file with each state's outline and its facts (population, capital,
friendship cohesion, 2020 vote, which of today's states it draws from), so the page can draw them and show a
state's details on click. ``python -m redistricting_addons.sci_states_page``.

For AI agents (detailed):
- Units: ``sci_states.build_units``; block-group polygons (``sci_states._bg_polygons``) are merged into unit
  polygons once (shapely ``unary_union``, cached as ``sci/unit_shapes.pkl``), then per version into state
  polygons, simplified (0.02 degrees before projection) and projected: lower 48 by x = lon cos 38°, y = lat;
  Alaska scaled 0.35 about its centre and moved below California; Hawaii moved beside it (the usual insets).
  Coordinates are rounded to 0.01 and flipped (y down) for SVG.
- Output ``site/data/sci_states/shapes/<version>.json``: {name, title, kind, objective, capitals_fixed, cohesion,
  ties_kept, border_km, max_dev_pct, view: [x0, y0, w, h], states: [{id, name, capital, pop, target, dev_pct,
  cohesion_pct, d2020, today_states, largest_county, paths: [SVG path strings]}]} and ``today.json`` (today's 50
  states, DC with Maryland) in the same form; ``index.json`` lists them. 2020 votes: block-group estimates from
  ``site/data/<ST>.json``; DC counted as 92% Democratic of 45% turnout (as ``samples.states``).
- Needs shapely (``pip install shapely``); nothing here changes the maps.
"""
from __future__ import annotations

import json
import pickle
import time

import numpy as np

from . import cenpop
from . import sci_states as S
from .sci_grow import unit_borders

OUT = S.OUT / "shapes"
NAMES = {"free.compact": "Equal states, friendship-compact", "free.strong": "Equal states, strongly compact",
         "free.raw": "Equal states, raw SCI", "free.adjusted": "Equal states, distance-adjusted SCI",
         "grown": "Grown from today's capitals", "grown.strong": "Grown from capitals, then compact",
         "capitals.compact": "Today's capitals, equal states", "capitals.raw": "Today's capitals, raw SCI",
         "capitals.adjusted": "Today's capitals, distance-adjusted SCI"}
ORDER = ["today", "free.strong", "free.compact", "grown.strong", "grown", "free.raw", "free.adjusted",
         "capitals.compact", "capitals.raw", "capitals.adjusted"]


def _project(lon, lat):
    """Lower 48 as is (x = lon cos 38°); Alaska and Hawaii as insets. Returns x, y (y down)."""
    lon = np.where(lon > 0, lon - 360, lon)
    c0 = np.cos(np.radians(38))
    x, y = lon * c0, lat.copy()
    ak = lat > 51
    hi = (lat < 30) & (lon < -150)
    # Alaska: shrink about (-150, 62), place at (-120 cos38, 27)
    x = np.where(ak, (lon + 150) * np.cos(np.radians(62)) * 0.35 + (-118.5) * c0, x)
    y = np.where(ak, (lat - 62) * 0.35 + 26.0, y)
    x = np.where(hi, (lon + 157) * np.cos(np.radians(20)) + (-105) * c0, x)
    y = np.where(hi, (lat - 20.5) + 25.0, y)
    return x, -y


def unit_shapes(U):
    p = cenpop.cache_dir() / "sci" / "unit_shapes.pkl"
    if p.exists():
        return pickle.loads(p.read_bytes())
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    t0 = time.perf_counter()
    polys = S._bg_polygons(U)
    bu = U["bg_unit"]
    n = len(U["pop"])
    groups = [[] for _ in range(n)]
    for i, rings in enumerate(polys):
        for r in rings:
            if len(r) >= 4:
                x = np.where(r[:, 0] > 0, r[:, 0] - 360, r[:, 0])
                pg = Polygon(np.column_stack([x, r[:, 1]]))
                groups[bu[i]].append(pg if pg.is_valid else pg.buffer(0))
    out = [unary_union(g).buffer(0) if g else None for g in groups]
    p.write_bytes(pickle.dumps(out))
    print(f"unit shapes: {n:,} units ({time.perf_counter() - t0:.0f}s)")
    return out


def _paths(geom, tol=0.02):
    g = geom.simplify(tol, preserve_topology=True)
    out = []
    for part in getattr(g, "geoms", [g]):
        if part.is_empty or part.geom_type != "Polygon" or part.area < 1e-4:
            continue
        c = part.centroid
        if c.y < 30 and c.x < -161:  # Northwestern Hawaiian Islands (no people): left off the inset
            continue
        d = []
        for ring in [part.exterior, *part.interiors]:
            a = np.asarray(ring.coords)
            x, y = _project(a[:, 0], a[:, 1])
            pts = " ".join(f"{u:.2f},{v:.2f}" for u, v in zip(x, y))
            d.append("M" + pts + "Z")
        out.append("".join(d))
    return out


def _votes(U):
    files = S._state_files()
    rows = [r for r in cenpop.rows() if r[0] != "72"]
    bd, br = np.zeros(len(rows)), np.zeros(len(rows))
    st = [r[0] for r in rows]
    i = 0
    while i < len(rows):
        j = i
        while j < len(rows) and st[j] == st[i]:
            j += 1
        if st[i] in files:
            e = files[st[i]]["elections"]["pres2020"]
            bd[i:j] = e["d"]; br[i:j] = e["r"]
        else:
            pp = np.array([float(r[4]) for r in rows[i:j]])
            bd[i:j] = 0.45 * 0.92 * pp; br[i:j] = 0.45 * 0.08 * pp
        i = j
    n = len(U["pop"])
    return np.bincount(U["bg_unit"], weights=bd, minlength=n), np.bincount(U["bg_unit"], weights=br, minlength=n)


def build(versions=None) -> list[str]:
    from shapely.ops import unary_union
    U = S.build_units()
    shapes = unit_shapes(U)
    ud, ur = _votes(U)
    km = unit_borders(U)
    E = U["edges"]
    files = S._state_files()
    fips_of = {d["abbr"]: f for f, d in files.items()}
    order = sorted(fips_of)
    pos = {key: i for i, key in enumerate(U["keys"].tolist())}
    n = len(U["pop"])
    OUT.mkdir(parents=True, exist_ok=True)
    done = []
    sc = S.Scorer(U["M_raw"], U["region"], U["pop"], 50, own=True)
    for v in versions or ORDER:
        if v == "today":
            lab = np.array([order.index(files[s]["abbr"]) if s in files else order.index("MD")
                            for s in U["unit_state"].tolist()])
            coh = sc.cohesion(lab)
            P = np.bincount(lab, weights=U["pop"], minlength=50)
            res = {"name": "today", "title": "Today's 50 states (DC with Maryland)", "kind": "today",
                   "capitals_fixed": True, "ties_kept_pct": S.score_labels(U, lab, 50),
                   "cohesion_pct": {"mean": round(100 * float(coh.mean()), 2), "min": round(100 * float(coh.min()), 2)},
                   "states": [{"id": s, "name": files[fips_of[ab]]["name"], "capital": S.CAPITALS[ab][0],
                               "pop": int(P[s]), "cohesion_pct": round(100 * float(coh[s]), 2),
                               "today_states": [ab]} for s, ab in enumerate(order)]}
        else:
            p = S.OUT / f"{v}.json"
            if not p.exists():
                continue
            res = json.loads(p.read_text(encoding="utf-8"))
            lab = np.full(n, -1, np.int64)
            for s_ in res["states"]:
                lab[[pos[x] for x in s_["units"].split(",")]] = s_["id"]
        k = int(lab.max()) + 1
        D = np.bincount(lab, weights=ud, minlength=k); R = np.bincount(lab, weights=ur, minlength=k)
        border = float(km[lab[E[:, 0]] != lab[E[:, 1]]].sum())
        states = []
        for s_ in res["states"]:
            s = s_["id"]
            geom = unary_union([shapes[u] for u in np.flatnonzero(lab == s) if shapes[u] is not None])
            st = {key: s_[key] for key in ("id", "name", "capital", "pop", "target", "dev_pct", "cohesion_pct",
                                           "today_states", "largest_county") if key in s_}
            st["d2020"] = round(100 * float(D[s] / max(D[s] + R[s], 1)), 1)
            st["paths"] = _paths(geom)
            states.append(st)
        allx = [float(t) for st in states for d in st["paths"] for t in d.replace("M", " ").replace("Z", " ").replace(",", " ").split()]
        xs, ys = allx[0::2], allx[1::2]
        out = {"name": v, "title": NAMES.get(v, res.get("title", v)) if v != "today" else res["title"],
               "long_title": res.get("title", ""), "kind": res.get("kind"), "objective": res.get("objective"),
               "capitals_fixed": bool(res.get("capitals_fixed")), "ties_kept": res["ties_kept_pct"],
               "cohesion": res.get("cohesion_pct", {}), "border_km": round(border), "max_dev_pct": res.get("max_dev_pct"),
               "d_states_2020": int(np.sum(D / np.maximum(D + R, 1) > 0.5)),
               "view": [round(min(xs) - 0.5, 2), round(min(ys) - 0.5, 2), round(max(xs) - min(xs) + 1, 2),
                        round(max(ys) - min(ys) + 1, 2)],
               "states": states}
        (OUT / f"{v}.json").write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
        done.append(v)
        print(f"  {v}: {(OUT / f'{v}.json').stat().st_size / 1e3:.0f} kB")
    idx = [{"name": v, "title": json.loads((OUT / f"{v}.json").read_text(encoding="utf-8"))["title"]}
           for v in ORDER if (OUT / f"{v}.json").exists()]
    (OUT / "index.json").write_text(json.dumps(idx), encoding="utf-8")
    return done


if __name__ == "__main__":
    build()
