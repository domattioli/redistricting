"""True block-group adjacency graphs from Census boundary shapes, as an alternative to the site's Delaunay graph.

In short (for people): the site connects each block group to its nearest neighbours by a triangulation of
their centre points. This builds the real thing instead: two block groups are neighbours when they share a
stretch of border on land, read from the Census Bureau's 2020 boundary file. Islands and other pieces cut off
by water get an explicit "water link" to their nearest neighbour, so the state stays connected and every such
link is listed. ``python -m redistricting_addons.graphs`` builds ``site/data/graphs/<ST>.json`` for every state.

For AI agents (detailed):
- Source: ``cb_2020_us_bg_500k`` (Census cartographic boundary file, 1:500,000, clipped to the shoreline),
  downloaded once to the cache (``redistricting_addons.cenpop.cache_dir``) and sha256-pinned. Its shared
  boundaries keep identical vertices (the file is generalised from one topology), so adjacency is found by
  exact matching of boundary segments (vertices quantised to 1e-7 degrees): rook adjacency, a shared segment
  of positive length; corner-only contact does not count.
- Units: each site unit's GEOID comes from ``cenpop.geoids`` (the site lists units in the Census file's order;
  checked here by population). Units with no polygon in the boundary file (or none on land) only get water links.
- Water links: while the land graph has more than one component, the smallest component is joined to the
  nearest unit outside it (site ``x, y`` in km); the same idea as ``redistricting.mesh``'s Kruskal re-bridging,
  but recorded separately so a run can use or drop them (``water_links``).
- Output ``site/data/graphs/<ST>.json``: ``edges`` (flat i, j pairs, i < j, site unit indices),
  ``border_km`` (shared border length per edge, equirectangular approximation), ``water_links`` (flat pairs),
  ``water_link_km``, ``no_polygon`` (unit indices), ``components_on_land`` and counts. Not touched by
  ``scripts/sync_from_research.py`` (it mirrors only ``site/data/*.json``, ``layers/`` and ``regions/``).
- Pure numpy/scipy: the shapefile and dBase readers below handle polygon shapefiles only.
"""
from __future__ import annotations

import argparse
import json
import struct
import zipfile
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from . import cenpop

URL = "https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_bg_500k.zip"
SHA256 = "318e550249833310f32b4752961f3f557d19db5fb199aa9b4316b548dd5fee46"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "site" / "data" / "graphs"
Q = 1e7  # vertex quantisation (1e-7 degrees, about 1 cm)


# --------------------------------------------------------------------------- readers

def boundary_files() -> tuple[Path, Path, Path]:
    """Paths to the .shp, .shx and .dbf of the 2020 block-group boundary file, downloading it once."""
    z = cenpop.fetch(URL, cenpop.cache_dir() / "cb_2020_us_bg_500k.zip", SHA256, timeout=900)
    d = cenpop.cache_dir() / "cb_bg"
    stem = d / "cb_2020_us_bg_500k"
    if not all((stem.with_suffix(s)).exists() for s in (".shp", ".shx", ".dbf")):
        d.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(d)
    return stem.with_suffix(".shp"), stem.with_suffix(".shx"), stem.with_suffix(".dbf")


def read_dbf(path: Path, fields: list[str]) -> dict[str, np.ndarray]:
    """Selected character fields of a dBase III file, as arrays of str."""
    blob = path.read_bytes()
    nrec, hlen, rlen = struct.unpack("<IHH", blob[4:12])
    cols, off, pos = {}, 1, 32  # byte 0 of each record is the deletion flag
    while blob[pos] != 0x0D:
        name = blob[pos:pos + 11].split(b"\0")[0].decode("ascii")
        size = blob[pos + 16]
        cols[name] = (off, size)
        off += size
        pos += 32
    recs = np.frombuffer(blob, dtype=f"S{rlen}", count=nrec, offset=hlen)
    out = {}
    for f in fields:
        o, s = cols[f]
        raw = np.frombuffer(recs.tobytes(), dtype=np.uint8).reshape(nrec, rlen)[:, o:o + s]
        out[f] = np.array([bytes(r).decode("latin-1").strip() for r in raw])
    return out


def read_polygons(shp: Path, shx: Path, records) -> dict[int, list[np.ndarray]]:
    """Rings (arrays of lon, lat) of the given record numbers of a polygon shapefile."""
    idx = np.frombuffer(shx.read_bytes(), dtype=">i4", offset=100).reshape(-1, 2)
    data = shp.read_bytes()
    out = {}
    for rec in records:
        off = int(idx[rec, 0]) * 2 + 8  # skip the record header
        stype = struct.unpack("<i", data[off:off + 4])[0]
        if stype == 0:  # null shape
            out[rec] = []
            continue
        if stype not in (5, 15, 25):
            raise ValueError(f"record {rec}: shape type {stype} is not a polygon")
        nparts, npts = struct.unpack("<ii", data[off + 36:off + 44])
        parts = np.frombuffer(data, dtype="<i4", count=nparts, offset=off + 44)
        pts = np.frombuffer(data, dtype="<f8", count=2 * npts, offset=off + 44 + 4 * nparts).reshape(npts, 2)
        bounds = list(parts) + [npts]
        out[rec] = [pts[bounds[i]:bounds[i + 1]] for i in range(nparts)]
    return out


# --------------------------------------------------------------------------- adjacency

def shared_borders(polys: dict[int, list[np.ndarray]]):
    """Pairs of polygons (keys) sharing a boundary segment, with the shared length in km."""
    seg_a, seg_b, owner, lat = [], [], [], []
    for key, rings in polys.items():
        for r in rings:
            q = np.round(r * Q).astype(np.int64)
            a, b = q[:-1], q[1:]
            keep = np.any(a != b, axis=1)
            seg_a.append(a[keep]); seg_b.append(b[keep])
            owner.append(np.full(int(keep.sum()), key, np.int64))
            lat.append(0.5 * (r[:-1, 1] + r[1:, 1])[keep])
    if not seg_a:
        return np.zeros((0, 2), np.int64), np.zeros(0)
    a, b, own, la = np.vstack(seg_a), np.vstack(seg_b), np.concatenate(owner), np.concatenate(lat)
    swap = (a[:, 0] > b[:, 0]) | ((a[:, 0] == b[:, 0]) & (a[:, 1] > b[:, 1]))
    lo = np.where(swap[:, None], b, a); hi = np.where(swap[:, None], a, b)
    keys = np.column_stack([lo, hi])
    order = np.lexsort((own, keys[:, 3], keys[:, 2], keys[:, 1], keys[:, 0]))
    keys, own, lo, hi, la = keys[order], own[order], lo[order], hi[order], la[order]
    same = np.all(keys[1:] == keys[:-1], axis=1) & (own[1:] != own[:-1])
    i = np.flatnonzero(same)
    pa, pb = own[i], own[i + 1]
    # segment length in km (equirectangular, fine at this scale)
    dlon = (hi[i, 0] - lo[i, 0]) / Q * np.cos(np.radians(la[i]))
    dlat = (hi[i, 1] - lo[i, 1]) / Q
    km = 111.195 * np.hypot(dlon, dlat)
    pairs = np.column_stack([np.minimum(pa, pb), np.maximum(pa, pb)])
    uniq, inv = np.unique(pairs, axis=0, return_inverse=True)
    return uniq, np.bincount(inv.ravel(), weights=km, minlength=len(uniq))


def water_links(n: int, edges: np.ndarray, xy: np.ndarray):
    """Join land components: the smallest component to its nearest unit outside, until one remains."""
    links, km = [], []
    e = edges.copy()
    while True:
        a = sparse.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)) if len(e) else sparse.coo_matrix((n, n))
        ncomp, comp = connected_components(a, directed=False)
        if ncomp == 1:
            return np.array(links, np.int64).reshape(-1, 2), np.array(km), ncomp
        sizes = np.bincount(comp)
        small = int(np.argmin(sizes))
        inside, outside = np.flatnonzero(comp == small), np.flatnonzero(comp != small)
        d, j = cKDTree(xy[outside]).query(xy[inside])
        k = int(np.argmin(d))
        u, v = int(inside[k]), int(outside[j[k]])
        links.append((min(u, v), max(u, v))); km.append(float(d[k]))
        e = np.vstack([e, [[min(u, v), max(u, v)]]])


def build_state(st: str, dbf: dict, shp: Path, shx: Path) -> dict:
    site = json.loads((ROOT / "site" / "data" / f"{st}.json").read_text(encoding="utf-8"))
    fips = site["fips"]
    rows = cenpop.rows(fips)
    if len(rows) != site["n"] or any(int(r[4]) != p for r, p in zip(rows, site["pop"])):
        raise RuntimeError(f"{st}: site units do not match the Census file's order")
    geo = [r[0] + r[1] + r[2] + r[3] for r in rows]
    index = {g: i for i, g in enumerate(geo)}
    recs = np.flatnonzero(dbf["STATEFP"] == fips)
    rec_unit = {int(r): index.get(dbf["GEOID"][r]) for r in recs}
    polys = read_polygons(shp, shx, [r for r, u in rec_unit.items() if u is not None])
    pairs, km = shared_borders({rec_unit[r]: rings for r, rings in polys.items()})
    have = np.zeros(site["n"], bool)
    have[[u for u in rec_unit.values() if u is not None]] = True
    xy = np.column_stack([site["x"], site["y"]])
    links, link_km, _ = water_links(site["n"], pairs, xy)
    a = sparse.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(site["n"],) * 2) if len(pairs) else sparse.coo_matrix((site["n"],) * 2)
    land_comp = int(connected_components(a, directed=False)[0])
    return {
        "v": 1, "abbr": st, "n": site["n"],
        "source": "Census 2020 cartographic boundary file cb_2020_us_bg_500k (shoreline-clipped); rook adjacency "
                  "by shared boundary segments; water links join land components to their nearest unit",
        "edges": pairs.ravel().tolist(),
        "border_km": np.round(km, 3).tolist(),
        "water_links": links.ravel().tolist(),
        "water_link_km": np.round(link_km, 2).tolist(),
        "no_polygon": np.flatnonzero(~have).tolist(),
        "components_on_land": land_comp,
        "counts": {"edges": int(len(pairs)), "water_links": int(len(links)), "no_polygon": int((~have).sum()),
                   "site_delaunay_edges": len(site["edges"]) // 2},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m redistricting_addons.graphs", description=__doc__.split("\n\n")[0])
    ap.add_argument("--states", nargs="*", help="two-letter codes (default: every state in site/data)")
    a = ap.parse_args(argv)
    states = a.states or sorted(p.stem for p in (ROOT / "site" / "data").glob("[A-Z][A-Z].json") if p.stem.isupper())
    shp, shx, dbf_path = boundary_files()
    dbf = read_dbf(dbf_path, ["STATEFP", "GEOID"])
    OUT.mkdir(parents=True, exist_ok=True)
    for st in states:
        g = build_state(st.upper(), dbf, shp, shx)
        p = OUT / f"{st.upper()}.json"
        p.write_text(json.dumps(g, separators=(",", ":")), encoding="utf-8")
        c = g["counts"]
        print(f"{st}: {c['edges']} shared-border edges (Delaunay {c['site_delaunay_edges']}), {g['components_on_land']} land "
              f"pieces, {c['water_links']} water links, {c['no_polygon']} units without a polygon, {p.stat().st_size // 1024} kB",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
