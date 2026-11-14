"""Votes per borderless district, for the Re-draw the states sampler.

Reads site/data/national.json (block groups with their borderless-district number and presidential votes)
and writes site/data/ensembles/regions_votes.json: Democratic and Republican votes for each of the 435
districts the regions page shows and each election. Each block group's votes go to the page district under its
point on the page's own district raster, so the totals belong to the districts the page actually draws.

    python -m redistricting_addons.regions_votes
"""
from __future__ import annotations

import base64
import json
import re

import numpy as np
from scipy.ndimage import distance_transform_edt

from .ensemble import ROOT


def decode(doc):
    blob = base64.b64decode(doc["bin"])
    return {L["name"]: np.frombuffer(blob, dtype=np.dtype(L["type"]).newbyteorder("<"), count=L["length"],
                                     offset=L["offset"]) for L in doc["layout"]}


def page_raster():
    """The regions page's district raster (page district id per pixel, -1 outside), as the page decodes it."""
    html = (ROOT / "site" / "regions" / "index.html").read_text(encoding="utf-8")
    m = re.search(r'<script id="regions-data" type="application/json">(.*?)</script>', html, re.S)
    R = json.loads(m.group(1))
    runs = np.frombuffer(base64.b64decode(R["rle"]), dtype="<u4").reshape(-1, 2)
    raster = np.repeat(runs[:, 0].astype(np.int64) - 1, runs[:, 1].astype(np.int64))
    if raster.size != R["W"] * R["H"]:
        raise ValueError("raster length mismatch")
    return R, raster


def main() -> int:
    doc = json.loads((ROOT / "site" / "data" / "national.json").read_text(encoding="utf-8"))
    a = decode(doc)
    plan = a["plan"].astype(np.int64)
    k = int(doc["k"])
    pop = a["pop"].astype(float) + 1e-6
    f = doc["frame"]
    R, raster = page_raster()
    # page frame: px = x_km / km_per_px + tx, py = ty - y_km / km_per_px, times the page's raster scale
    px = (a["x"].astype(float) / f["km_per_px"] + f["tx"]) * R["scale"]
    py = (f["ty"] - a["y"].astype(float) / f["km_per_px"]) * R["scale"]
    ix = np.clip(np.round(px).astype(int), 0, R["W"] - 1)
    iy = np.clip(np.round(py).astype(int), 0, R["H"] - 1)
    # each block group's votes go to the page district under its point; points on water or edges take the nearest
    grid = raster.reshape(R["H"], R["W"])
    _, (ny, nx) = distance_transform_edt(grid < 0, return_indices=True)
    page_id = grid[ny[iy, ix], nx[iy, ix]]
    nd = len(R["district_new"])
    dpop = np.bincount(page_id, weights=pop, minlength=nd)
    out = {"v": 1, "districts": nd,
           "source": "block-group votes from site/data/national.json, each assigned to the regions page district "
                     "under its point (the page's own district raster)",
           "pop_quantiles": [int(x) for x in np.percentile(dpop, [0, 5, 50, 95, 100])], "elections": {}}
    for el in doc["elections"]:
        d = np.bincount(page_id, weights=a[f"{el}_d"].astype(float), minlength=nd)
        r = np.bincount(page_id, weights=a[f"{el}_r"].astype(float), minlength=nd)
        out["elections"][el] = {"d": np.round(d).astype(int).tolist(), "r": np.round(r).astype(int).tolist()}
    p = ROOT / "site" / "data" / "ensembles" / "regions_votes.json"
    p.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {p} ({p.stat().st_size / 1024:.1f} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
