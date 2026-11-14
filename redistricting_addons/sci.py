"""Social ties kept together: Meta's Social Connectedness Index (SCI) per state, for scoring district plans.

In short (for people): the SCI says how likely people in two counties are to be Facebook friends. Multiplying it
by both counties' populations gives a rough count of friendships between them. A plan "keeps" a friendship when
both people end up in the same district. This builds, for every state, the county-to-county friendship weights the
ensemble and the toys use to score how many cross-county social ties a plan keeps together, so any map can be
placed among the neutral maps. Two versions: **raw** (mostly says "nearby counties are friends", which compact
districts already reward) and **distance-adjusted** (friendships beyond what distance alone predicts, e.g. a city
and the rural counties its people came from). The distance-adjusted one is the default. SCI is not party- or
race-blind (it follows migration and is lower between places that vote differently), so it is a lens, never a
neutral rule.

For AI agents (detailed):
- Source: Humanitarian Data Exchange dataset ``social-connectedness-index`` (licence "public domain, not
  restricted"), file ``us_counties.csv`` (2026 snapshot, 245 MB, 10,265,616 rows: user_country, friend_country,
  user_region, friend_region, scaled_sci). Downloaded to ``$REDISTRICTING_CACHE/sci/`` (sha256 pinned), never
  committed. Parsed once into ``us_counties.npz`` beside it.
- SCI_ij = links_ij / (users_i * users_j), rescaled; pairs with too few users are absent (privacy) and count as 0.
  Friendship weight w_ij = SCI_ij * P_i * P_j with P = 2020 population of the county in ``site/data/<ST>.json``.
- Distance adjustment: one national least-squares fit log SCI = a + b log(d_km) over all county pairs with
  known centres (population-weighted block-group centres from the 2020 CenPop file, haversine distance);
  adjusted SCI = SCI / exp(a + b log d). Own-county pairs are left out of both versions (the county-splits count
  already covers keeping a county whole).
- Score of a plan: F[c, d] = share of county c's population in district d (block-group level, so split counties
  count fractionally). ties_kept = sum_{i != j} w_ij * sum_d F[i, d] F[j, d] / sum_{i != j} w_ij, in percent.
  ``ties_kept(sci_state, county, pop, labels, k)`` computes it; ``ensemble.plan_stats`` records it as
  ``sci_ties_kept: {raw, adjusted}`` when ``site/data/sci/<ST>.json`` exists.
- Output ``site/data/sci/<ST>.json``: counties (the site's county order), ``raw`` and ``adjusted`` (upper
  triangle i < j, row-major, length C(C-1)/2), ``missing_pairs``, ``fit`` {a, b, pairs}. Connecticut: the 2026
  SCI uses Connecticut's nine planning regions, the site uses its eight 2020 counties, so Connecticut is skipped
  (recorded in ``site/data/sci/index.json``). Counties absent from the SCI file (very small ones dropped for
  privacy, or areas that changed, e.g. Alaska's Valdez-Cordova split) have no ties; each state file lists them
  with their population (``counties_missing_from_sci``).
- ``python -m redistricting_addons.sci [--states PA TX]`` builds the files (all states by default).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from . import cenpop

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "site" / "data" / "sci"
URL = ("https://data.humdata.org/dataset/e9988552-74e4-4ff4-943f-c782ac8bca87/resource/"
       "97dc352f-c9c5-47d6-a6ef-88709e14006c/download/us_counties.csv")
SHA256 = "094dbc533aa5d371585db4b22478c9cd52a7464eb4319884c6e6d0110b85f504"
NAME = "us_counties.csv"


def _dir() -> Path:
    d = cenpop.cache_dir() / "sci"
    d.mkdir(parents=True, exist_ok=True)
    return d


def table():
    """(user, friend, sci) as int64 county FIPS numbers and float SCI, cached as .npz after the first parse."""
    npz = _dir() / "us_counties.npz"
    if npz.exists():
        z = np.load(npz)
        return z["u"], z["f"], z["s"]
    src = cenpop.fetch(URL, _dir() / NAME, SHA256, timeout=1800)
    u, f, s = [], [], []
    with open(src, newline="", encoding="utf-8") as fh:
        r = csv.reader(fh)
        next(r)
        for row in r:
            if row[0] == "US" and row[1] == "US":
                u.append(int(row[2])); f.append(int(row[3])); s.append(float(row[4]))
    u, f, s = np.asarray(u, np.int64), np.asarray(f, np.int64), np.asarray(s, float)
    np.savez(npz, u=u, f=f, s=s)
    return u, f, s


def county_centres() -> dict[int, tuple[float, float]]:
    """Population-weighted (lat, lon) of every 2020 county, from the CenPop block-group file."""
    acc: dict[int, list[float]] = {}
    for r in cenpop.rows():
        c = int(r[0] + r[1]); p = float(r[4])
        a = acc.setdefault(c, [0.0, 0.0, 0.0])
        a[0] += p; a[1] += p * float(r[5]); a[2] += p * float(r[6])
    return {c: (a[1] / a[0], a[2] / a[0]) for c, a in acc.items() if a[0] > 0}


def _km(a, b):
    la1, lo1, la2, lo2 = map(np.radians, (a[:, 0], a[:, 1], b[:, 0], b[:, 1]))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(h))


def distance_fit(u, f, s, centres) -> dict:
    """National fit log SCI = a + b log(km) over pairs of different counties with known centres."""
    keys = np.array(sorted(centres))
    xy = np.array([centres[k] for k in keys])
    iu, jf = np.searchsorted(keys, u), np.searchsorted(keys, f)
    ok = (u != f) & (iu < len(keys)) & (jf < len(keys))
    ok[ok] &= (keys[iu[ok]] == u[ok]) & (keys[jf[ok]] == f[ok])
    d = _km(xy[iu[ok]], xy[jf[ok]])
    good = d > 1
    x, y = np.log(d[good]), np.log(s[ok][good])
    b, a = np.polyfit(x, y, 1)
    return {"a": round(float(a), 5), "b": round(float(b), 5), "pairs": int(good.sum())}


def build_state(st: str, u, f, s, centres, fit) -> dict:
    d = json.loads((ROOT / "site" / "data" / f"{st}.json").read_text(encoding="utf-8"))
    fips = [int(c["fips"]) for c in d["counties"]]
    C = len(fips)
    pos = {c: i for i, c in enumerate(fips)}
    lo, hi = min(fips), max(fips)
    m = (u >= lo) & (u <= hi) & (f >= lo) & (f <= hi) & (u != f)
    sci = np.zeros((C, C))
    for a, b, v in zip(u[m].tolist(), f[m].tolist(), s[m].tolist()):
        if a in pos and b in pos:
            sci[pos[a], pos[b]] = v
    sci = np.maximum(sci, sci.T)
    iu = np.triu_indices(C, 1)
    present = sci[iu] > 0
    seen = set(np.unique(u[m]).tolist())
    covered = sum(1 for c in fips if c in seen)
    cpop = np.bincount(np.asarray(d["cty"], np.int64), weights=np.asarray(d["pop"], float), minlength=C)
    missing = [d["counties"][i]["name"] for i, c in enumerate(fips) if c not in seen]
    xy = np.array([centres.get(c, (np.nan, np.nan)) for c in fips])
    dist = _km(xy[iu[0]], xy[iu[1]])
    expected = np.exp(fit["a"] + fit["b"] * np.log(np.maximum(dist, 1.0)))
    raw = sci[iu]
    adj = np.where(present, raw / expected, 0.0)
    out = {"v": 1, "abbr": st, "counties": [c["fips"] for c in d["counties"]], "pairs": int(len(raw)),
           "missing_pairs": int((~present).sum()), "counties_in_sci": covered,
           "people_in_counties_missing_from_sci": int(sum(cpop[i] for i, c in enumerate(fips) if c not in seen)),
           "counties_missing_from_sci": missing,
           "raw": [int(x) for x in raw], "adjusted": [float(f"{x:.4g}") for x in adj], "fit": fit,
           "source": "Meta Social Connectedness Index, US counties, HDX 2026 snapshot (us_counties.csv)"}
    return out


def matrices(sci_state: dict, pop_by_county) -> dict[str, np.ndarray]:
    """Friendship-weight matrices (zero diagonal, summing to 1) for 'raw' and 'adjusted'."""
    C = len(sci_state["counties"])
    iu = np.triu_indices(C, 1)
    P = np.asarray(pop_by_county, float)
    out = {}
    for name in ("raw", "adjusted"):
        w = np.zeros((C, C))
        w[iu] = np.asarray(sci_state[name], float)
        w = w + w.T
        w *= P[:, None] * P[None, :]
        t = w.sum()
        out[name] = w / t if t > 0 else w
    return out


def ties_kept(weights: dict[str, np.ndarray], county, pop, labels, k: int) -> dict[str, float]:
    """Percent of cross-county friendship weight whose two counties' people share a district (fractional for splits)."""
    county, labels = np.asarray(county, np.int64), np.asarray(labels, np.int64)
    C = next(iter(weights.values())).shape[0]
    fr = np.bincount(county * k + labels, weights=pop, minlength=C * k).reshape(C, k)
    tot = fr.sum(1, keepdims=True)
    fr = np.divide(fr, tot, out=np.zeros_like(fr), where=tot > 0)
    return {name: float(100 * np.sum((w @ fr) * fr)) for name, w in weights.items()}


def load(st: str):
    """The state's SCI file, or None when there is none (not built, or Connecticut)."""
    p = OUT / f"{st}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m redistricting_addons.sci", description=__doc__.split("\n\n")[0])
    ap.add_argument("--states", nargs="*", help="two-letter codes (default: every state)")
    a = ap.parse_args(argv)
    states = a.states or sorted(p.stem for p in (ROOT / "site" / "data").glob("[A-Z][A-Z].json") if p.stem.isupper())
    u, f, s = table()
    centres = county_centres()
    fit = distance_fit(u, f, s, centres)
    print(f"distance fit: log SCI = {fit['a']} + {fit['b']} log km over {fit['pairs']:,} pairs")
    OUT.mkdir(parents=True, exist_ok=True)
    idx_p = OUT / "index.json"
    index = json.loads(idx_p.read_text(encoding="utf-8")) if idx_p.exists() else {}
    index["fit"] = fit
    for st in states:
        res = build_state(st, u, f, s, centres, fit)
        if res["counties_in_sci"] < 0.5 * len(res["counties"]):
            index[st] = {"skipped": f"only {res['counties_in_sci']} of {len(res['counties'])} counties are in the SCI "
                                    "file (county areas changed after 2020)"}
            (OUT / f"{st}.json").unlink(missing_ok=True)
            print(f"{st}: skipped, {index[st]['skipped']}")
            continue
        (OUT / f"{st}.json").write_text(json.dumps(res, separators=(",", ":")), encoding="utf-8")
        index[st] = {"counties": len(res["counties"]), "missing_pairs": res["missing_pairs"], "pairs": res["pairs"],
                     "counties_missing_from_sci": res["counties_missing_from_sci"],
                     "people_in_counties_missing_from_sci": res["people_in_counties_missing_from_sci"]}
        print(f"{st}: {len(res['counties'])} counties, {res['missing_pairs']} of {res['pairs']} pairs missing"
              + (f"; not in SCI: {', '.join(res['counties_missing_from_sci'])} "
                 f"({res['people_in_counties_missing_from_sci']:,} people)" if res["counties_missing_from_sci"] else ""))
    idx_p.write_text(json.dumps(index, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
