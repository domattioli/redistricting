"""Districts drawn to keep social ties together (Social Connectedness Index), state by state.

In short (for people): for each state this draws a congressional map that keeps as many cross-county
friendships (Meta's Social Connectedness Index) inside one district as it can, while staying contiguous and
within 1% of equal population. It then asks what such a map does: how many seats each party would win, how
compact it is and how many counties it splits, compared with the 10,000 neutral maps of the same state. It is a
lens: social ties follow migration, party and race, so a "connected" map is not neutral by that fact.
``python -m redistricting_addons.sci_districts`` writes ``site/data/sci/districts/<ST>.<raw|adjusted>.json``
and ``docs/SCI_DISTRICTS.md``.

For AI agents (detailed):
- Graph and data: ``ensemble.load_state`` (site block-group graph, its SCI weights from ``site/data/sci/<ST>.json``).
- Objective: ``sci_ties_kept[objective]`` from ``ensemble.plan_stats`` (percent of cross-county friendship weight
  inside one district; split counties fractional). Default objective distance-adjusted, also raw.
- Optimiser: short bursts (Cannon et al. 2023, arXiv:2011.02288): from the best plan so far, ``--burst`` plain
  ReCom moves (``ensemble.recom_chain``, 1% tolerance); the best plan of the burst replaces it when at least as
  good. Starts from the solver's plan. Deterministic for a given ``--seed``.
- Output per state and objective: the plan (``labels``, one district number per block group in site order),
  its ``plan_stats`` (seats per election, cut edges, county splits, ties kept both ways, spanning-tree score),
  its place among the neutral county-weighted and plain ensembles (share of neutral plans with fewer / more D
  seats, cut-edge and ties-kept percentiles), and the run parameters. ``docs/SCI_DISTRICTS.md`` summarises all states.
"""
from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .ensemble import ROOT, load_state, plan_stats, recom_chain

OUT = ROOT / "site" / "data" / "sci" / "districts"
ENS = ROOT / "site" / "data" / "ensembles"


def optimise(st: str, objective: str, bursts: int, burst: int, seed: int, tol: float = 0.01, compact: bool = False) -> dict:
    g = load_state(ROOT / "site" / "data" / f"{st}.json")
    if g.k < 2 or g.sci is None or "solver" not in g.plans:
        return {}
    best = g.plans["solver"].copy()
    cap = None
    if compact:  # no less compact than the median county-weighted neutral plan (cut edges)
        e = json.loads((ENS / f"{st}.county20.json").read_text(encoding="utf-8"))
        cap = e["cut_edges"]["quantiles"][2]

    def score(lab):
        s = plan_stats(g, lab, g.k)
        return -1.0 if cap is not None and s["cut_edges"] > cap else s["sci_ties_kept"][objective]
    best_s = start_s = score(best)
    t0 = time.perf_counter()
    for b in range(bursts):
        bb, bl = -1.0, None
        for lab in recom_chain(g, best, g.k, burst, tol, 1.0, seed + 7919 * b):
            s = score(lab)
            if s > bb:
                bb, bl = s, lab.copy()
        if bl is not None and bb >= best_s:
            best, best_s = bl, bb
    stats = plan_stats(g, best, g.k, trees=True)
    out = {"v": 1, "abbr": st, "districts": g.k, "objective": objective,
           "method": "short bursts of ReCom (Cannon et al. 2023) maximising cross-county friendship weight kept "
                     f"inside districts ({objective} SCI), from the solver's plan, {int(tol * 100)}% population tolerance",
           "params": {"bursts": bursts, "burst": burst, "seed": seed, "tol": tol, "compact": compact,
                      **({"max_cut_edges": cap} if cap is not None else {})},
           "seconds": round(time.perf_counter() - t0, 1),
           "start_ties_kept": round(start_s, 3), "ties_kept": {k: round(v, 3) for k, v in stats["sci_ties_kept"].items()},
           "cut_edges": stats["cut_edges"], "county_splits": stats.get("county_splits"),
           "log_spanning_trees": round(stats["log_spanning_trees"], 2), "max_dev_pct": round(stats["max_dev_pct"], 3),
           "elections": {name: {"seats_d": stats[name]["seats_d"],
                                "sorted_d_share": [round(float(x), 4) for x in stats[name]["sorted_d_share"]]}
                         for name in g.elections},
           "labels": best.tolist()}
    out["vs_neutral"] = {}
    for variant, suffix in (("county20", ".county20"), ("plain", "")):
        p = ENS / f"{st}{suffix}.json"
        if not p.exists():
            continue
        e = json.loads(p.read_text(encoding="utf-8"))
        row = {}
        for name, el in e["elections"].items():
            h = {int(k): v for k, v in el["seats_d_histogram"].items()}
            t = sum(h.values()); s = out["elections"][name]["seats_d"]
            row[name] = {"seats_d": s, "neutral_fewer_pct": round(100 * sum(v for k, v in h.items() if k < s) / t, 1),
                         "neutral_more_pct": round(100 * sum(v for k, v in h.items() if k > s) / t, 1),
                         "neutral_median": sorted(h)[np.searchsorted(np.cumsum([h[k] for k in sorted(h)]), t / 2)]}
        q = e["cut_edges"]["quantiles"]
        row["cut_edges_quantiles"] = q
        if "sci_ties_kept" in e:
            row["ties_kept_quantiles"] = {k: v["quantiles"] for k, v in e["sci_ties_kept"].items()}
        out["vs_neutral"][variant] = row
    return out


def _job(args):
    st, obj, bursts, burst, seed, compact = args
    try:
        res = optimise(st, obj, bursts, burst, seed, compact=compact)
    except Exception as err:  # keep the batch going; report the state
        return st, obj, f"error: {err!r}"
    if not res:
        return st, obj, "skipped"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{st}.{obj}{'.compact' if compact else ''}.json").write_text(json.dumps(res, separators=(",", ":")), encoding="utf-8")
    return st, obj + (" compact" if compact else ""), f"{res['start_ties_kept']} -> {res['ties_kept'][obj]}% in {res['seconds']}s"


def _table(suffix: str):
    rows, flagged = [], 0
    for p in sorted(OUT.glob(f"[A-Z][A-Z].adjusted{suffix}.json")):
        a = json.loads(p.read_text(encoding="utf-8"))
        rp = OUT / f"{a['abbr']}.raw{suffix}.json"
        r = json.loads(rp.read_text(encoding="utf-8")) if rp.exists() else None
        v = a["vs_neutral"].get("county20", {})
        tq = v.get("ties_kept_quantiles", {}).get("adjusted")
        seats, flag = [], False
        for el in ("pres2016", "pres2020", "pres2024"):
            x = v.get(el)
            if not x:
                continue
            tag = " ▲" if x["neutral_fewer_pct"] >= 95 else " ▼" if x["neutral_more_pct"] >= 95 else ""
            flag |= bool(tag)
            seats.append(f"{x['seats_d']} ({x['neutral_median']}){tag}")
        flagged += flag
        cq = v.get("cut_edges_quantiles")
        raw_cell = f"{r['ties_kept']['raw']:.1f}%" if r else "–"
        rows.append(f"| {a['abbr']} | {a['districts']} | {a['start_ties_kept']:.1f}% → {a['ties_kept']['adjusted']:.1f}%"
                    f"{f' ({tq[2]:.1f}%)' if tq else ''} | {raw_cell} | {' / '.join(seats)} | "
                    f"{a['cut_edges']:,}{f' ({cq[0]:,.0f}–{cq[4]:,.0f})' if cq else ''} | {a['county_splits']} |")
    head = ["| State | Seats | Ties kept, adjusted: solver → connected (neutral median) | Raw-objective map, raw ties | "
            "D seats 2016 / 2020 / 2024 (neutral median) | Cut edges (neutral 5–95%) | County splits |",
            "| --- | --: | --- | --- | --- | --- | --: |"]
    return head + rows, len(rows), flagged


def _totals(suffix: str) -> str:
    out = []
    for el, yr in (("pres2016", "2016"), ("pres2020", "2020"), ("pres2024", "2024")):
        s = m = up = dn = 0
        for p in OUT.glob(f"[A-Z][A-Z].adjusted{suffix}.json"):
            v = json.loads(p.read_text(encoding="utf-8"))["vs_neutral"].get("county20", {}).get(el)
            if v:
                s += v["seats_d"]; m += v["neutral_median"]
                up += v["seats_d"] > v["neutral_median"]; dn += v["seats_d"] < v["neutral_median"]
        out.append(f"{yr}: {s} vs {m} ({up} states more, {dn} fewer)")
    return "; ".join(out)


def report():
    comp, n_c, f_c = _table(".compact")
    free, n_f, f_f = _table("")
    lines = ["# Districts drawn to keep social ties together", "", "## In short (for people)", "",
             f"For each of {max(n_c, n_f)} states with more than one district, a map was drawn to keep as many friendships "
             "between people in different counties inside one district as it could (Meta's Social Connectedness "
             "Index, adjusted for distance), then compared with 10,000 neutral county-respecting maps of the same state.",
             ""]
    if n_c:
        lines += [f"- **Connected and compact** (no less compact than the typical neutral map): in {f_c} of {n_c} states "
                  "the Democratic seat count falls outside the middle 90% of neutral maps in at least one of 2016, 2020 "
                  "and 2024. Added up, Democratic seats in these states versus the neutral medians: "
                  f"{_totals('.compact')}. So keeping social ties together while staying compact does not tilt the "
                  "national result."]
    if n_f:
        lines += [f"- **Ties only** (no compactness limit): in {f_f} of {n_f} states; added up: {_totals('')}. These maps "
                  "sprawl (they cut far more neighbour links than any neutral map) and lean a few seats Democratic, which "
                  "is why the compact version is the main one."]
    lines += ["", "Social ties follow migration, party and race, so this shows what such a rule would do; it is not a "
              "neutral rule.", "", "---", "", "## For AI agents (detailed)", "",
              "Generated by `python -m redistricting_addons.sci_districts` (`--compact` for the first table). Ties kept = "
              "percent of cross-county friendship weight (SCI × both populations) inside one district. Neutral = "
              "county-weighted ensemble `<ST>.county20.json`; compact = cut edges at most that ensemble's median. "
              "D seats: connected map (neutral median); ▲ / ▼ when at least 95% of neutral maps give fewer / more.", ""]
    if n_c:
        lines += ["### Connected and compact", ""] + comp + [""]
    if n_f:
        lines += ["### Ties only (no compactness limit)", ""] + free + [""]
    (ROOT / "docs" / "SCI_DISTRICTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m redistricting_addons.sci_districts", description=__doc__.split("\n\n")[0])
    ap.add_argument("--states", nargs="*")
    ap.add_argument("--objectives", nargs="*", default=["adjusted", "raw"])
    ap.add_argument("--bursts", type=int, default=400)
    ap.add_argument("--burst", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--compact", action="store_true",
                    help="only accept plans with no more cut edges than the median county-weighted neutral plan")
    a = ap.parse_args(argv)
    if not a.report_only:
        states = a.states or sorted(p.stem for p in (ROOT / "site" / "data" / "sci").glob("[A-Z][A-Z].json"))
        jobs = [(st, o, a.bursts, a.burst, a.seed, a.compact) for st in states for o in a.objectives]
        with Pool(a.workers) as pool:
            for st, o, msg in pool.imap_unordered(_job, jobs):
                print(f"{st} {o}: {msg}", flush=True)
    report()
    print("wrote docs/SCI_DISTRICTS.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
