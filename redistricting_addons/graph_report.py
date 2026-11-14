"""How much do the results depend on which units count as neighbours? Site graph versus Census borders.

In short (for people): the toys decide which block groups are neighbours with a triangulation of their centre
points, which also links places across water and across gaps. The Census graph uses only shared land borders
(plus a few recorded water links for islands). This compares the neutral baselines built on each, state by state,
the check Caldera, DeFord, Duchin et al. (2020) recommend after finding that water-adjacency rules alone changed
results in Alaska. ``python -m redistricting_addons.graph_report`` writes ``docs/GRAPH_SENSITIVITY.md`` and
``site/data/ensembles/graph_sensitivity.json``.

For AI agents (detailed): inputs are ``site/data/ensembles/<ST>[.county20].json`` (site graph) and
``<ST>[.county20].census.json`` (Census graph, ``batch --variants plain-census county20-census``), plus
``site/data/graphs/<ST>.json``. Per state and variant it reports: total-variation distance between the two
Democratic-seat histograms per election (0 = identical, 1 = disjoint) and the largest over elections; median
seats on each graph; the official plan's share of samples with fewer / more Democratic seats on each graph;
median county splits; whether the solver's and official plans are contiguous on each graph
(``ensemble.contiguous``); and the graph facts (edges, water links, people on water-linked pieces, share of site
edges that are real shared borders). States with one district are listed but have nothing to compare.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .ensemble import ROOT, contiguous, load_state

ENS = ROOT / "site" / "data" / "ensembles"
GRAPHS = ROOT / "site" / "data" / "graphs"


def _hist(e):
    h = {int(k): v for k, v in e["seats_d_histogram"].items()}
    t = sum(h.values())
    return {k: v / t for k, v in h.items()}


def _tv(a, b):
    return 0.5 * sum(abs(a.get(s, 0) - b.get(s, 0)) for s in set(a) | set(b))


def _median(h):
    c = 0.0
    for s in sorted(h):
        c += h[s]
        if c >= 0.5:
            return s


def _place(h, s):
    return {"fewer": round(100 * sum(v for k, v in h.items() if k < s), 1),
            "more": round(100 * sum(v for k, v in h.items() if k > s), 1)}


def state_rows(st: str) -> dict:
    site_g = load_state(ROOT / "site" / "data" / f"{st}.json")
    cen_g = load_state(ROOT / "site" / "data" / f"{st}.json", graph="census")
    g = json.loads((GRAPHS / f"{st}.json").read_text(encoding="utf-8"))
    site_edges = {tuple(e) for e in site_g.edges.tolist()}
    border = {tuple(e) for e in np.asarray(g["edges"]).reshape(-1, 2).tolist()}
    wl = np.asarray(g["water_links"], np.int64).reshape(-1, 2)
    facts = {
        "units": g["n"], "census_edges": g["counts"]["edges"], "site_edges": len(site_edges),
        "site_edges_that_share_a_border_pct": round(100 * len(site_edges & border) / max(1, len(site_edges)), 1),
        "water_links": int(len(wl)),
        "people_on_water_linked_units": int(site_g.pop[np.unique(wl)].sum()) if len(wl) else 0,
    }
    out = {"abbr": st, "districts": site_g.k, "graph": facts, "variants": {}}
    if site_g.k == 1:
        return out
    out["contiguous"] = {name: {"site": bool(contiguous(site_g, lab, site_g.k)), "census": bool(contiguous(cen_g, lab, cen_g.k))}
                         for name, lab in site_g.plans.items()}
    for variant, suffix in (("plain", ""), ("county20", ".county20")):
        a_p, b_p = ENS / f"{st}{suffix}.json", ENS / f"{st}{suffix}.census.json"
        if not (a_p.exists() and b_p.exists()):
            continue
        a, b = json.loads(a_p.read_text(encoding="utf-8")), json.loads(b_p.read_text(encoding="utf-8"))
        row = {"elections": {}}
        for el in a["elections"]:
            ha, hb = _hist(a["elections"][el]), _hist(b["elections"][el])
            off = b["reference"]["official"][el]["seats_d"] if "official" in b.get("reference", {}) else None
            row["elections"][el] = {
                "seats_tv": round(_tv(ha, hb), 3), "median_seats": {"site": _median(ha), "census": _median(hb)},
                **({"official_seats": off, "official_vs_site": _place(ha, off), "official_vs_census": _place(hb, off)}
                   if off is not None else {}),
            }
        row["max_seats_tv"] = max(v["seats_tv"] for v in row["elections"].values())
        row["median_county_splits"] = {"site": a["county_splits"]["quantiles"][2], "census": b["county_splits"]["quantiles"][2]}
        out["variants"][variant] = row
    return out


def main() -> int:
    states = sorted(p.stem for p in GRAPHS.glob("[A-Z][A-Z].json") if p.stem.isupper())
    rows = [state_rows(st) for st in states]
    (ENS / "graph_sensitivity.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    multi = [r for r in rows if r["districts"] > 1 and "county20" in r["variants"]]
    tv = sorted(multi, key=lambda r: -r["variants"]["county20"]["max_seats_tv"])
    lines = [
        "# Does the neighbour graph change the results?",
        "",
        "## In short (for people)",
        "",
    ]
    big = [r for r in tv if r["variants"]["county20"]["max_seats_tv"] > 0.2]
    shift = [r for r in multi if any(v["median_seats"]["site"] != v["median_seats"]["census"]
                                     for v in r["variants"]["county20"]["elections"].values())]
    border_share = np.mean([r["graph"]["site_edges_that_share_a_border_pct"] for r in rows])
    broken = [r["abbr"] for r in multi if not r["contiguous"]["solver"]["census"]]
    lines += [
        f"The toys treat two block groups as neighbours when a triangulation of their centres links them. On average only "
        f"{border_share:.0f}% of those links are real shared borders; the rest jump across gaps or water. Rebuilding the "
        f"neutral baselines on true Census borders changes the typical (median) number of Democratic seats in "
        f"{len(shift)} of {len(multi)} multi-district states in at least one election, and moves the seat distribution "
        f"by more than 20% (total variation) in {len(big)}. The solver's own plans are not contiguous on true borders in "
        f"{len(broken)} states ({', '.join(broken) if broken else 'none'}).",
        "",
        "Use the Census-border baseline when exact contiguity matters (it is what courts and commissions count); the "
        "site graph stays the default in the toys, and both are selectable in the Neutral baseline panel.",
        "",
        "---",
        "",
        "## For AI agents (detailed)",
        "",
        "Generated by `python -m redistricting_addons.graph_report` from the county-weighted ensembles "
        "(`<ST>.county20.json` vs `<ST>.county20.census.json`); full numbers for both variants are in "
        "`site/data/ensembles/graph_sensitivity.json`. Seat TV = total-variation distance between the two Democratic-seat "
        "histograms (largest over 2016, 2020, 2024). Official fewer/more = share of neutral samples giving Democrats fewer / "
        "more seats than the official plan, 2020 votes, site graph → Census graph.",
        "",
        "| State | Seats | Site edges on real borders | Water links | Seat TV | Median D seats 2016/2020/2024 (site → census) | Official 2020: fewer / more | Median county splits | Solver plan contiguous on borders |",
        "| --- | --: | --: | --: | --: | --- | --- | --- | :-: |",
    ]
    for r in tv:
        v = r["variants"]["county20"]
        med = " / ".join(f"{e['median_seats']['site']}→{e['median_seats']['census']}" for e in v["elections"].values())
        e20 = v["elections"].get("pres2020", {})
        off = (f"{e20['official_vs_site']['fewer']}/{e20['official_vs_site']['more']} → "
               f"{e20['official_vs_census']['fewer']}/{e20['official_vs_census']['more']}") if "official_vs_site" in e20 else "–"
        lines.append(f"| {r['abbr']} | {r['districts']} | {r['graph']['site_edges_that_share_a_border_pct']}% | "
                     f"{r['graph']['water_links']} | {v['max_seats_tv']:.3f} | {med} | {off} | "
                     f"{v['median_county_splits']['site']:.0f} → {v['median_county_splits']['census']:.0f} | "
                     f"{'yes' if r['contiguous']['solver']['census'] else 'no'} |")
    single = [r["abbr"] for r in rows if r["districts"] == 1]
    lines += ["", f"At-large states (nothing to compare): {', '.join(single)}.", ""]
    (ROOT / "docs" / "GRAPH_SENSITIVITY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote docs/GRAPH_SENSITIVITY.md and site/data/ensembles/graph_sensitivity.json ({len(multi)} states compared)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
