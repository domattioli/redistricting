"""Build ReCom ensembles for many states in parallel, re-running any state whose two chains disagree.

    python -m redistricting_addons.batch                      # every state, both variants, 12 workers
    python -m redistricting_addons.batch --states CA TX --variants plain

A state passes when, for every election, the two chains' Democratic-seat histograms differ by at most
``--max-tv`` (total variation distance) and their median district shares by at most ``--max-share-diff``.
A failing state is re-run with twice the moves, up to ``--retries`` times. Results and the pass/fail record
go to ``site/data/ensembles/<ST>.json`` / ``<ST>.county20.json`` and ``site/data/ensembles/index.json``.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .ensemble import ROOT, load_state, run

OUT = ROOT / "site" / "data" / "ensembles"
VARIANTS = {"plain": (1.0, ""), "county20": (20.0, ".county20")}
# opt-in variants on the Census shared-border graph (redistricting_addons.graphs); not in the default run
GRAPH_VARIANTS = {"plain-census": (1.0, ".census"), "county20-census": (20.0, ".county20.census")}
ALL_VARIANTS = {**VARIANTS, **GRAPH_VARIANTS}


def all_states():
    # glob is case-insensitive on Windows, so "[A-Z][A-Z]" also matches us.json: keep upper-case stems only
    return sorted(p.stem for p in (ROOT / "site" / "data").glob("[A-Z][A-Z].json") if p.stem.isupper())


def out_path(st, variant):
    return OUT / f"{st}{ALL_VARIANTS[variant][1]}.json"


def default_steps(k):
    return max(50000, 3000 * k)


def check(res, max_tv, max_share):
    """(passed, worst seats TV, worst median-share difference) for one result."""
    agr = res.get("chain_agreement")
    if agr is None:  # at-large state: nothing sampled
        return True, 0.0, 0.0
    tv = max(v["seats_total_variation"] for k, v in agr.items() if k != "cut_edges_mean")
    sh = max(v["median_share_max_diff"] for k, v in agr.items() if k != "cut_edges_mean")
    return tv <= max_tv and sh <= max_share, tv, sh


def job(st, variant, steps, burn, every, tol, seed):
    g = load_state(ROOT / "site" / "data" / f"{st}.json", graph="census" if variant in GRAPH_VARIANTS else "site")
    if not g.is_connected():
        raise RuntimeError(f"{st}: graph is not connected")
    res = run(g, g.k, steps, burn, every, tol, ALL_VARIANTS[variant][0], seed, chains=2, log=None)
    p = out_path(st, variant)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, separators=(",", ":")), encoding="utf-8")
    return st, variant, steps, res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m redistricting_addons.batch", description=__doc__.split("\n\n")[0])
    ap.add_argument("--states", nargs="*", help="two-letter codes (default: every state in site/data)")
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS), choices=list(ALL_VARIANTS),
                    help="default: plain county20; add plain-census county20-census for the Census border graph")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--steps", type=int, help="moves per chain (default: max(50000, 3000 x districts))")
    ap.add_argument("--burn", type=int, default=2000)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tv", type=float, default=0.10)
    ap.add_argument("--max-share-diff", type=float, default=0.01)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--skip-existing", action="store_true", help="keep existing results that already pass")
    ap.add_argument("--from-index", action="store_true",
                    help="re-run each state at the move count and seed that passed before (same chains, same numbers; "
                         "use after adding a statistic)")
    a = ap.parse_args(argv)

    states = [s.upper() for s in (a.states or all_states())]
    seats = {st: json.loads((ROOT / "site" / "data" / f"{st}.json").read_text(encoding="utf-8"))["seats"] for st in states}
    index_path = OUT / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}

    todo = []
    for st in sorted(states, key=lambda s: -seats[s]):  # biggest first, so the long jobs start early
        for v in a.variants:
            p = out_path(st, v)
            if a.skip_existing and p.exists():
                ok, tv, sh = check(json.loads(p.read_text(encoding="utf-8")), a.max_tv, a.max_share_diff)
                if ok:
                    index.setdefault(st, {})[v] = index.get(st, {}).get(v) or {"file": p.name, "passed": True,
                                                                                  "seats_tv": tv, "share_diff": sh}
                    print(f"{st} {v}: kept existing (TV {tv}, share diff {sh})")
                    continue
            prev = index.get(st, {}).get(v, {}) if a.from_index else {}
            if prev.get("passed") and prev.get("moves_per_chain") and prev.get("attempt"):
                todo.append((st, v, prev["moves_per_chain"], prev["attempt"] - 1))
            else:
                todo.append((st, v, a.steps or default_steps(seats[st]), 0))

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        running = {ex.submit(job, st, v, steps, a.burn, a.every, a.tol, a.seed + 7 * tries): (st, v, steps, tries)
                   for st, v, steps, tries in todo}
        while running:
            fut = next(as_completed(running))
            st, v, steps, tries = running.pop(fut)
            try:
                _, _, _, res = fut.result()
            except Exception as e:  # noqa: BLE001 - record and go on with the other states
                print(f"{st} {v}: FAILED {e!r}", flush=True)
                index.setdefault(st, {})[v] = {"file": None, "passed": False, "error": repr(e)}
                continue
            ok, tv, sh = check(res, a.max_tv, a.max_share_diff)
            mins = (time.perf_counter() - t0) / 60
            print(f"{st} {v}: {'pass' if ok else 'FAIL'}  TV {tv}  share diff {sh}  {steps} moves/chain  "
                  f"{res.get('seconds', 0)}s  [{mins:.0f} min]", flush=True)
            index.setdefault(st, {})[v] = {"file": out_path(st, v).name, "passed": ok, "seats_tv": tv,
                                           "share_diff": sh, "moves_per_chain": steps, "attempt": tries + 1}
            if not ok and tries < a.retries:
                running[ex.submit(job, st, v, steps * 2, a.burn, a.every, a.tol, a.seed + 7 * (tries + 1))] = \
                    (st, v, steps * 2, tries + 1)
            index_path.write_text(json.dumps(dict(sorted(index.items())), indent=1), encoding="utf-8")
    failed = [(st, v) for st, vs in index.items() for v, r in vs.items() if not r.get("passed")]
    print(f"done in {(time.perf_counter() - t0) / 60:.0f} min; {len(failed)} not passing: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
