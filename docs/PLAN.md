# Plan: neutral baselines and social connectedness for the redistricting toys

## In short (for people)

This fork adds a **neutral baseline** to the redistricting toys. Instead of judging a map against one "fair" map, it compares any map with 10,000 maps drawn without party data for the same state, the method Moon Duchin and Daryl DeFord's group developed (ReCom). You can now see, for example, that under 2016, 2020 and 2024 votes Pennsylvania's official map gives Democrats as many seats as, or more than, 99% of neutral maps (only 0.1–1% of neutral maps give them more).

- **Done (phases 1 and 2):** baselines for all 50 states, a "Neutral baseline" panel in the Draw and Fair Lines toys, and a sampler in Re-draw the states.
- **Done (phase 3):** true neighbours from Census borders (with a report on how much they change results), block-level refinement to within one person (Pennsylvania, Rhode Island, New Hampshire), packaging so the original solver and tests run in this fork, and README notes.
- **Done (SCI track, first version):** Meta's Social Connectedness Index (who is friends with whom between places) as an optional lens. Districts drawn to keep social ties together, compared with the neutral maps: compact ones do not tilt the national result (2020: 223 Democratic seats against 225 for the neutral medians, 43 states). Fifty new equal-population states drawn by social ties (best keeps 34.2% of cross-county friendships inside one state, against 30.1% for neutral states), and a version that keeps today's capitals (29.9%). See `docs/SCI_DISTRICTS.md` and `docs/SCI_STATES.md`.
- **Rule:** everything is added as an option. Nothing existing is replaced.

The living version of this plan is a shared doc: <https://claude.ai/artifact/95zAVtG7q6fFeg7176KnBK>.

---

## For AI agents (detailed)

### Ground rules

1. **Add, never replace.** Every change is a new file, flag, parameter (defaulting to today's behaviour), toggle, column or row. Do not change existing code paths, defaults, metrics, table rows or README text. Old and new sit side by side. (The repo owner's explicit instruction.)
2. **Stay out of the sync's way.** `scripts/sync_from_research.py` copies this repo's content from the private SegmentAnyMesh repo:
   - it *mirrors* `redistricting/` (`*.py`), `site/data/` (top level, `*.json`), `site/data/layers/` and `site/data/regions/`, deleting unknown files there;
   - it *overwrites* `site/fair-lines/index.html`, `site/draw/index.html`, `site/regions/index.html`, `tests/test_redistricting.py`, `docs/METHOD.md`, `site/SYNC.json`.
   New code therefore lives in `redistricting_addons/`, `site/addons/`, `site/data/ensembles/`, `tests/test_ensemble.py` and `docs/`. The only edits to synced files are one `<script>` line per page and one line in Fair Lines that publishes its state; a re-sync removes those lines (re-add them).
3. **Dependencies:** numpy and scipy only (Python); no build step and no CDN (pages).

### Architecture

| Path | Role |
| --- | --- |
| `redistricting_addons/ensemble.py` | ReCom chain on `site/data/<ST>.json` (block groups, edges + bridges, population, county, votes). Writes summaries to `site/data/ensembles/<ST>.json` (plain) or `<ST>.county20.json` (county weight 20). Functions: `load_state`, `recom_chain`, `tree_seed`, `plan_stats`, `log_spanning_trees`, `summarize`, `reference`, `run`. |
| `redistricting_addons/batch.py` | Runs every state × variant in a process pool; a state passes when, for each election, the two chains' D-seat histograms differ by total variation ≤ 0.10 and median district shares by ≤ 0.01; failures re-run with 2× moves (seed + 7 per attempt), up to 2 retries. Writes `site/data/ensembles/index.json`. `--from-index` reproduces passing runs exactly (same moves and seed). |
| `redistricting_addons/regions_votes.py` | Builds `site/data/ensembles/regions_votes.json`: D/R votes per district of the Re-draw the states page, by placing each block group of `site/data/national.json` on the page's own district raster (nearest district for water). The page's 435 districts do not match `national.json`'s plan one to one (only 67% of people match), so do not renumber. |
| `site/addons/ensemble-panel.js` | Draw + Fair Lines panel: seats histogram with proportional marker, sorted-district box plot, cut edges, spanning-tree score and county splits vs the ensemble, plain/county switch, every-election table, least-change slider readout, efficiency-gap caveat, ReCom row + footnote in Fair Lines' methods table. Draw: reads global `S`, wraps global `refreshAll()`. Fair Lines: its script is an IIFE, so the page publishes `window.__RD_PAGE = { S, isFairNow }` and the panel redraws on a `MutationObserver` of `#ladder`. Uses `setTimeout`, not `requestAnimationFrame` (paused in background tabs and some embedded viewers). Exposes `window.__RD_ENSEMBLE.logTrees` for checks. |
| `site/addons/regions-sampler.js` | Re-draw the states: ReCom on the page's 435-district graph `G.nb` with each state's seat count fixed (cut a random spanning tree where one side has exactly the first state's count). 2,000 warm-up moves, then 1,000 groupings, one every 20 moves; seeded RNG. Compares the page's grouping on Senate seats and Electoral College votes (535, no DC). Reads globals `G`, `SIM`, `dnew`, `adopt`, `render`, `table`. |
| `tests/test_ensemble.py` | Grid tests: reaches all 70 balanced 4×4 bipartitions and favours compact ones; contiguity and tolerance; county weighting reduces splits; known spanning-tree counts (3×3 = 192, 2×2 = 4, path = 1); summaries. Run `python tests/test_ensemble.py` (pytest optional). |
| `redistricting_addons/graphs.py` | Census shared-border graphs `site/data/graphs/<ST>.json` (rook adjacency from `cb_2020_us_bg_500k`, `water_links` for islands). `ensemble.load_state(graph="census", water_links=...)`, CLI `--graph census`, `--no-water-links`; batch variants `plain-census`, `county20-census`. |
| `redistricting_addons/graph_report.py` | Site graph vs Census borders per state → `docs/GRAPH_SENSITIVITY.md`, `site/data/ensembles/graph_sensitivity.json`. |
| `redistricting_addons/blocks.py` | Block-level refinement to within one person (TIGER/Line `tabblock20`, `POP20`); reports and block-equivalency files in `site/data/blocks/`. |
| `redistricting_addons/cenpop.py`, `generate.py`, `pyproject.toml`, `tests/conftest.py` | Official census.gov population file (own pinned hash), GEOIDs of site units, the original solver CLI with `--cenpop` added when `research/` is missing, packaging, and the original tests runnable in this fork. |
| `tests/test_phase3.py` | Border matching (rook only, hashed = exact), water links, block repair and balance to one person, contiguity, Census graph loading, CLI fallback. |

### Ensemble file format (`site/data/ensembles/<ST>[.county20].json`)

`districts`, `samples`, `params` (`moves_per_chain`, `burn_in`, `keep_every`, `pop_tolerance`, `county_weight`, `chains`, `spanning_trees`), `chains` (start, seed, seconds), `quantiles` = [5, 25, 50, 75, 95], `cut_edges`, `max_dev_pct`, `log_spanning_trees`, `county_splits` (histogram + quantiles), `elections.<key>` (`label`, `statewide_d_share`, `seats_d_histogram`, `sorted_d_share_quantiles` per district rank), `chain_agreement`, `reference.solver` / `reference.official` (each plan's numbers and the share of samples at or below it). At-large states (AK, DE, ND, SD, VT, WY) have `districts: 1` and no samples.

### Commands

```bash
python -m redistricting_addons.ensemble --state PA                # one state, plain ReCom
python -m redistricting_addons.ensemble --state PA --county-weight 20 --out site/data/ensembles/PA.county20.json
python -m redistricting_addons.batch --workers 12                  # every state, both variants (about 70-90 min)
python -m redistricting_addons.batch --from-index --workers 12     # reproduce passing runs, e.g. after adding a statistic
python -m redistricting_addons.regions_votes                       # votes for Re-draw the states
python tests/test_ensemble.py
python -m http.server -d site 8000                                 # then open the toys
```

### Research behind it (papers in the owner's local `papers/` folder, not committed)

| Finding | Source | Where it shows up |
| --- | --- | --- |
| ReCom (merge two districts, cut a random spanning tree) samples compact, contiguous, balanced plans without tuning and mixes in tens of thousands of moves; single-unit flip walks do not mix. | [DeFord, Duchin & Solomon 2019](https://arxiv.org/abs/1911.05725), [Cannon et al. 2022](https://arxiv.org/abs/2210.01401), [Najt et al. 2019](https://arxiv.org/abs/1908.08881) | `ensemble.py`, `regions-sampler.js` |
| A neutral process does not give proportional seats (every Massachusetts plan: 0 of 9 Republican seats). | [Duchin et al. 2018](https://arxiv.org/abs/1810.09051) | Panel wording, proportional marker |
| Efficiency gap prescribes a 2× seat bonus; mean-median and partisan bias miss extreme seat counts. | [Duchin 2018](https://arxiv.org/abs/1801.02064), [DeFord et al. 2021](https://arxiv.org/abs/2008.06930), [DeFord & Veomett 2024](https://arxiv.org/abs/2406.12167) | Efficiency-gap caveat; MM/PB deliberately not added |
| Outline scores (Polsby-Popper, Reock) depend on projection and outline detail; cut edges and spanning-tree counts are discrete alternatives. | Duchin, *Political Geometry* ch. 1; [Duchin & Tenner 2018](https://arxiv.org/abs/1808.05860); Procaccia & Tucker-Foltz 2021 | Cut edges, spanning-tree score |
| County-internal edge weights (w = 20) cut county splits to human-drawn levels. | [Clelland et al. 2021](https://arxiv.org/abs/2011.06049) | `--county-weight 20` variant |
| How fixed districts are grouped swings as many seats as a full redraw; water-adjacency rules matter. | [Caldera et al. 2020](https://arxiv.org/abs/2005.12732) | Re-draw the states sampler; phase 3 water switch |
| 2020 Census privacy noise does not prevent equal-population districts. | [Cohen et al. 2022](https://arxiv.org/abs/2203.05085) | Phase 3 block-level step |

### Phases and status (as of 2026-11-12)

**Phase 1: ensembles — done, committed.**
- [x] ReCom module, CLI, tests.
- [x] Pennsylvania pilot (plain and county-weighted).
- [x] All 50 states × 2 variants with `batch.py`; all 100 runs pass (Florida and Maryland county-weighted after longer re-runs: 336,000 and 200,000 moves per chain).

Pennsylvania result (county-weighted ensemble, median and 5–95%): 19 (16–22) counties split; Democratic seats 7 (6–8) in 2016, 8 (7–9) in 2020, 6 (5–8) in 2024. Official plan: 14 splits; 8, 9, 8 seats. In each election at least 99% of samples give Democrats as many seats or fewer (strictly fewer: 88%, 74%, 87%), and only 0.1–1% give more. Solver's plan: typical seats (6, 8, 6), far more compact than any sample (1,221 cut edges vs 1,622–1,905). Caveats: block-group vote estimates, official plan approximated by block-group centres, 1% tolerance.

**Phase 2: toys — done.**
- [x] Neutral baseline panel (Draw, Fair Lines) with seats histogram, box plot, percentiles, proportional marker, plain/county switch, every-election table.
- [x] Least-change slider readout (Fair Lines). Not exercised end to end in the test pane, whose paused `requestAnimationFrame` stops the page's own slider; the readout reuses the verified panel code.
- [x] ReCom row and footnote in Fair Lines' methods table; the ensemble's median population gap (0.96%) equals the GerryChain row's (0.955%), confirming that row reflects the sampler's tolerance.
- [x] Efficiency-gap caveat on both pages.
- [x] Spanning-tree compactness score: Python (`log_spanning_trees`, sparse LU) and browser (reverse Cuthill-McKee + envelope Cholesky, 50–90 ms a map); they agree to 0.01 on PA, CA and TX solver and official plans. Ensembles regenerated with `--from-index`, so every other number is unchanged, except Pennsylvania: its pilot was first run at 50,000 moves per chain and is now at the batch default of 51,000, so its numbers moved slightly.
- [x] Re-draw the states sampler. With 2020 votes the page's grouping gives 48 Democratic senators; 99.6% of neutral groupings give more (stable across seeds). Neutral groupings ignore today's state lines (about 1% of districts stay with their old state), so this mostly measures how much following today's lines shapes the Senate.

**Phase 3: graph, data, docs — done.**
- [x] True block-group adjacency as an option: `redistricting_addons/graphs.py` builds `site/data/graphs/<ST>.json` from the Census 2020 cartographic boundary file (shoreline-clipped; rook adjacency by shared boundary segments; islands joined by recorded water links). Use `ensemble.py --graph census` or `batch.py --variants plain-census county20-census`; the site graph stays the default. Finding: only about three quarters of the site graph's links are real shared borders, and the solver's own plans for several states (e.g. PA, MI, CA) are not contiguous on true borders. Census-graph ensembles: 99 of 100 pass the two-chain check; California county-weighted (`CA.county20.census.json`) does not (seat TV 0.110, median share diff 0.039 after 312,000 moves per chain) and is kept, marked `passed: false` in `index.json`, so the panel warns on it. Its chains agree on median seats (44/45/40), so the sensitivity row is indicative; a longer run or more chains would settle it.
- [x] Water switch and sensitivity report: `--no-water-links` (site bridges or Census water links) and `redistricting_addons/graph_report.py` → `docs/GRAPH_SENSITIVITY.md` (site graph vs Census borders, per state: seat distributions, official plan's place, county splits, contiguity). States with islands cannot be sampled without water links, so the comparison is between the permissive site graph and the strict border graph.
- [x] Optional block-level refinement: `redistricting_addons/blocks.py` (TIGER/Line `tabblock20` with `POP20`): repair pieces, then balance to within one person without disconnecting a district. Pilots in `site/data/blocks/`: PA (1,265 people apart and 24 pieces → 1 person apart and 17 pieces; 530 of 336,985 blocks, 20,618 people changed), RI and NH (2 blocks each). District populations were re-checked independently from the Census counts.
- [x] Packaging: `pyproject.toml` (`pip install -e ".[districts]"`, `".[test]"`); `redistricting_addons/cenpop.py` downloads the official census.gov population file (same rows and populations as the private copy, different bytes, own pinned hash); `python -m redistricting_addons.generate` runs the original solver with it (New Hampshire reproduces the README table row exactly); `tests/conftest.py` lets the original tests use it. All tests pass on a fresh clone.
- [x] README additions (appended, nothing changed): neutral process vs outcome with the Massachusetts example; random starts in the browser vs deterministic RCB in the package; the results table and research links refer to SegmentAnyMesh files.

**SCI track — first version done (2026-11-14).**

- [x] Data: `redistricting_addons/sci.py` downloads the HDX county file (2026 snapshot, sha256 pinned, cached, not committed) and writes `site/data/sci/<ST>.json` (raw and distance-adjusted SCI per county pair; national fit log SCI = 16.64 − 1.257 log km over 10.2 million pairs). Connecticut is skipped (the SCI uses its 2022 planning regions).
- [x] Score: share of cross-county friendship weight (SCI × both populations, own-county pairs left out) inside one district; every sampled and reference plan in the site-graph ensembles now carries it (`sci_ties_kept`; regenerated with `--from-index`, every other number unchanged), and the panel shows it ("Social ties kept together"; browser matches Python exactly).
- [x] Districts: `redistricting_addons/sci_districts.py`, short bursts of ReCom maximising ties kept, with and without a compactness limit (cut edges at most the neutral median). Compact: 4 of 43 states outside the middle 90% of neutral maps in some election; national totals match the neutral medians within 2 seats. Ties only: maps sprawl and lean a few seats Democratic (2020: 232 against 225). `docs/SCI_DISTRICTS.md`.
- [x] Fifty states: `redistricting_addons/sci_states.py`, 67,503 units (counties; tracts in counties above 100,000), equal population within 1%. Free: 34.2% raw (neutral 30.1%, 5–95% 29.4–30.4%); distance-adjusted objective gives sprawling states. Capitals fixed: 29.9% (start: weighted graph Voronoi from the capitals, balanced by two-capital Voronoi splits, which keep both states connected; moves are the same splits with random, SCI-shortened edge lengths). Today's states keep 43.1% but are very unequal in size. `docs/SCI_STATES.md`, maps in `docs/sci_states/`.
- Not done: ZIP-level (ZCTA) SCI for districts; an SCI slider in the Draw toy; a page for the 50-state maps (they are images and JSON for now).

The [Social Connectedness Index](https://data.humdata.org/dataset/social-connectedness-index) (CC0, January 2026 snapshot) gives the relative probability that people in two US counties or ZIP areas (ZCTAs) are Facebook friends: links divided by the product of user counts, rescaled to 1–1,000,000,000 per file, with Gaussian differential-privacy noise and places under 500 users dropped ([data descriptor](https://pages.stern.nyu.edu/~jstroebe/PDF/JKKS_SCI_DateInBrief.pdf)). `us_counties.csv` is 245 MB (keep out of git; download at build time).

- Redraw the 50 states: county graph with SCI on edges (raw and distance-adjusted); score "ties kept within states"; contiguity-constrained average-linkage clustering (reproduce [Bailey et al. 2018](https://pages.stern.nyu.edu/~tkuchler/PDF/JEP_SCI.pdf)'s 50 communities); balanced version by ReCom with SCI-weighted spanning trees; district-level version for `site/regions/`.
- Districts: map block groups to ZCTAs; "ties kept" score with ensemble percentile; SCI-weighted ReCom, or the Marked Edge Walk / Cycle Walk with an SCI term; optional `sci_weight` (default 0) in `refine()` and an SCI layer slider (default 0) in Draw. Prior art with mobility data: [Kruse et al. 2023](https://arxiv.org/abs/2309.13480).
- **Caution:** SCI is not partisan- or race-blind (a 10-point gap in 2008 Obama share goes with 6% fewer links; links follow migration history). Always show SCI maps against the neutral ensemble and check Voting Rights Act effects; never call them neutral.

### Open decisions (owner)

- Offer this work upstream (domattioli/redistricting or SegmentAnyMesh), or keep it in the fork? **Decided (2026-11-13): offered upstream as domattioli/redistricting#3.**
- Population rule for redrawn states: equal population, whole numbers of the 435 districts, or unconstrained? **Decided (2026-11-13): equal population.**
- Raw or distance-adjusted SCI as the main criterion? Owner unsure; results: raw for whole states (adjusted sprawls), adjusted for districts (raw mostly restates compactness). Proposed: compute and show both; distance-adjusted as the default, because raw SCI mostly restates distance, which ReCom's compactness already rewards.
- Add voting-age population by race so ensembles and SCI maps can be checked for Voting Rights Act effects? **Decided (2026-11-13): no.** The SCI caution about race still stands in the docs.

### Risks

- A re-sync from SegmentAnyMesh removes the page hooks (one `<script>` line per page, one state line in Fair Lines).
- Election figures are block-group estimates; the official plan is approximated by block-group centres.
- Re-draw the states votes are placed on the page's raster, so state totals are approximate.
