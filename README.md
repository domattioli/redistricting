# Redistricting

(better name TBD)

Fair-lines redistricting toys: equal-population, contiguous, compact districts built from Census block groups with meshing and optimal-transport methods.

## Run locally

No build step, just Python 3:

```
git clone https://github.com/domattioli/redistricting && cd redistricting && python -m http.server -d site 8000
```

Then open http://localhost:8000. The tools are the static pages in `site/` ([site/index.html](site/index.html) is the landing page).

## Embed in your site (see embed/README.md)

`site/` is self-contained and copies as a unit into any static site, under any subpath. Copy it, then link or iframe `fair-lines/`, `draw/` and `regions/`; or skip the 39 MB of data and point the toys at jsDelivr with `?data-base=`. Copy-paste iframe snippets, the CSP notes and the three recipes are in [embed/README.md](embed/README.md); a machine-readable description is in [embed/manifest.json](embed/manifest.json).

## Toys

**1. State solver** ([`site/fair-lines/index.html`](site/fair-lines/index.html)). Pick any state and get equal-population, contiguous, compact districts from Census block groups, computed with a balanced centroidal power diagram. A "how far from fair?" least-change slider morphs the fair plan toward the official map and shows how many people must move.

**2. Borderless US** ([`site/fair-lines/index.html`](site/fair-lines/index.html), national view). The same solver run across the whole country: 435 equal-population districts that ignore state lines.

**3. Draw your own** ([`site/draw/index.html`](site/draw/index.html)). Segment-Anything-style drawing: click a block group and a district grows from it. Then hand off with Finish my map, Improve my map or Solve from my seeds, and compare your plan to the official map on a scorecard.

**4. Re-draw the states** ([`site/regions/index.html`](site/regions/index.html)). Regroup the borderless districts into 50 new contiguous states. In progress.

## How it works

Block-group adjacency graph, then a recursive coordinate bisection (RCB) seed, then Lloyd relaxation with optimal-transport balancing (damped Newton on the power-diagram weights), then contiguity repair, then boundary refinement (Fiduccia-Mattheyses style moves) with a penalty for splitting counties.

## What "fair" means here

Equal population, contiguous, compact and county-respecting. No party, race or incumbent data enters the solver. Voting Rights Act compliance and communities of interest are out of scope.

## Data

2020 Census block groups (population and population-weighted centroids). Election data used today: 2016 and 2020 presidential returns from precinct results disaggregated to blocks, plus 2024 presidential county totals spread over block groups by the 2020 pattern. Planned: 2024 precinct-level, and 2026 when certified. Full provenance is in the toy's `SOURCES.md` (see `toys/README.md`).

## Site layout and sync

`site/` is a static, framework-free prototype site (landing page plus the toys). Layout: `site/fair-lines/`, `site/draw/`, `site/regions/` (one `index.html` each), `site/data/` (all shared JSON, about 39 MB: the 51 state files, `national.json` at 9.3 MB, `us.json`, `us_sizefn.json`, `layers/`, `regions/`) and `site/img/`. Pages read data from `window.REDISTRICTING_DATA_BASE`, else `?data-base=`, else `../data/`. No file is near GitHub's 50 MB warning size, so no Git LFS is used; if a future data file passes it, keep it out of git and fetch it at build time.

Refreshing from research: `python scripts/sync_from_research.py /path/to/SegmentAnyMesh` copies the built pages, data, solver package (`redistricting/`, `tests/`) and `docs/METHOD.md`, rewrites the pages' data fetches to the data-base config (it aborts if the upstream strings changed), and records the source commit in `site/SYNC.json`. It is idempotent and read-only on the source. The synced files are generated copies: edit them in SegmentAnyMesh, not here, until migration is complete.

Check the site: `python scripts/verify_site.py` (needs Playwright with Chromium; set `PW_CHROMIUM` to a browser binary if needed). It serves `site/` under the subpath `/projects/redistricting/`, loads every page, runs a PA solve and a draw Finish, checks the `REDISTRICTING_DATA_BASE` / `?data-base=` override against a second server, checks the iframe snippets, and fails on any console error or 404. Python tests: `pytest tests` (needs numpy, scipy, shapely).

## Links

- Write-up: [TBD]
- Origin: research in the SegmentAnyMesh repo (`research/redistricting_*`, `redistricting/`)

## Additions in this fork (neutral baselines, true borders, block level)

### In short (for people)

This fork adds tools that judge a map against **thousands of neutral maps** instead of one "fair" map, following Moon Duchin and Daryl DeFord's work. Everything is added beside the original; nothing existing was changed. In plain terms:

- **A neutral process is not a promise of proportional seats.** Where voters live decides that. In Massachusetts, for example, every possible plan gives Republicans 0 of 9 seats ([Duchin et al. 2018](https://arxiv.org/abs/1810.09051)). The toys therefore now show where a map sits among 10,000 neutral maps (the "Neutral baseline" panel), not only how it compares with the solver's map.
- **True borders.** The toys' neighbour graph is a triangulation of block-group centres; only about three quarters of its links are real shared borders. The fork adds the Census shared-border graph as an option and reports how much results change ([docs/GRAPH_SENSITIVITY.md](docs/GRAPH_SENSITIVITY.md)).
- **Within one person.** An optional step refines a block-group plan to census blocks, so districts differ by at most one person (pilots: Pennsylvania, Rhode Island, New Hampshire).
- **Runs here.** `pip install -e ".[districts]"` now works, and `python -m redistricting_addons.generate --state PA` runs the solver with the official Census population file when the private `research/` copy is missing.

The plan and status are in [docs/PLAN.md](docs/PLAN.md); the tools are described in [redistricting_addons/README.md](redistricting_addons/README.md).

### For AI agents (detailed)

- **Where things live.** `scripts/sync_from_research.py` mirrors `redistricting/` and `site/data/*.json` from SegmentAnyMesh and overwrites the three toy pages, so new code is in `redistricting_addons/`, `site/addons/`, `site/data/ensembles/`, `site/data/graphs/`, `site/data/blocks/`, `tests/` (new files) and `docs/`. The only edits to synced files are one `<script>` line per page and one line in Fair Lines publishing its state (`window.__RD_PAGE`).
- **Random versus deterministic starts.** The browser's Fair Lines "Draw it fresh" starts from random population-weighted centres (`randomCenters`), so runs differ slightly; the Python package defaults to deterministic recursive coordinate bisection (`--init rcb`, seed = state FIPS) and only uses random starts with `--init random`.
- **Results table and research links.** The "Results" table, `research/...` paths and figures referenced by `docs/METHOD.md` come from the SegmentAnyMesh repository and are not reproduced here. `python -m redistricting_addons.generate` reproduces table rows (checked: New Hampshire matches exactly).
- **Population file.** `redistricting.data` pins the hash of a private copy of `CenPop2020_Mean_BG.txt`. The official census.gov file has the same rows and populations but different bytes; `redistricting_addons/cenpop.py` downloads and pins it, `generate.py` passes it with `--cenpop`, and `tests/conftest.py` points the original tests at it when the private copy is missing (all tests pass on a fresh clone).
- **Commands.** `python -m redistricting_addons.batch` (neutral baselines, add `--variants plain-census county20-census` for the Census graph), `python -m redistricting_addons.graphs` (shared-border graphs), `python -m redistricting_addons.graph_report` (sensitivity report), `python -m redistricting_addons.blocks --state PA` (block-level refinement), `python -m redistricting_addons.regions_votes`, `pytest tests`.

### Social connectedness (SCI), added 2026-11-14

In short: Meta's Social Connectedness Index (how likely people in two counties are to be Facebook friends) is available as a lens. The toys' panel shows how many cross-county friendships each map keeps inside its districts; `docs/SCI_DISTRICTS.md` compares maps drawn to keep those ties together with the neutral maps; `docs/SCI_STATES.md` redraws the country as 50 equal-population states by social ties (free, and keeping today's capitals). The index follows migration, party and race, so none of this is presented as neutral.

For agents: `redistricting_addons/sci.py` (data, score), `sci_districts.py` (optimised districts), `sci_states.py` (50 states); details in `redistricting_addons/README.md` and `docs/PLAN.md`.
