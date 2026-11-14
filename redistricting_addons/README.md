# redistricting_addons

## In short (for people)

These tools build a **neutral baseline** for every state: 10,000 district maps drawn by computer with no party data, following the method of Moon Duchin and Daryl DeFord's group (ReCom). The toys then show where any map (yours, the solver's, the official one) falls among them, for seats, compactness and county splits. Run `python -m redistricting_addons.batch` to rebuild everything; the results are already in `site/data/ensembles/`. See `docs/PLAN.md` for the bigger picture.

They also add: **true neighbours** (block groups that share a border, from Census boundary files, instead of the site's triangulation) with a report on how much that changes results; a **block-level refinement** that makes districts equal to within one person; and a way to **run the original solver in this fork** without the author's private data folder.

## For AI agents (detailed)

Add-on tools for the redistricting toys. They are additive: they read the same data as the site and change nothing in `redistricting/`, `site/` pages or existing data. They live outside `redistricting/` because `scripts/sync_from_research.py` mirrors that folder from SegmentAnyMesh and deletes files it does not know. `site/data/ensembles/`, `site/data/graphs/`, `site/data/blocks/` and the new files in `tests/` are also outside the sync's reach.

Needs numpy and scipy only (`generate.py` runs the original solver, which also needs shapely). Downloads (Census population file, boundary files, block shapes) are cached in `$REDISTRICTING_CACHE` or `~/.cache/redistricting` and checked against pinned sha256 hashes where noted.

### ensemble.py: a neutral baseline (ReCom)

Samples thousands of valid plans for a state with ReCom (DeFord, Duchin & Solomon 2019, [arXiv:1911.05725](https://arxiv.org/abs/1911.05725)). Each move merges two neighbouring districts, draws a random spanning tree over their block groups and cuts it at an edge where both sides are within the population tolerance. Every plan is contiguous and balanced, and compact plans are favoured without tuning.

```
python -m redistricting_addons.ensemble --state PA
python -m redistricting_addons.ensemble --state PA --county-weight 20 --out site/data/ensembles/PA.county20.json
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--steps` | 20000 | ReCom moves per chain after burn-in |
| `--burn` | 1000 | moves discarded at the start of each chain |
| `--every` | 10 | keep every n-th plan |
| `--tol` | 0.01 | population tolerance (1% of the ideal; block groups cannot reach legal precision) |
| `--county-weight` | 1 | 1 = plain ReCom; above 1 keeps counties whole more (Clelland et al. 2021, [arXiv:2011.06049](https://arxiv.org/abs/2011.06049)) |
| `--chains` | 2 | chain 1 starts from the solver's plan, chain 2 from a random spanning-tree plan |

Input: `site/data/<ST>.json`. Output: `site/data/ensembles/<ST>.json` (about 5 kB), summaries only:

- `elections.<name>.seats_d_histogram`: how many sampled plans give Democrats each seat count (two-party; a seat = more D than R votes);
- `elections.<name>.sorted_d_share_quantiles`: for each district rank (least to most Democratic), the 5/25/50/75/95% Democratic share;
- `cut_edges`, `county_splits`, `max_dev_pct`: quantiles over the sample;
- `log_spanning_trees`: quantiles of the spanning-tree score, the natural log of each district's number of spanning trees (Kirchhoff's matrix-tree theorem) summed over districts. Compact districts have far more spanning trees, so higher = more compact (Procaccia & Tucker-Foltz 2021). A district in pieces is counted piece by piece. `--no-trees` skips it;
- `chain_agreement`: how far the two chains' seat histograms (total variation) and median district shares differ. Small numbers mean the sample is reliable;
- `reference.solver`, `reference.official`: each existing plan's numbers and the share of sampled plans at or below it.

Cut edges are counted on the site's block-group graph (a Delaunay triangulation of centre points), not on true shared borders. The official plan is assigned to block groups by centre point, so its population deviation here is larger than the real one.

Tests: `python tests/test_ensemble.py` (or `pytest tests/test_ensemble.py`).

### batch.py: every state

`python -m redistricting_addons.batch` builds both variants for every state in parallel (`--workers 12`), re-running a state with twice the moves when its two chains disagree, and records the outcome in `site/data/ensembles/index.json`. `--from-index` re-runs each state at the move count and seed that passed before, so the chains (and every existing number) are reproduced exactly; use it after adding a statistic.

### regions_votes.py: votes for Re-draw the states

`python -m redistricting_addons.regions_votes` writes `site/data/ensembles/regions_votes.json`: presidential votes for each of the 435 districts the regions page draws. Each block group in `site/data/national.json` is placed on the page's own district raster (nearest district for points on water), because the page's districts differ slightly from the borderless plan in `national.json`.

### Pages

`site/addons/ensemble-panel.js` (Draw, Fair Lines) and `site/addons/regions-sampler.js` (Re-draw the states) read these files. Each page loads its add-on with one `<script>` line; Fair Lines also publishes its state with one line (`window.__RD_PAGE`). The panel computes the spanning-tree score for the maps on the page in the browser (reverse Cuthill-McKee order and envelope Cholesky, about 50-90 ms a map); it matches the Python values to 0.01. A second switch picks the neighbour graph ("Site neighbours" or "Census borders"); with Census borders the panel loads `<ST>[.county20].census.json` and `data/graphs/<ST>.json` and computes cut edges and the spanning-tree score of the page's maps on that graph (checked: Pennsylvania's official plan, 1,467 cut edges, score 14,086, as in Python).

### cenpop.py and generate.py: run the original solver here

`redistricting.data` reads `research/redistricting_2020/CenPop2020_Mean_BG.txt` (only in the private SegmentAnyMesh repo) and pins its hash. `cenpop.py` downloads the official census.gov file (same 242,335 rows and populations, different bytes; sha256 `e4a7ebbf…`) and `cenpop.geoids(fips)` gives each site unit's GEOID (the site lists units in this file's order; checked by population and position). `python -m redistricting_addons.generate <solver options>` runs `redistricting.cli.main` with `--cenpop <official file>` added only when the private copy is missing and `--pop-year` is 2020. `tests/conftest.py` does the same for the original tests. `pyproject.toml` makes `pip install -e ".[districts]"` / `".[test]"` work.

### graphs.py: true neighbours from Census borders

`python -m redistricting_addons.graphs` writes `site/data/graphs/<ST>.json` from `cb_2020_us_bg_500k` (Census cartographic boundary file, shoreline-clipped, sha256 pinned): rook adjacency by exact matching of shared boundary segments, `border_km` per edge, and `water_links` joining land components (islands, peninsulas across water) to their nearest unit, recorded separately. Units without a polygon have zero population. Use with `ensemble.py --graph census` (add `--no-water-links` to leave the links out; states with islands then cannot be sampled) or `batch.py --variants plain-census county20-census` (files `<ST>.census.json`, `<ST>.county20.census.json`). When the solver's plan is not contiguous on the chosen graph, chain 1 starts from a random spanning-tree plan instead (recorded in `chains`). Census-graph ensembles: 99 of 100 pass the two-chain check; California county-weighted (`CA.county20.census.json`) does not (seat TV 0.110, median share diff 0.039 after 312,000 moves per chain) and is kept, marked `passed: false` in `index.json`, so the panel warns on it. Its chains agree on median seats (44/45/40), so the sensitivity row is indicative; a longer run or more chains would settle it.

### graph_report.py: does the neighbour graph change the results?

`python -m redistricting_addons.graph_report` compares the site-graph and Census-graph ensembles per state: seat-histogram total variation per election, median seats, the official plan's place on each graph, median county splits, contiguity of the solver's and official plans on each graph, and graph facts. Writes `docs/GRAPH_SENSITIVITY.md` and `site/data/ensembles/graph_sensitivity.json`.

### blocks.py: within one person, at census-block level

`python -m redistricting_addons.blocks --state PA [--plan solver|official]` downloads TIGER/Line `tl_2020_<FIPS>_tabblock20` (block shapes with `POP20`), builds rook block adjacency (hashed segment keys; water blocks are ordinary nodes; islands joined by recorded water links), gives each block its block group's district, repairs districts that are in pieces (smaller pieces join the neighbour sharing most edges), then balances: one hop at a time from the largest surplus toward the nearest deficit, moving border blocks largest first without overshooting and never disconnecting a district, with a final step that swaps one border block each way for an exact net amount. Stops when the largest and smallest districts differ by at most one person. Writes `site/data/blocks/<ST>.<plan>.json` (report) and `<ST>.<plan>.bef.csv.gz` (block GEOID, district 1..k). Pilots committed: PA (336,985 blocks; from 1,265 people apart and 24 pieces to 1 person apart and 17 pieces; 530 blocks / 20,618 people changed; about 50 s), RI and NH (2 blocks each).

### sci.py: social ties kept together (Social Connectedness Index)

`python -m redistricting_addons.sci` downloads Meta's county-to-county Social Connectedness Index (`us_counties.csv` from the Humanitarian Data Exchange, 2026 snapshot, 245 MB, sha256 pinned, cached, never committed) and writes `site/data/sci/<ST>.json`: for each pair of the state's counties the raw SCI and a distance-adjusted SCI (SCI divided by what one national fit, log SCI = a + b log km over 10.2 million county pairs, predicts from the distance between population centres). The score of a plan is the share of cross-county friendship weight (SCI × both counties' populations) whose two counties' people share a district, split counties counting fractionally (`sci.ties_kept`). `ensemble.load_state` loads the file when present, so every sampled and reference plan gets `sci_ties_kept: {raw, adjusted}`; `load_state(..., sci=False)` leaves it out. It adds statistics only: the chains are unchanged, so `batch --from-index` reproduces every existing number. The panel shows a "Social ties kept together" section (distance-adjusted first, raw in brackets) with the page's maps placed among the neutral plans; the browser computation matches Python. Connecticut is skipped (the SCI uses its 2022 planning regions; the site uses its 2020 counties); a few tiny counties absent from the SCI for privacy are listed in each file. SCI follows migration, party and race, so the panel calls it a lens, not a neutral rule.

### sci_states.py: fifty states drawn by social connectedness

`python -m redistricting_addons.sci_states [--bursts 8000] [--variants free.raw free.adjusted capitals.raw capitals.adjusted]` redraws the country as 50 contiguous states of equal population (within 1%) that keep as many cross-county friendships inside one state as it can find. Units are counties, with counties above 100,000 people split into census tracts (67,503 units; 50 states plus DC, Puerto Rico left out). Adjacency: the per-state Census border graphs plus block-group borders across state lines; Alaska is joined to Washington and Hawaii to the nearest lower-48 unit by recorded water links. `free` lets states form anywhere (start: recursive random spanning-tree split); `capitals` keeps each of today's 50 capitals in its own state (start: the free split, then balanced ReCom moves pass surplus capitals toward states without one, `fix_capitals`). Both are then optimised by short bursts of ReCom moves with spanning trees that favour strongly tied edges (Cannon et al. 2023). Score = share of cross-county friendship weight inside one state (own-county pairs left out), raw or distance-adjusted; each output also scores today's 50 states (DC with Maryland) and neutral equal-population states drawn without the SCI. Writes `site/data/sci_states/<variant>.json` (states with unit GEOIDs) and `docs/sci_states/<variant>.png`. The unit graph is cached in the download cache, not committed.

### sci_districts.py: districts drawn to keep social ties together

`python -m redistricting_addons.sci_districts [--bursts 3000] [--workers 10]` draws, for each multi-district state with SCI data, the congressional map that keeps the most cross-county friendship weight inside districts (short bursts of plain ReCom from the solver's plan, 1% tolerance; distance-adjusted and raw objectives), and places it among the neutral ensembles: Democratic seats per election, cut edges, county splits, ties kept. Writes `site/data/sci/districts/<ST>.<objective>.json` (with the plan) and `docs/SCI_DISTRICTS.md`.

Friendship compactness: `--variants free.compact capitals.compact` score each state by its *cohesion*, the share of its people's friendship weight (raw SCI, pairs within one county included) that stays inside the state, and maximise the mean over states while every cut unit edge above today's state lines (2,491) costs 100 / 2,491 points. Each output records per-state cohesion, the mean and minimum, and its cut edges.

### samples.py: today's map, a neutral map and a social-ties map side by side

`python -m redistricting_addons.samples [--states PA NC TX WI] [--fifty free.compact]` writes `docs/samples/<ST>.png` (official map, a neutral county-weighted ReCom map from the ensemble chain, and the compact social-ties map, districts filled by 2020 Democratic share with outlines, seat counts for 2016/2020/2024 in the titles) and `docs/samples/states.png` (today's 50 states, a neutral equal-population ReCom sample, and the chosen social-ties variant, with the number of states voting Democratic in 2020).

### sci_grow.py: no corner touching, stronger compactness, states grown from today's capitals

`python -m redistricting_addons.sci_grow [--variants grown grown.strong free.strong] [--frac 0.05] [--floor 0.5] [--cap 0.85] [--bursts 3000]` adds three things to `sci_states` without changing it. **No corner touching:** each unit edge gets its shared border length in km (block-group borders summed, cached); an edge is kept only when it is at least `--frac` of the shorter land perimeter of its two units (water links kept; dropped edges are put back only where needed to keep the country connected), so states cannot hang together by a corner or a sliver (13,405 of 184,496 unit edges dropped). **Stronger compactness** (`StrongChain`): border measured in km over every unit edge, a soft cap (100 / cap points per km over it), and the least cohesive state's cohesion added with weight `--floor`. **Grown from capitals** (`grow`): each of today's capitals keeps its state's 2020 population (DC with Maryland) and, from one shared priority queue, takes the neighbouring unit whose county is most tied to the capital's county (raw SCI; ties by distance) while it stays within 1% over its target; leftovers join the bordering state whose capital they are most tied to; `balance_targets` re-splits neighbouring pairs between their capitals until every state is within 1% of its own target. `grown.strong` then smooths borders (capitals fixed, cap = today's border km); `free.strong` is the equal-population version with cap 0.85 × today's km. Writes `site/data/sci_states/<variant>.json` (plus `border_km`, per-state `target`), `docs/sci_states/<variant>.png` and `docs/SCI_GROW.md`. `samples --grown` writes `docs/samples/states_grown.png` (today's states beside the three, coloured by 2020 vote).
