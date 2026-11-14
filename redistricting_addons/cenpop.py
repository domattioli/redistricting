"""The 2020 Census block-group population file, downloaded from census.gov when the repository copy is missing.

In short (for people): the solver expects ``research/redistricting_2020/CenPop2020_Mean_BG.txt``, which only
exists in the author's private repo. This downloads the same data from the Census Bureau instead:
``python -m redistricting_addons.cenpop`` prints the path, and ``python -m redistricting_addons.generate
--state PA`` runs the original solver with it.

For AI agents: the official file (10,583,505 bytes, sha256 pinned below) has the same 242,335 rows and the same
populations as the repository copy (331,449,281 people in the 50 states and DC; PA, AK and CA rows identical in
order, population and position), but different bytes, so it does not satisfy ``redistricting.data``'s own
hash pin. It is therefore stored under its own name and passed explicitly (``cenpop_path=`` / ``--cenpop``),
never swapped in silently. ``site/data/<ST>.json`` lists units in this file's order, so ``geoids(state)``
gives each site unit's 12-digit block-group GEOID.
"""
from __future__ import annotations

import csv
import hashlib
import io
import os
import urllib.request
from pathlib import Path

URL = "https://www2.census.gov/geo/docs/reference/cenpop2020/blkgrp/CenPop2020_Mean_BG.txt"
SHA256 = "e4a7ebbf07a8a8d846cea9d2daeb600dea50b1ec2d7703909e40d45c013cd63a"
NAME = "CenPop2020_Mean_BG.census.txt"
ROOT = Path(__file__).resolve().parents[1]
REPO_COPY = ROOT / "research" / "redistricting_2020" / "CenPop2020_Mean_BG.txt"


def cache_dir() -> Path:
    """Same cache as ``redistricting.data.cache_dir`` (``$REDISTRICTING_CACHE`` or ``~/.cache/redistricting``)."""
    p = Path(os.environ.get("REDISTRICTING_CACHE") or Path.home() / ".cache" / "redistricting")
    p.mkdir(parents=True, exist_ok=True)
    return p


def fetch(url: str, dest: Path, sha256: str | None = None, timeout: int = 300) -> Path:
    """Download ``url`` to ``dest`` once (atomically), checking ``sha256`` when given."""
    if dest.exists() and (sha256 is None or hashlib.sha256(dest.read_bytes()).hexdigest() == sha256):
        return dest
    req = urllib.request.Request(url, headers={"User-Agent": "redistricting-addons/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        blob = r.read()
    if sha256 is not None:
        got = hashlib.sha256(blob).hexdigest()
        if got != sha256:
            raise RuntimeError(f"{url}: sha256 {got[:16]}... does not match the pinned {sha256[:16]}...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(blob)
    tmp.replace(dest)
    return dest


def path() -> Path:
    """Path to the official 2020 file, downloading it on first use."""
    return fetch(URL, cache_dir() / NAME, SHA256)


def rows(fips: str | None = None) -> list[list[str]]:
    """Data rows (STATEFP, COUNTYFP, TRACTCE, BLKGRPCE, POPULATION, LATITUDE, LONGITUDE), optionally one state."""
    text = path().read_bytes().decode("utf-8-sig")
    out = list(csv.reader(io.StringIO(text)))[1:]
    return [r for r in out if r and (fips is None or r[0] == fips)]


def geoids(fips: str) -> list[str]:
    """12-digit block-group GEOIDs of one state, in the order the site's ``<ST>.json`` lists its units."""
    return [r[0] + r[1] + r[2] + r[3] for r in rows(fips)]


def main() -> int:
    print(path())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
