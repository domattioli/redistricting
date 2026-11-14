"""pytest set-up added in this fork.

In short (for people): lets the original tests run without the author's private ``research/`` folder, by
pointing them at the official Census copy of the 2020 population file.

For AI agents: only when ``research/redistricting_2020/CenPop2020_Mean_BG.txt`` is missing, this sets
``redistricting.data.CENPOP2020_REPO_PATH`` and ``CENPOP2020_SHA256`` to the official census.gov file and its
hash (``redistricting_addons.cenpop``; same rows and populations, different bytes). With the repository copy
present nothing changes. If the solver's dependencies (shapely) are missing, it does nothing and the solver
tests fail on import as before; ``tests/test_ensemble.py`` needs only numpy and scipy.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from redistricting_addons import cenpop

    if not cenpop.REPO_COPY.exists():
        import redistricting.data as _data

        _data.CENPOP2020_REPO_PATH = cenpop.path()
        _data.CENPOP2020_SHA256 = cenpop.SHA256
except ImportError:  # solver dependencies not installed
    pass
