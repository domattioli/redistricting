"""Run the original solver command line with the official 2020 population file when the repository copy is missing.

In short (for people): ``python -m redistricting_addons.generate --state PA`` does exactly what
``python -m redistricting --state PA`` does, but works in this fork, which lacks the author's private
``research/`` folder. All of the original options are accepted.

For AI agents: if ``--cenpop`` is not given, ``--pop-year`` is 2020 (the default) and
``research/redistricting_2020/CenPop2020_Mean_BG.txt`` is missing, this adds ``--cenpop <official file>``
(see ``redistricting_addons.cenpop``) and calls ``redistricting.cli.main`` unchanged. The plans are the same
as with the repository copy: same rows, populations and positions. Needs the solver's dependencies
(shapely >= 2.1, matplotlib unless ``--no-render``).
"""
from __future__ import annotations

import sys

from . import cenpop


def with_fallback(argv: list[str]) -> list[str]:
    """``argv`` plus ``--cenpop <official file>`` when the solver would otherwise fail to find its 2020 file."""
    if "--cenpop" in argv or cenpop.REPO_COPY.exists():
        return list(argv)
    if "--pop-year" in argv and argv[argv.index("--pop-year") + 1] != "2020":
        return list(argv)
    return [*argv, "--cenpop", str(cenpop.path())]


def main(argv=None) -> int:
    from redistricting.cli import main as solver_main  # imports shapely; only needed here

    return solver_main(with_fallback(list(sys.argv[1:] if argv is None else argv)))


if __name__ == "__main__":
    raise SystemExit(main())
