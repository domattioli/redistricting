"""Tests for redistricting_addons.sci_zcta (ZIP-code social connectedness).

In short (for people): checks that the fast reader of Meta's 9.4 GB ZIP-code file reads its lines exactly, and
that link strengths are ranked from weakest (0) to strongest (255).

For AI agents (detailed): ``_parse_block`` on hand-written lines (header skipped, Windows and Unix line ends,
values of 1 to 10 digits, leading-zero ZIP codes); ``_rank255`` order and range.
"""
import numpy as np

from redistricting_addons import sci_zcta as Z


def test_parse_block():
    buf = (b"user_country,friend_country,user_region,friend_region,scaled_sci\r\n"
           b"US,US,00601,00601,83642549\r\n"
           b"US,US,00601,99950,7\n"
           b"US,US,12345,54321,1000000000\r\n")
    u, f, v = Z._parse_block(buf)
    assert u.tolist() == [601, 601, 12345]
    assert f.tolist() == [601, 99950, 54321]
    assert v.tolist() == [83642549, 7, 1000000000]


def test_rank255():
    r = Z._rank255(np.array([3.0, -1.0, 10.0, 0.5]))
    assert r.min() == 0 and r.max() == 255
    assert np.argsort(r).tolist() == [1, 3, 0, 2]
