# -*- coding: utf8

from gb.collections.table import Table

from numpy.testing import assert_equal

import numpy as np


def test_table_single():
    h = Table(1)
    h._set_cell(0, 0, 1)
    assert_equal(1, h._get_cell(0, 0))
    assert_equal(0, h._get_cell(0, 1))


def test_table_various_rows():
    h = Table(100)
    rows = np.random.randint(0, 100, size=1000)
    cols = np.random.randint(0, 100, size=1000)
    values = np.random.randint(1, 1000, size=1000)

    for i in range(len(rows)):
        h._set_cell(rows[i], cols[i], values[i])
        assert_equal(values[i], h._get_cell(rows[i], cols[i]))
