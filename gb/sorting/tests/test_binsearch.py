# -*- coding: utf8

from gb.sorting.binsearch import _searchsorted

import numpy as np


def test_searchsorted_size0():
    x = np.zeros(shape=(0, ), dtype='d')
    p = _searchsorted(x, 1)
    assert p == 0

    p = _searchsorted(x, 0)
    assert p == 0

    p = _searchsorted(x, -1)
    assert p == 0


def test_searchsorted_size1():
    x = np.zeros(shape=(1, ), dtype='d')
    x[0] = 10
    p = _searchsorted(x, 1)
    assert p == 0

    p = _searchsorted(x, 0)
    assert p == 0

    p = _searchsorted(x, -1)
    assert p == 0

    p = _searchsorted(x, 10)
    assert p == 0

    p = _searchsorted(x, 11)
    assert p == 1


def test_searchsorted_size2():
    x = np.zeros(shape=(2, ), dtype='d')
    x[0] = 2
    x[1] = 8
    p = _searchsorted(x, 1)
    assert p == 0

    p = _searchsorted(x, 0)
    assert p == 0

    p = _searchsorted(x, -1)
    assert p == 0

    p = _searchsorted(x, 5)
    assert p == 1

    p = _searchsorted(x, 10)
    assert p == 2

    p = _searchsorted(x, 11)
    assert p == 2


def test_searchsorted_numpy():
    values = np.array([1, 2, 3, 4, 5], dtype='d')
    assert np.searchsorted(values, 3.5) == _searchsorted(values, 3.5)

    assert np.searchsorted(values, 3) == _searchsorted(values, 3)

    assert np.searchsorted(values[2:], 3) == _searchsorted(values, 3, 2)-2

    assert np.searchsorted(values[2:], 4) == _searchsorted(values, 4, 2)-2

    assert np.searchsorted(values[2:], 4.5) == _searchsorted(values, 4.5, 2)-2

    assert np.searchsorted(values[2:], -1) == _searchsorted(values, -1, 2)-2

    assert np.searchsorted(values[2:], 10) == _searchsorted(values, 10, 2)-2
