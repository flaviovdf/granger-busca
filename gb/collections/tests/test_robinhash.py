# -*- coding: utf8

from gb.collections.robinhash import RobinHoodHash

from numpy.testing import assert_equal

import numpy as np


def test_single():
    h = RobinHoodHash()
    h._set(0, 1)
    assert_equal(1, h._get(0))


def test_set_overwrite():
    h = RobinHoodHash()
    h._set(0, 1)
    assert_equal(1, h._get(0))
    h._set(0, 7)
    assert_equal(7, h._get(0))


def test_updates():
    h = RobinHoodHash()
    h._set(0, 2)
    assert_equal(2, h._get(0))
    h._set(0, h._get(0) + 1)
    assert_equal(3, h._get(0))
    h._set(0, h._get(0) - 1)
    assert_equal(2, h._get(0))

    h._set(2, h._get(2) + 1)
    assert_equal(1, h._get(2))


def test_many_seq():
    h = RobinHoodHash()
    for i in range(1000):
        h._set(i, 1)
        assert_equal(1, h._get(i))


def test_buggy_seq():
    h = RobinHoodHash()

    keys = np.array([462086, 145847, 359195, 522353, 685283, 924029, 139382,
                     421709, 788651, 100304, 180565], dtype='i')
    values = np.array([306476, 600756, 160486, 835873, 239423, 909089, 20944,
                       365417, 861805, 406798, 442754], dtype='i')
    assert_equal(11, keys.shape[0])
    assert_equal(11, values.shape[0])
    assert_equal(11, np.unique(keys).shape[0])
    assert_equal(11, np.unique(values).shape[0])
    for i in range(keys.shape[0]):
        h._set(keys[i], values[i])
        assert_equal(values[i], h._get(keys[i]))

    for i in range(keys.shape[0]):
        assert_equal(values[i], h._get(keys[i]))


def test_many_rnd():
    h = RobinHoodHash()
    keys = np.random.randint(1000000, size=1000)
    values = np.random.randint(1000000, size=1000)
    for i in range(keys.shape[0]):
        h._set(keys[i], values[i])
        assert_equal(values[i], h._get(keys[i]))


def test_removals():
    h = RobinHoodHash()
    keys = np.arange(100)
    values = np.arange(100) + 1
    for i in range(keys.shape[0]):
        h._set(keys[i], values[i])
        print(i, h._size())
    assert_equal(100, h._size())

    for i in range(keys.shape[0]):
        h._set(keys[i], 0)
    assert_equal(100, h._size())
