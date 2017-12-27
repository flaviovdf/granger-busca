# -*- coding: utf8

from gb.collections.fptree import FPTree

from numpy.testing import assert_almost_equal
from numpy.testing import assert_equal


def test_creation():
    tree = FPTree(4)

    assert_equal(tree._get_value(0), 0)
    assert_equal(tree._get_value(1), 0)
    assert_equal(tree._get_value(2), 0)
    assert_equal(tree._get_value(3), 0)
    assert_equal(tree._get_total(), 0)


def test_set_value():
    tree = FPTree(4)
    tree._set_value(0, 0.3)
    tree._set_value(1, 1.5)
    tree._set_value(2, 0.4)
    tree._set_value(3, 0.3)

    assert_equal(tree._get_value(0), 0.3)
    assert_equal(tree._get_value(1), 1.5)
    assert_equal(tree._get_value(2), 0.4)
    assert_equal(tree._get_value(3), 0.3)
    assert_equal(tree._get_total(), 2.5)


def test_set_value_2():
    tree = FPTree(4)
    tree._set_value(0, 0.3)
    tree._set_value(1, 1.5)
    tree._set_value(2, 0.4)
    tree._set_value(3, 0.3)

    assert_equal(tree._get_value(3), 0.3)

    tree._set_value(3, 0.9)
    assert_almost_equal(tree._get_value(3), 0.9)
    assert_almost_equal(tree._get_total(), 3.1)


def test_sample():
    tree = FPTree(4)
    tree._set_value(0, 0.3)
    tree._set_value(1, 1.5)
    tree._set_value(2, 0.4)
    tree._set_value(3, 0.3)

    assert_equal(tree._sample(2.1), 2)
