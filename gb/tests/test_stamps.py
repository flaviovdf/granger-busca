# -*- coding: utf8

from gb.stamps import Timestamps

from numpy.testing import assert_equal
from numpy.testing import assert_array_equal


def test_gets():
    d = {}
    d[0] = [1, 2, 3]
    d[1] = [4, 5]
    d[2] = [6, 7, 8, 9, 10]
    d[3] = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    d[4] = [21]
    stamps = Timestamps(d)

    assert_array_equal([1, 2, 3], stamps._get_stamps(0))
    assert_array_equal([4, 5], stamps._get_stamps(1))
    assert_array_equal([6, 7, 8, 9, 10], stamps._get_stamps(2))
    assert_array_equal([11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                       stamps._get_stamps(3))
    assert_array_equal([21], stamps._get_stamps(4))

    assert_array_equal([5, 5, 5], stamps._get_causes(0))
    assert_array_equal([5, 5], stamps._get_causes(1))
    assert_array_equal([5, 5, 5, 5, 5], stamps._get_causes(2))
    assert_array_equal([5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
                       stamps._get_causes(3))
    assert_array_equal([5], stamps._get_causes(4))


def test_changes():
    d = {}
    d[0] = [1, 2, 3]
    d[1] = [4, 5]
    d[2] = [6, 7, 8, 9, 10]
    d[3] = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    d[4] = [21]
    stamps = Timestamps(d)

    causes = stamps._get_causes(0)
    assert_array_equal([5, 5, 5], stamps._get_causes(0))
    causes[0] = 2
    causes[1] = 3
    causes[2] = 9
    assert_array_equal([2, 3, 9], stamps._get_causes(0))


def test_find_previous():
    d = {}
    d[0] = [1, 2, 3]
    d[1] = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    stamps = Timestamps(d)

    assert_equal(0, stamps._find_previous(0, 1))
    assert_equal(1, stamps._find_previous(0, 2))
    assert_equal(2, stamps._find_previous(0, 3))
    assert_equal(3, stamps._find_previous(0, 4))

    assert_equal(0, stamps._find_previous(1, 1))
    assert_equal(0, stamps._find_previous(1, 11))
    assert_equal(14, stamps._find_previous(1, 14.5))
