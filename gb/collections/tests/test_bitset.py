# -*- coding: utf8

from gb.collections.bitset import BitSet

from numpy.testing import assert_equal


def test_add_remove():
    N = 100
    bitset = BitSet(N)

    for i in range(N):
        print(i)
        assert_equal(bitset._get(i), 0)

    bitset._add(0)
    assert_equal(bitset._get(0), 1)
    for i in range(1, N):
        assert_equal(bitset._get(i), 0)

    bitset._add(N-1)
    assert_equal(bitset._get(0), 1)
    assert_equal(bitset._get(N-1), 1)
    for i in range(1, N-1):
        assert_equal(bitset._get(i), 0)

    bitset._add(1)
    assert_equal(bitset._get(1), 1)
    bitset._add(2)
    assert_equal(bitset._get(2), 1)
    bitset._add(4)
    assert_equal(bitset._get(4), 1)
    bitset._add(8)
    assert_equal(bitset._get(8), 1)
    bitset._add(16)
    assert_equal(bitset._get(16), 1)

    bitset._add(3)
    assert_equal(bitset._get(3), 1)
    bitset._add(5)
    assert_equal(bitset._get(5), 1)
    bitset._add(7)
    assert_equal(bitset._get(7), 1)
    bitset._add(9)
    assert_equal(bitset._get(9), 1)

    bitset._remove(1)
    assert_equal(bitset._get(1), 0)
    bitset._remove(2)
    assert_equal(bitset._get(2), 0)
    bitset._remove(4)
    assert_equal(bitset._get(4), 0)
    bitset._remove(8)
    assert_equal(bitset._get(8), 0)
    bitset._remove(16)
    assert_equal(bitset._get(16), 0)

    bitset._remove(3)
    assert_equal(bitset._get(3), 0)
    bitset._remove(5)
    assert_equal(bitset._get(5), 0)
    bitset._remove(7)
    assert_equal(bitset._get(7), 0)
    bitset._remove(9)
    assert_equal(bitset._get(9), 0)
