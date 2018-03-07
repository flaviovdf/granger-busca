# -*- coding: utf8

from gb.collections.table import Table
from gb.samplers import BaseSampler
from gb.samplers import CollapsedGibbsSampler
from gb.stamps import Timestamps

from numpy.testing import assert_equal

import numpy as np


def test_get_probability():
    table = Table(2)
    table._set_cell(0, 0, 3)
    table._set_cell(0, 1, 4)
    table._set_cell(1, 0, 2)
    table._set_cell(1, 1, 1)

    d = {}
    d[0] = [1, 2, 3, 4]
    d[1] = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    stamps = Timestamps(d)

    nb = np.array([3, 11], dtype='uint64')
    sampler = CollapsedGibbsSampler(BaseSampler(table, stamps, nb, 0.1, 0), 2)
    assert_equal(0.911764705882353, sampler._get_probability(0))
    assert_equal(0.3596491228070175, sampler._get_probability(1))

    sampler._set_current_process(1)
    assert_equal(0.525, sampler._get_probability(0))
    assert_equal(0.09166666666666667, sampler._get_probability(1))


def test_inc_dec():
    table = Table(2)
    table._set_cell(0, 0, 3)
    table._set_cell(0, 1, 4)
    table._set_cell(1, 0, 2)
    table._set_cell(1, 1, 1)

    d = {}
    d[0] = [1, 2, 3, 4]
    d[1] = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    stamps = Timestamps(d)

    nb = np.array([3, 11], dtype='uint64')
    sampler = CollapsedGibbsSampler(BaseSampler(table, stamps, nb, 0.1, 0), 2)
    assert_equal(0.911764705882353, sampler._get_probability(0))
    assert_equal(0.3596491228070175, sampler._get_probability(1))

    sampler._inc_one(0)
    assert_equal(0.9318181818181817, sampler._get_probability(0))
    assert_equal(0.3596491228070175, sampler._get_probability(1))
    sampler._dec_one(0)
    assert_equal(0.911764705882353, sampler._get_probability(0))
