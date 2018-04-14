# -*- coding: utf8

from gb.randomkit.random import RNG

from gb.samplers import BaseSampler
from gb.samplers import FenwickSampler
from gb.stamps import Timestamps
from gb.sloppy import SloppyCounter

from numpy.testing import assert_equal

import numpy as np


def test_get_probability():
    d = {}
    d[0] = [1, 2, 3, 4, 5, 6, 7]
    d[1] = [11, 12, 13]
    stamps = Timestamps(d)
    causes = stamps._get_causes(0)
    causes[0] = 0
    causes[1] = 0
    causes[2] = 0
    causes[3] = 1
    causes[4] = 1
    causes[5] = 1
    causes[6] = 1

    causes = stamps._get_causes(1)
    causes[0] = 0
    causes[1] = 0
    causes[2] = 1

    nb = np.array([5, 5], dtype='uint64')
    init_state = np.array([[5, 5]], dtype='uint64')
    id_ = 0
    sloppy = SloppyCounter(1, 9999, nb, init_state)
    sampler = FenwickSampler(BaseSampler(stamps, sloppy, id_, 0.1, RNG()), 2)
    sampler._set_current_process(0)
    assert_equal(0.5961538461538461, sampler._get_probability(0))
    assert_equal(0.7884615384615383, sampler._get_probability(1))

    sampler._set_current_process(1)
    assert_equal(0.40384615384615385, sampler._get_probability(0))
    assert_equal(0.21153846153846154, sampler._get_probability(1))


def test_inc_dec():
    d = {}
    d[0] = [1, 2, 3, 4, 5, 6, 7]
    d[1] = [11, 12, 13]
    stamps = Timestamps(d)
    causes = stamps._get_causes(0)
    causes[0] = 0
    causes[1] = 0
    causes[2] = 0
    causes[3] = 1
    causes[4] = 1
    causes[5] = 1
    causes[6] = 1

    causes = stamps._get_causes(1)
    causes[0] = 0
    causes[1] = 0
    causes[2] = 1

    nb = np.array([5, 5], dtype='uint64')
    init_state = np.array([[5, 5]], dtype='uint64')
    id_ = 0
    sloppy = SloppyCounter(1, 9999, nb, init_state)
    sampler = FenwickSampler(BaseSampler(stamps, sloppy, id_, 0.1, RNG()), 2)
    sampler._set_current_process(0)
    assert_equal(0.5961538461538461, sampler._get_probability(0))
    assert_equal(0.7884615384615383, sampler._get_probability(1))

    sampler._inc_one(0)
    assert_equal(0.6612903225806451, sampler._get_probability(0))
    assert_equal(0.7884615384615383, sampler._get_probability(1))
    sampler._dec_one(0)
    assert_equal(0.5961538461538461, sampler._get_probability(0))
