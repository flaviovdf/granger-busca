# -*- coding: utf8

from gb.sorting.largest import _median

import numpy as np


def test_median():
    x = np.array([1.0])
    assert 1.0 == _median(x)

    x = np.array([1.0, 2.0, 3.0])
    assert 2.0 == _median(x)

    x = np.array([1.0, 2.0, 3.0, 4.0])
    print(_median(x))
    assert 2.5 == _median(x)

    for i in range(100):
        x = np.random.random(size=101)
        print(np.median(x), _median(x))
        assert np.median(x) == _median(x)

        x = np.random.random(size=100)
        print(np.median(x), _median(x))
        assert np.median(x) == _median(x)
