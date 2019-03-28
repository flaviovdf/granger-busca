# -*- coding: utf8

from gb.bvls.solve import _bvls

from numpy.testing import assert_almost_equal
from numpy.testing import assert_equal

import numpy as np


def test_bvls():
    lb = np.array([1.0, 3.0], order='C', dtype='d')
    ub = np.array([2.0, 4.0], order='C', dtype='d')

    A = np.zeros(shape=(2, 2), order='C', dtype='d')
    A[0, 0] = 0.965915
    A[0, 1] = 0.747928
    A[1, 0] = 0.367391
    A[1, 1] = 0.480637

    b = np.array([0.997560, 0.566825], order='C', dtype='d')
    result = np.zeros(2, order='C', dtype='d')
    status = _bvls(A, b, lb, ub, result)

    assert_equal(0, status)
    assert_almost_equal(result[0], 1.00000)
    assert_almost_equal(result[1], 3.00000)
