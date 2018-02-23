# -*- coding: utf8
# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: wraparound=False

import numpy as np

cdef class BitSet(object):

    def __init__(self, int size):
        self.size = size
        self.num_set = 0
        self.max = -1
        cdef int data_size = 1 + self.size // 8
        self.data = np.zeros(data_size, dtype='i8')

    cdef void add(self, int i) nogil:
        if self.get(i) == 0:
            self.data[i / 8] |= (1 << (i % 8))
            self.num_set += 1
            self.max = max(i, self.max)

    cdef void remove(self, int i) nogil:
        if self.get(i) != 0:
            self.data[i / 8] &= ~(1 << (i % 8))
            self.num_set -= 1

    cdef int get(self, int i) nogil:
        return self.data[i / 8] & (1 << (i % 8))
