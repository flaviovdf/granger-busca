# -*- coding: utf8
# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: wraparound=False


import numpy as np


cdef class BitSet(object):

    def __init__(self, size_t size):
        cdef size_t data_size = 1 + ((size-1) // 64)
        self.size = size
        self.data = np.zeros(data_size, dtype='uint64')

    cdef void add(self, size_t i) nogil:
        self.data[i >> 6] |= ((<uint64_t>1) << (i & 63))

    def _add(self, size_t i):
        self.add(i)

    cdef void remove(self, size_t i) nogil:
        self.data[i >> 6] &= ~((<uint64_t>1) << (i & 63))

    def _remove(self, size_t i):
        self.remove(i)

    cdef int get(self, size_t i) nogil:
        return (self.data[i >> 6] & ((<uint64_t>1) << (i & 63))) > 0

    def _get(self, size_t i):
        return self.get(i)
