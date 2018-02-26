# -*- coding: utf8
# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: wraparound=False


from gb.collections cimport BitSet
from gb.collections cimport FPTree

from libcpp.unordered_map cimport unordered_map


cdef double COMPRESS_FACTOR = 0.3


cdef class SparseFenwickSampler(object):
    cdef int n
    cdef double alpha_prior
    cdef FPTree tree
    cdef BitSet bit_set
    cdef int non_zero
    cdef int load
    cdef int[::1] counts
    cdef double[::1] current_denominators
    cdef unordered_map[int, int] non_zero_idx
    cdef unordered_map[int, int] reverse_idx

    cdef void renormalize(self, double[::1] denominators) nogil
    cdef double get_probability(self, int i) nogil
    cdef void inc_one(self, int i) nogil
    cdef void dec_one(self, int i) nogil
    cdef int get_count(self, int i) nogil
    cdef int sample(self) nogil
    cdef void force_compress(self) nogil
