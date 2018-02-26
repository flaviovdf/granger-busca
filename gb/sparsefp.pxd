# -*- coding: utf8
# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: wraparound=False


from gb.collections.bitset cimport BitSet
from gb.collections.fptree cimport FPTree

from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector


cdef double COMPRESS_FACTOR = 0.3


cdef class SparseFenwickSampler(object):
    cdef int n
    cdef double alpha_prior
    cdef FPTree tree
    cdef BitSet bit_set
    cdef int non_zero
    cdef int load
    cdef vector[int] counts
    cdef int[::1] current_denominators
    cdef unordered_map[int, int] non_zero_idx
    cdef unordered_map[int, int] reverse_idx
    cdef int needs_resize

    cdef void renormalize(self, int[::1] denominators) nogil
    cdef double get_probability(self, int i) nogil
    cdef void inc_one(self, int i) nogil
    cdef void dec_one(self, int i) nogil
    cdef int sample(self) nogil
