# -*- coding: utf8
# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: wraparound=False

from gb.collections cimport BitSet
from gb.collections cimport FPTree


cdef double COMPRESS_FACTOR = 0.3


cdef class SparseStateKeeper(object):

    cdef void init(self, double[::1] joint_counts,
                   double[::1] denominators) nogil
    cdef void update(self, vector[double] denominators) nogil
    cdef int get_count(self, int i) nogil
    cdef double get_probability(self, int i) nogil
    cdef void inc_one(self, int i) nogil
    cdef void dec_one(self, int i) nogil
    cdef int sample(self) nogil
