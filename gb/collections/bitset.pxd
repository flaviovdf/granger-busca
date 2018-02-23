# -*- coding: utf8
# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: wraparound=False

cdef class BitSet(object):
    cdef int size
    cdef int num_set
    cdef int max
    cdef int8[::1] data

    cdef void add(self, int i) nogil
    cdef void remove(self, int i) nogil
    cdef int get(self, int i) nogil
