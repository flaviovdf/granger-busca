# -*- coding: utf8
# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: wraparound=False


from libc.stdint cimport uint64_t


cdef class BitSet(object):
    cdef int size
    cdef uint64_t[::1] data

    cdef void add(self, int i) nogil
    cdef void remove(self, int i) nogil
    cdef int get(self, int i) nogil
