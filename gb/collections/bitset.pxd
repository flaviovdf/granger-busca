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

    cdef void add(self, size_t i) nogil
    cdef void remove(self, size_t i) nogil
    cdef int get(self, size_t i) nogil
