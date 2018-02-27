# -*- coding: utf8
# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: wraparound=False


from libc.stdint cimport uint32_t
from libc.stdint cimport uint64_t
from libc.stdint cimport UINT32_MAX

from libcpp.vector cimport vector


cdef class RobinHoodHash(object):
    cdef int size
    cdef int capacity
    cdef int initial_capacity
    cdef vector[uint64_t] data

    cdef void insert(self, uint32_t key, uint32_t value) nogil
    cdef void remove(self, uint32_t key, uint32_t value) nogil
    cdef int size(self) nogil
