# -*- coding: utf8
# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: wraparound=False


from libcpp.vector cimport vector


ctypedef struct entry_t:
    size_t key
    size_t value
    size_t dib


cdef class RobinHoodHash(object):
    cdef size_t inserted
    cdef size_t n_to_prime
    cdef double load_factor
    cdef vector[entry_t] data

    cdef void _resize(self, size_t new_size) nogil
    cdef void set(self, size_t key, size_t delta) nogil
    cdef size_t get(self, size_t key) nogil
    cdef size_t size(self) nogil
