# -*- coding: utf8
# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: language_level=3
# cython: wraparound=False


ctypedef struct vector:
    double *data
    size_t size
    size_t capacity

cdef class IntToVector(object):
    cdef size_t n_proc
    cdef vector *vectors

    cdef void push_back(self, size_t i, double value) nogil
    cdef void reset(self) nogil
    cdef size_t get_size(self, size_t i) nogil
    cdef double *get_values(self, size_t i) nogil
