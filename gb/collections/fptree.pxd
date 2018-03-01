# -*- coding: utf8
# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: wraparound=False

from libcpp.vector cimport vector

cdef class FPTree:
    cdef size_t size
    cdef vector[double] values # values[0] == T in the paper

    cdef void reset(self) nogil
    cdef void _build(self, size_t size) nogil
    cdef void set_value(self, size_t i, double val) nogil
    cdef double get_value(self, size_t i) nogil
    cdef size_t sample(self, double urnd) nogil
    cdef double get_total(self) nogil
