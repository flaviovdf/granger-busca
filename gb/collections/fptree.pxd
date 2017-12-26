# -*- coding: utf8
# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: wraparound=False

from libcpp.vector cimport vector

cdef class FPTree:
    cdef int size
    cdef vector[double] values # values[0] == T in the paper

    cdef void reset(self) nogil
    cdef void _build(self, int size) nogil
    cdef void set_value(self, int i, double val) nogil
    cdef double get_value(self, int i) nogil
    cdef int sample(self, double urnd) nogil
    cdef double get_total(self) nogil
