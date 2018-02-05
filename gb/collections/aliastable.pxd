# -*- coding: utf8
# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: wraparound=False

from libcpp.vector cimport vector

cdef class AliasTable:
    cdef int size
    cdef vector[int] alias
    cdef vector[double] prob
    cdef vector[double] P
    cdef vector[int] L
    cdef vector[int] S

    cdef int sample(self, double urnd1, double urnd2) nogil
