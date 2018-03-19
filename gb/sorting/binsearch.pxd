# -*- coding: utf8
# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: wraparound=False

cdef size_t searchsorted(double *array, size_t n, double value,
                         size_t lower) nogil
