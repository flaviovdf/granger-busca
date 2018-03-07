# -*- coding: utf8
# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: wraparound=False

cdef double k_largest(double[::1] array, size_t k) nogil

cdef double quick_median(double[::1] array) nogil
