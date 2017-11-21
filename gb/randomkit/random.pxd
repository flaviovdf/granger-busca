# -*- coding: utf8
# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: wraparound=False

cdef void set_seed(unsigned long seed) nogil
cdef double rand() nogil
cdef double gamma(double shape, double scale) nogil
