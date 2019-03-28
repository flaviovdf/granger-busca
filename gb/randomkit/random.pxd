# -*- coding: utf8
# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: language_level=3
# cython: wraparound=False


cdef extern from 'randomkit.h':
    ctypedef struct rk_state:
        unsigned long key[624]
        int pos
        int has_gauss
        double gauss


cdef class RNG(object):
    cdef rk_state *rng_state

    cdef void set_seed(self, unsigned long seed) nogil
    cdef double rand(self) nogil
    cdef double gamma(self, double scale, double shape) nogil
