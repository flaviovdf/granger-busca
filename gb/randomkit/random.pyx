# -*- coding: utf8
# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: wraparound=False

import os

cdef extern from 'randomkit.h':
    ctypedef struct rk_state:
        unsigned long key[624]
        int pos
        int has_gauss
        double gauss


cdef extern from 'randomkit.h':
    cdef void rk_seed(unsigned long seed, rk_state *state) nogil
    cdef double rk_double(rk_state *state) nogil


cdef extern from 'distributions.h':
    cdef double rk_gamma(rk_state *state, double shape, double scale) nogil


cdef extern from 'stdlib.h':
    cdef void *malloc(size_t) nogil
    cdef void free(void *) nogil


cdef class RNG:

    cdef rk_state *rng_state

    def __cinit__(self):
        self.rng_state = <rk_state *> malloc(sizeof(rk_state))
        if self.rng_state == NULL:
            raise MemoryError()

        cdef unsigned long *seedptr
        cdef object seed = os.urandom(sizeof(unsigned long))
        seedptr = <unsigned long *>(<void *>(<char *> seed))
        self.set_seed(seedptr[0])

    def __dealloc__(self):
        if self.rng_state != NULL:
            free(self.rng_state)
            self.rng_state = NULL

    cdef void set_seed(self, unsigned long seed) nogil:
        rk_seed(seed, self.rng_state)

    cdef double rand(self) nogil:
        return rk_double(self.rng_state)

    cdef double gamma(self, double scale, double shape) nogil:
        return rk_gamma(self.rng_state, scale, shape)

cdef RNG _global_rng = RNG()


cdef void set_seed(unsigned long seed) nogil:
    _global_rng.set_seed(seed)


cdef double rand() nogil:
    return _global_rng.rand()


cdef double gamma(double scale, double shape) nogil:
    return _global_rng.gamma(scale, shape)
