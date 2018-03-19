# -*- coding: utf8
# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: wraparound=False


from gb.collections.inttovector cimport IntToVector
from gb.stamps cimport Timestamps

from libc.stdint cimport uint64_t


cdef class AbstractKernel(object):
    cdef void set_current_process(self, size_t process) nogil
    cdef double background_probability(self, double dt) nogil
    cdef double cross_rate(self, size_t i, size_t b, double alpha_ab) nogil
    cdef double[::1] get_beta_rates(self) nogil
    cdef double[::1] get_mu_rates(self) nogil


cdef class PoissonKernel(AbstractKernel):
    cdef Timestamps timestamps
    cdef size_t current_process
    cdef double[::1] mu


cdef class BuscaKernel(AbstractKernel):
    cdef PoissonKernel poisson
    cdef double[::1] beta_rates
    cdef IntToVector dts

cdef class TruncatedHawkesKernel(BuscaKernel):
    pass
