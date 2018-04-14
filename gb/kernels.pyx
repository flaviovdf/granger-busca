# -*- coding: utf8
# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: wraparound=False


from libc.stdio cimport printf
from libc.stdint cimport uint64_t
from libc.stdlib cimport abort

import numpy as np


cdef extern from 'math.h':
    double exp(double) nogil


cdef class AbstractKernel(object):
    cdef void set_current_process(self, size_t process) nogil:
        printf('[gb.kernels] Do not use the AbstractKernel\n')
        abort()
    cdef double background_probability(self, double dt) nogil:
        printf('[gb.kernels] Do not use the AbstractKernel\n')
        abort()
    cdef double cross_rate(self, size_t i, size_t b, double alpha_ab) nogil:
        printf('[gb.kernels] Do not use the AbstractKernel\n')
        abort()
    cdef double mu_rate(self, size_t process) nogil:
        printf('[gb.kernels] Do not use the AbstractKernel\n')
        abort()
    cdef double[::1] get_mu_rates(self) nogil:
        printf('[gb.kernels] Do not use the AbstractKernel\n')
        abort()


cdef class PoissonKernel(AbstractKernel):

    def __init__(self, Timestamps timestamps, size_t n_proc, RNG rng):
        self.timestamps = timestamps
        self.mu = np.zeros(n_proc, dtype='d')
        self.rng = rng

    cdef void set_current_process(self, size_t proc) nogil:
        self.current_process = proc

        cdef size_t n_proc = self.timestamps.num_proc()
        cdef size_t n = self.timestamps.get_size(proc)
        cdef size_t *causes
        self.timestamps.get_causes(proc, &causes)

        cdef double *stamps
        self.timestamps.get_stamps(proc, &stamps)

        cdef size_t i
        cdef double count_background = 0
        cdef double prev_poisson = -1
        cdef double dts = 0.0
        for i in range(n):
            if causes[i] == n_proc:
                if prev_poisson != -1:
                    count_background += 1
                    dts += stamps[i] - prev_poisson
                prev_poisson = stamps[i]

        if dts > 0:
            self.mu[proc] = count_background / dts
        else:
            self.mu[proc] = 0

    cdef double background_probability(self, double dt) nogil:
        cdef double mu_rate = self.mu[self.current_process]
        return mu_rate * dt * exp(-mu_rate * dt)

    cdef double mu_rate(self, size_t process) nogil:
        return self.mu[process]

    cdef double cross_rate(self, size_t i, size_t b, double alpha_ab) nogil:
        return 0.0

    cdef double[::1] get_mu_rates(self) nogil:
        return self.mu


cdef class WoldKernel(AbstractKernel):

    def __init__(self, PoissonKernel poisson, double[::1] beta):
        self.poisson = poisson
        self.beta = beta

    cdef void set_current_process(self, size_t proc) nogil:
        self.poisson.set_current_process(proc)

    cdef double background_probability(self, double dt) nogil:
        return self.poisson.background_probability(dt)

    cdef double mu_rate(self, size_t process) nogil:
        return self.poisson.mu_rate(process)

    cdef double cross_rate(self, size_t i, size_t b, double alpha_ab) nogil:
        cdef size_t a = self.poisson.current_process

        cdef double *stamps
        self.poisson.timestamps.get_stamps(a, &stamps)

        cdef double t = stamps[i]
        cdef double tp
        cdef double tpp
        if i > 0:
            tp = stamps[i-1]
        else:
            tp = 0
        if tp != 0:
            if a == b:
                if i > 1:
                    tpp = stamps[i-2]
                else:
                    tpp = 0
            else:
                tpp = self.poisson.timestamps.find_previous(b, tp)
        else:
            tpp = 0

        cdef double rate = alpha_ab / (self.beta[b] + tp - tpp)
        return rate


cdef class TruncatedHawkesKernel(WoldKernel):

    cdef double cross_rate(self, size_t i, size_t b, double alpha_ab) nogil:
        cdef size_t a = self.poisson.current_process
        cdef double *stamps
        self.poisson.timestamps.get_stamps(a, &stamps)
        cdef double t = stamps[i]
        cdef double tp
        if a == b:
            if i > 1:
                tp = stamps[i-2]
            else:
                tp = 0
        else:
            tp = self.poisson.timestamps.find_previous(b, t)

        cdef double rate = alpha_ab / (self.beta[b] + t - tp)
        return rate
