# -*- coding: utf8
# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: wraparound=False


from gb.collections.inttovector cimport IntToVector
from gb.sorting.largest cimport quick_median

from libc.stdio cimport printf
from libc.stdint cimport uint64_t
from libc.stdlib cimport abort

import numpy as np


cdef extern from 'math.h':
    double exp(double) nogil


cdef inline double update_beta_rate(size_t proc_a, Timestamps all_stamps,
                                    double[::1] beta_rates,
                                    IntToVector dts) nogil:

    cdef size_t n_proc = all_stamps.num_proc()
    dts.reset()

    cdef double ti, tp
    cdef double max_ti = 0
    cdef size_t n = all_stamps.get_size(proc_a)
    cdef size_t *state_a
    cdef double *stamps_a
    all_stamps.get_stamps(proc_a, &stamps_a)
    all_stamps.get_causes(proc_a, &state_a)
    cdef size_t proc_b, i
    for i in range(n):
        ti = stamps_a[i]
        proc_b = state_a[i]
        if ti > max_ti:
            max_ti = ti
        if proc_b != n_proc:
            tp = all_stamps.find_previous(proc_b, ti)
            dts.push_back(proc_b, ti - tp)

    for proc_b in range(n_proc):
        if dts.get_size(proc_b) >= 1:
            beta_rates[proc_b] = quick_median(dts.get_values(proc_b),
                                              dts.get_size(proc_b))
        else:
            beta_rates[proc_b] = max_ti


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
    cdef double[::1] get_mu_rates(self) nogil:
        printf('[gb.kernels] Do not use the AbstractKernel\n')
        abort()
    cdef double[::1] get_beta_rates(self) nogil:
        printf('[gb.kernels] Do not use the AbstractKernel\n')
        abort()


cdef class PoissonKernel(AbstractKernel):

    def __init__(self, Timestamps timestamps, size_t n_proc):
        self.timestamps = timestamps
        self.mu = np.zeros(n_proc, dtype='d')

    cdef void set_current_process(self, size_t proc) nogil:
        self.current_process = proc

        cdef size_t n_proc = self.timestamps.num_proc()
        cdef size_t n = self.timestamps.get_size(proc)
        cdef size_t *causes
        self.timestamps.get_causes(proc, &causes)

        cdef size_t i
        cdef double count_background = 0
        for i in range(n):
            if causes[i] == n_proc:
                count_background += 1

        cdef double T = self.timestamps.get_stamp(proc, n-1)
        cdef double rate
        if T == 0:
            rate = 0
        else:
            rate = (<double>count_background) / T
        self.mu[proc] = rate

    cdef double background_probability(self, double dt) nogil:
        cdef double mu_rate = self.mu[self.current_process]
        return mu_rate * dt * exp(-mu_rate * dt)

    cdef double cross_rate(self, size_t i, size_t b, double alpha_ab) nogil:
        return 0.0

    cdef double[::1] get_mu_rates(self) nogil:
        return self.mu


cdef class BuscaKernel(AbstractKernel):

    def __init__(self, PoissonKernel poisson, size_t n_proc):
        self.poisson = poisson
        self.beta_rates = np.zeros(n_proc, dtype='d')
        self.dts = IntToVector(n_proc, 100)

    cdef void set_current_process(self, size_t proc) nogil:
        self.poisson.set_current_process(proc)
        update_beta_rate(proc, self.poisson.timestamps, self.beta_rates,
                         self.dts)

    cdef double background_probability(self, double dt) nogil:
        return self.poisson.background_probability(dt)

    cdef double cross_rate(self, size_t i, size_t b, double alpha_ab) nogil:
        cdef double E = 2.718281828459045
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

        cdef double rate = alpha_ab / (self.beta_rates[b]/E + tp - tpp)
        return rate

    cdef double[::1] get_beta_rates(self) nogil:
        return self.beta_rates


cdef class TruncatedHawkesKernel(BuscaKernel):

    cdef double cross_rate(self, size_t i, size_t b, double alpha_ab) nogil:
        cdef double E = 2.718281828459045
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

        cdef double rate = alpha_ab / (self.beta_rates[b]/E + t - tp)
        return rate
