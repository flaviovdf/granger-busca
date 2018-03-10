# -*- coding: utf8
# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: wraparound=False


from gb.stamps cimport Timestamps
from gb.sorting.largest cimport quick_median

from libc.stdio cimport printf
from libc.stdint cimport uint64_t
from libc.stdlib cimport abort

import numpy as np


cdef extern from 'math.h':
    double exp(double) nogil


cdef inline double update_beta_rate(size_t proc_a, Timestamps all_stamps,
                                    size_t n_proc, double[::1] all_deltas) nogil:
    cdef size_t proc_b, i
    cdef double ti, tp
    cdef double max_ti = 0
    cdef double[::1] stamps_b
    cdef size_t[::1] state_b
    cdef size_t n_elements = 0
    for proc_b in range(n_proc):
        stamps_b = all_stamps.get_stamps(proc_b)
        state_b = all_stamps.get_causes(proc_b)
        for i in range(<size_t>stamps_b.shape[0]):
            if state_b[i] == proc_a:
                ti = stamps_b[i]
                if ti > max_ti:
                    max_ti = ti
                tp = all_stamps.find_previous(proc_a, ti)
                all_deltas[n_elements] = ti - tp
                n_elements += 1

    if n_elements >= 1:
        return quick_median(all_deltas[:n_elements])
    else:
        return max_ti


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
    cdef void update_mu_rate(self, uint64_t count_background) nogil:
        printf('[gb.kernels] Do not use the AbstractKernel\n')
        abort()
    cdef void update_cross_rates(self) nogil:
        printf('[gb.kernels] Do not use the AbstractKernel\n')
        abort()
    cdef double[::1] get_beta_rates(self) nogil:
        printf('[gb.kernels] Do not use the AbstractKernel\n')
        abort()


cdef class PoissonKernel(AbstractKernel):

    def __init__(self, Timestamps timestamps, size_t n_proc):
        self.timestamps = timestamps
        self.mu = np.zeros(n_proc, dtype='d')

    cdef void set_current_process(self, size_t process) nogil:
        self.current_process = process

    cdef double background_probability(self, double dt) nogil:
        cdef double mu_rate = self.mu[self.current_process]
        return mu_rate * dt * exp(-mu_rate * dt)

    cdef double cross_rate(self, size_t i, size_t b, double alpha_ab) nogil:
        return 0.0

    cdef void update_mu_rate(self, uint64_t count_background) nogil:
        cdef size_t proc_a = self.current_process
        cdef double[::1] timestamps_proc_a = self.timestamps.get_stamps(proc_a)
        cdef double T = timestamps_proc_a[timestamps_proc_a.shape[0]-1]
        cdef double rate
        if T == 0:
            rate = 0
        else:
            rate = (<double>count_background) / T
        self.mu[proc_a] = rate

    cdef void update_cross_rates(self) nogil:
        pass

    cdef double[::1] get_mu_rates(self) nogil:
        return self.mu


cdef class BuscaKernel(AbstractKernel):

    def __init__(self, PoissonKernel poisson, Timestamps timestamps,
                 size_t n_proc, size_t n_events):
        self.poisson = poisson
        self.timestamps = timestamps
        self.beta_rates = np.zeros(n_proc, dtype='d')
        self.all_stamps_buffer = np.zeros(n_events, dtype='d')

    cdef void set_current_process(self, size_t process) nogil:
         self.poisson.set_current_process(process)

    cdef double background_probability(self, double dt) nogil:
        return self.poisson.background_probability(dt)

    cdef double cross_rate(self, size_t i, size_t b, double alpha_ab) nogil:
        cdef double E = 2.718281828459045
        cdef size_t a = self.poisson.current_process
        cdef double[::1] stamps = self.timestamps.get_stamps(a)
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
                tpp = self.timestamps.find_previous(b, tp)
        else:
            tpp = 0

        cdef double rate = alpha_ab / (self.beta_rates[b]/E + tp - tpp)
        return rate

    cdef void update_mu_rate(self, uint64_t count_background) nogil:
        self.poisson.update_mu_rate(count_background)

    cdef void update_cross_rates(self) nogil:
        cdef size_t a = self.poisson.current_process
        self.beta_rates[a] = update_beta_rate(a, self.timestamps,
                                              self.beta_rates.shape[0],
                                              self.all_stamps_buffer)

    cdef double[::1] get_beta_rates(self) nogil:
        return self.beta_rates


cdef class TruncatedHawkesKernel(AbstractKernel):

    def __init__(self, PoissonKernel poisson, Timestamps timestamps,
                 size_t n_proc, size_t n_events):
        self.timestamps = timestamps
        self.poisson = poisson
        self.beta_rates = np.zeros(n_proc, dtype='d')
        self.all_stamps_buffer = np.zeros(n_events, dtype='d')

    cdef void set_current_process(self, size_t process) nogil:
         self.poisson.set_current_process(process)

    cdef double background_probability(self, double dt) nogil:
        return self.poisson.background_probability(dt)

    cdef double cross_rate(self, size_t i, size_t b, double alpha_ab) nogil:
        cdef double E = 2.718281828459045
        cdef size_t a = self.poisson.current_process
        cdef double[::1] stamps = self.timestamps.get_stamps(a)
        cdef double t = stamps[i]
        cdef double tp
        if a == b:
            if i > 1:
                tp = stamps[i-2]
            else:
                tp = 0
        else:
            tp = self.timestamps.find_previous(b, t)

        cdef double rate = alpha_ab / (self.beta_rates[b]/E + t - tp)
        return rate

    cdef void update_mu_rate(self, uint64_t count_background) nogil:
        self.poisson.update_mu_rate(count_background)

    cdef void update_cross_rates(self) nogil:
        cdef size_t a = self.poisson.current_process
        self.beta_rates[a] = update_beta_rate(a, self.timestamps,
                                              self.beta_rates.shape[0],
                                              self.all_stamps_buffer)

    cdef double[::1] get_beta_rates(self) nogil:
        return self.beta_rates
