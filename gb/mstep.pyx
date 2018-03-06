# -*- coding: utf8
# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: wraparound=False


import numpy as np


cdef inline void update_mu_rate(size_t proc_a, Timestamps all_stamps,
                                int count_background,
                                double[::1] mu_rates) nogil:

    cdef double[::1] timestamps_proc_a = all_stamps.get_stamps(proc_a)
    cdef double T = timestamps_proc_a[timestamps_proc_a.shape[0]-1]
    cdef double rate
    if T == 0:
        rate = 0
    else:
        rate = count_background / T
    mu_rates[proc_a] = rate


cdef inline void update_beta_rate(size_t proc_a, Timestamps all_stamps,
                                  double[::1] beta_rates,
                                  double[::1] all_deltas) nogil:
    cdef size_t n_proc = beta_rates.shape[0]
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
        beta_rates[proc_a] = quick_median(all_deltas[:n_elements])
    else:
        beta_rates[proc_a] = max_ti


cdef class MStep(object):

    def __init__(self, size_t n_events):
        self.all_stamps_buffer = np.zeros(n_events, dtype='d')

    cdef void update_mu_rates(self, Timestamps all_stamps,
                              uint64_t[::1] num_background,
                              double[::1] mu_rates) nogil:
        cdef size_t n_proc = mu_rates.shape[0]
        cdef size_t proc_a
        for proc_a in range(n_proc):
            update_mu_rate(proc_a, all_stamps, num_background[proc_a],
                           mu_rates)

    cdef void update_beta_rates(self, Timestamps all_stamps,
                                double[::1] beta_rates) nogil:
        cdef size_t n_proc = beta_rates.shape[0]
        cdef size_t proc_a
        for proc_a in range(n_proc):
            update_beta_rate(proc_a, all_stamps, beta_rates,
                             self.all_stamps_buffer)
