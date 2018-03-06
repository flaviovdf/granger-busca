# -*- coding: utf8
# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: wraparound=False


from libc.stdint cimport uint64_t

from gb.sorting.largest cimport quick_median
from gb.stamps cimport Timestamps


cdef class MStep(object):
    cdef double[::1] all_stamps_buffer

    cdef void update_mu_rates(self, Timestamps all_stamps,
                              uint64_t[::1] num_background,
                              double[::1] mu_rates) nogil
    cdef void update_beta_rates(self, Timestamps all_stamps,
                                double[::1] beta_rates) nogil
