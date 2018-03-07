# -*- coding: utf8
# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: wraparound=False


from gb.collections.table cimport Table
from gb.collections.fptree cimport FPTree

from gb.stamps cimport Timestamps

from libc.stdint cimport uint64_t


cdef class AbstractSampler(object):
    cdef void update_denominators(self, uint64_t[::1] denominators) nogil
    cdef void set_current_process(self, size_t a) nogil
    cdef double get_probability(self, size_t b) nogil
    cdef void inc_one(self, size_t b) nogil
    cdef void dec_one(self, size_t b) nogil
    cdef size_t sample_for_idx(self, size_t i, double[::1] beta_rates) nogil


cdef class BaseSampler(AbstractSampler):
    cdef size_t n_proc
    cdef double alpha_prior
    cdef uint64_t[::1] denominators
    cdef size_t current_process
    cdef size_t current_process_size

    cdef FPTree tree
    cdef Timestamps timestamps
    cdef Table joint_counts


cdef class FenwickSampler(AbstractSampler):
    cdef BaseSampler base
    cdef FPTree tree

cdef class CollapsedGibbsSampler(AbstractSampler):
    cdef BaseSampler base
    cdef double[::1] buffer
