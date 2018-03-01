# -*- coding: utf8
# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: wraparound=False


from gb.stamps cimport Timestamps

from gb.collections.table cimport Table
from gb.collections.fptree cimport FPTree

from libc.stdint cimport uint64_t


cdef class FenwickSampler(object):
    cdef size_t n_proc
    cdef double alpha_prior
    cdef uint64_t[::1] denominators
    cdef size_t current_process
    cdef size_t current_process_size

    cdef FPTree tree
    cdef Timestamps timestamps
    cdef Table joint_counts

    cdef void update_denominators(self, uint64_t[::1] denominators) nogil
    cdef void set_current_process(self, size_t a) nogil
    cdef double get_probability(self, size_t b) nogil
    cdef void inc_one(self, size_t b) nogil
    cdef void dec_one(self, size_t b) nogil
    cdef size_t sample(self) nogil
