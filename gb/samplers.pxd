# -*- coding: utf8
# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: language_level=3
# cython: wraparound=False


from gb.collections.fptree cimport FPTree

from gb.kernels cimport AbstractKernel
from gb.randomkit.random cimport RNG
from gb.sloppy cimport SloppyCounter
from gb.stamps cimport Timestamps

from libc.stdint cimport uint64_t


cdef class AbstractSampler(object):
    cdef void set_current_process(self, size_t a) nogil
    cdef double get_probability(self, size_t b) nogil
    cdef void inc_one(self, size_t b) nogil
    cdef void dec_one(self, size_t b) nogil
    cdef size_t sample_for_idx(self, size_t i, AbstractKernel kernel) nogil
    cdef uint64_t is_background(self, double mu_rate, double dt_mu) nogil
    cdef void get_alphas(self, size_t n, double[::1] alphas) nogil

cdef class BaseSampler(AbstractSampler):
    cdef size_t n_proc
    cdef double alpha_prior
    cdef SloppyCounter sloppy
    cdef uint64_t *denominators
    cdef uint64_t[::1] nab
    cdef size_t current_process
    cdef size_t worker_id
    cdef Timestamps timestamps
    cdef RNG rng

cdef class FenwickSampler(AbstractSampler):
    cdef BaseSampler base
    cdef FPTree tree

cdef class CollapsedGibbsSampler(AbstractSampler):
    cdef BaseSampler base
    cdef double[::1] buffer
