# -*- coding: utf8
# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: wraparound=False


include 'dirichlet.pxi'

from gb.sorting.binsearch cimport searchsorted

from libc.stdio cimport printf
from libc.stdlib cimport abort

import numpy as np


cdef class AbstractSampler(object):
    cdef void set_current_process(self, size_t a) nogil:
        printf('[gb.samplers] Do not use the BaseSampler or AbstractSampler\n')
        abort()
    cdef double get_probability(self, size_t b) nogil:
        printf('[gb.samplers] Do not use the BaseSampler or AbstractSampler\n')
        abort()
    cdef void inc_one(self, size_t b) nogil:
        printf('[gb.samplers] Do not use the BaseSampler or AbstractSampler\n')
        abort()
    cdef void dec_one(self, size_t b) nogil:
        printf('[gb.samplers] Do not use the BaseSampler or AbstractSampler\n')
        abort()
    cdef size_t sample_for_idx(self, size_t i, AbstractKernel kernel) nogil:
        printf('[gb.samplers] Do not use the BaseSampler or AbstractSampler\n')
        abort()
    cdef uint64_t is_background(self, AbstractKernel kernel, double dt) nogil:
        printf('[gb.samplers] Do not use the BaseSampler or AbstractSampler\n')
        abort()


cdef class BaseSampler(AbstractSampler):

    def __init__(self, Timestamps timestamps, SloppyCounter sloppy,
                 size_t worker_id, double alpha_prior):
        self.n_proc = timestamps.num_proc()
        self.alpha_prior = alpha_prior
        self.sloppy = sloppy
        self.worker_id = worker_id
        self.timestamps = timestamps
        self.nab = np.zeros(self.n_proc, dtype='uint64')
        self.rng = RNG()

    cdef void set_current_process(self, size_t a) nogil:
        self.current_process = a
        self.current_process_size = self.timestamps.get_size(a)

        cdef size_t b, i
        for b in range(self.timestamps.num_proc()):
            self.nab[b] = 0
        for i in range(self.current_process_size):
            b = self.timestamps.get_cause(a, i)
            if b != self.timestamps.num_proc():
                self.nab[b] += 1
        self.sloppy.get_local_counts(self.worker_id, &self.denominators)

    cdef double get_probability(self, size_t b) nogil:
        cdef size_t a = self.current_process
        return dirmulti_posterior(self.nab[b], self.denominators[b],
                                  self.current_process_size, self.alpha_prior)

    cdef void inc_one(self, size_t b) nogil:
        cdef size_t a = self.current_process
        self.denominators[b] += 1
        self.nab[b] += 1
        self.sloppy.inc_one(self.worker_id, b)

    cdef void dec_one(self, size_t b) nogil:
        cdef size_t a = self.current_process
        self.denominators[b] -= 1
        self.nab[b] -= 1
        self.sloppy.dec_one(self.worker_id, b)

    cdef uint64_t is_background(self, AbstractKernel kernel, double dt) nogil:
        return self.rng.rand() < kernel.background_probability(dt)


cdef class FenwickSampler(AbstractSampler):

    def __init__(self, BaseSampler base, size_t n_proc):
        self.base = base
        self.tree = FPTree(n_proc)
        self.n_proc = n_proc

    cdef void set_current_process(self, size_t a) nogil:
        self.base.set_current_process(a)

        self.tree.reset()
        cdef size_t b
        for b in range(self.base.n_proc):
            self.tree.set_value(b, self.get_probability(b))

    def _set_current_process(self, size_t a):
        return self.set_current_process(a)

    cdef double get_probability(self, size_t b) nogil:
        return self.base.get_probability(b)

    def _get_probability(self, size_t b):
        return self.get_probability(b)

    cdef void inc_one(self, size_t b) nogil:
        self.base.inc_one(b)
        self.tree.set_value(b, self.get_probability(b))

    def _inc_one(self, size_t b):
        return self.inc_one(b)

    cdef void dec_one(self, size_t b) nogil:
        self.base.dec_one(b)
        self.tree.set_value(b, self.get_probability(b))

    def _dec_one(self, size_t b):
        return self.dec_one(b)

    cdef size_t sample_for_idx(self, size_t i, AbstractKernel kernel) nogil:
        cdef size_t proc_a = self.base.current_process
        cdef size_t candidate = self.tree.sample(self.base.rng.rand() * \
                                                 self.tree.get_total())

        cdef size_t proc_b = self.base.timestamps.get_cause(proc_a, i)
        if proc_b == self.n_proc:
            return candidate

        cdef double alpha_ba = self.get_probability(proc_b)
        cdef double alpha_ca = self.get_probability(candidate)

        cdef double p_b = kernel.cross_rate(i, proc_b, alpha_ba)
        cdef double p_c = kernel.cross_rate(i, candidate, alpha_ca)

        cdef int choice
        if self.base.rng.rand() < min(1, (p_c * alpha_ba) / (p_b * alpha_ca)):
            choice = candidate
        else:
            choice = proc_b
        return choice

    cdef uint64_t is_background(self, AbstractKernel kernel, double dt) nogil:
        return self.base.is_background(kernel, dt)

cdef class CollapsedGibbsSampler(AbstractSampler):

    def __init__(self, BaseSampler base, size_t n_proc):
        self.base = base
        self.buffer = np.zeros(n_proc, dtype='d')

    cdef void set_current_process(self, size_t a) nogil:
        self.base.set_current_process(a)

    def _set_current_process(self, size_t a):
        return self.set_current_process(a)

    cdef double get_probability(self, size_t b) nogil:
        return self.base.get_probability(b)

    def _get_probability(self, size_t b):
        return self.get_probability(b)

    cdef void inc_one(self, size_t b) nogil:
        self.base.inc_one(b)

    def _inc_one(self, size_t b):
        return self.inc_one(b)

    cdef void dec_one(self, size_t b) nogil:
        self.base.dec_one(b)

    def _dec_one(self, size_t b):
        return self.dec_one(b)

    cdef size_t sample_for_idx(self, size_t i, AbstractKernel kernel) nogil:
        cdef size_t n_proc = self.buffer.shape[0]
        cdef size_t b
        cdef double alpha_ba
        for b in range(n_proc):
            alpha_ba = self.get_probability(b)
            self.buffer[b] = kernel.cross_rate(i, b, alpha_ba)
            if b > 0:
                self.buffer[b] += self.buffer[b-1]
        return searchsorted(&self.buffer[0], self.buffer.shape[0],
                            self.base.rng.rand() * self.buffer[n_proc-1], 0)

    cdef uint64_t is_background(self, AbstractKernel kernel, double dt) nogil:
        return self.base.is_background(kernel, dt)
