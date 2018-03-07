# -*- coding: utf8
# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: wraparound=False


include 'dirichlet.pxi'

from gb.randomkit.random cimport rand
from gb.sorting.binsearch cimport searchsorted

from libc.stdio cimport printf
from libc.stdlib cimport abort

import numpy as np


cdef class AbstractSampler(object):
    cdef void update_denominators(self, uint64_t[::1] denominators) nogil:
        printf('[gb.samplers] Do not use the BaseSampler or AbstractSampler\n')
        abort()
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
    cdef size_t sample_for_idx(self, size_t i, double[::1] beta_rates) nogil:
        printf('[gb.samplers] Do not use the BaseSampler or AbstractSampler\n')
        abort()


cdef class BaseSampler(AbstractSampler):

    def __init__(self, Table joint_counts, Timestamps timestamps,
                 uint64_t[::1] denominators, double alpha_prior,
                 size_t initial_process):
        self.n_proc = denominators.shape[0]
        self.alpha_prior = alpha_prior
        self.denominators = denominators
        self.joint_counts = joint_counts
        self.timestamps = timestamps
        self.set_current_process(initial_process)

    cdef void update_denominators(self, uint64_t[::1] denominators) nogil:
        self.denominators = denominators

    cdef void set_current_process(self, size_t a) nogil:
        self.current_process = a
        self.current_process_size = self.timestamps.get_stamps(a).shape[0]

    cdef double get_probability(self, size_t b) nogil:
        cdef size_t a = self.current_process
        cdef uint64_t joint_count = self.joint_counts.get_cell(a, b)
        return dirmulti_posterior(joint_count, self.denominators[b],
                                  self.current_process_size, self.alpha_prior)

    cdef void inc_one(self, size_t b) nogil:
        cdef size_t a = self.current_process
        cdef uint64_t joint_count = self.joint_counts.get_cell(a, b)
        self.denominators[b] += 1
        self.joint_counts.set_cell(a, b, joint_count + 1)

    cdef void dec_one(self, size_t b) nogil:
        cdef size_t a = self.current_process
        cdef uint64_t joint_count = self.joint_counts.get_cell(a, b)
        self.denominators[b] -= 1
        self.joint_counts.set_cell(a, b, joint_count - 1)

    cdef size_t sample_for_idx(self, size_t i, double[::1] beta_rates) nogil:
        printf('[gb.samplers] Do not use the BaseSampler or AbstractSampler\n')
        abort()


cdef class FenwickSampler(AbstractSampler):

    def __init__(self, BaseSampler base, size_t n_proc):
        self.base = base
        self.tree = FPTree(n_proc)

    cdef void update_denominators(self, uint64_t[::1] denominators) nogil:
        self.base.update_denominators(denominators)

    def _update_denominators(self, uint64_t[::1] denominators):
        return self.update_denominators(denominators)

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

    cdef size_t sample_for_idx(self, size_t i, double[::1] beta_rates) nogil:
        cdef size_t n_proc = beta_rates.shape[0]
        cdef size_t proc_a = self.base.current_process
        cdef size_t candidate = self.tree.sample(rand()*self.tree.get_total())
        cdef size_t[::1] causes = self.base.timestamps.get_causes(proc_a)
        cdef size_t proc_b = causes[i]

        if proc_b == n_proc:
            return candidate

        cdef double alpha_ba = self.get_probability(proc_b)
        cdef double alpha_ca = self.get_probability(candidate)

        cdef double p_b = busca_probability(i, proc_a, proc_b,
                                            self.base.timestamps, alpha_ba,
                                            beta_rates[proc_b])
        cdef double p_c = busca_probability(i, proc_a, candidate,
                                            self.base.timestamps, alpha_ca,
                                            beta_rates[candidate])
        cdef int choice
        if rand() < min(1, (p_c * alpha_ba) / (p_b * alpha_ca)):
            choice = candidate
        else:
            choice = proc_b
        return choice


cdef class CollapsedGibbsSampler(AbstractSampler):

    def __init__(self, BaseSampler base, size_t n_proc):
        self.base = base
        self.buffer = np.zeros(n_proc, dtype='d')

    cdef void update_denominators(self, uint64_t[::1] denominators) nogil:
        self.base.update_denominators(denominators)

    def _update_denominators(self, uint64_t[::1] denominators):
        return self.update_denominators(denominators)

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

    cdef size_t sample_for_idx(self, size_t i, double[::1] beta_rates) nogil:
        cdef size_t n_proc = beta_rates.shape[0]
        cdef size_t b
        for b in range(<size_t>self.base.denominators.shape[0]):
            self.buffer[b] = busca_probability(i, self.base.current_process, b,
                                               self.base.timestamps,
                                               self.get_probability(b),
                                               beta_rates[b])
            if b > 0:
                self.buffer[b] += self.buffer[b-1]
        return searchsorted(self.buffer, self.buffer[n_proc-1] * rand(), 0)
