# -*- coding: utf8
# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: wraparound=False


include 'dirichlet.pxi'

from gb.randomkit.random cimport rand


cdef class FenwickSampler(object):

    def __init__(self, Table joint_counts, Timestamps timestamps,
                 uint64_t[::1] denominators, double alpha_prior,
                 size_t initial_process):
        self.n_proc = denominators.shape[0]
        self.alpha_prior = alpha_prior
        self.denominators = denominators
        self.joint_counts = joint_counts
        self.timestamps = timestamps
        self.tree = FPTree(self.n_proc)
        self.set_current_process(initial_process)

    cdef void update_denominators(self, uint64_t[::1] denominators) nogil:
        self.denominators = denominators

    def _update_denominators(self, uint64_t[::1] denominators):
        return self.update_denominators(denominators)

    cdef void set_current_process(self, size_t a) nogil:
        self.current_process = a
        self.current_process_size = self.timestamps.get_stamps(a).shape[0]
        self.tree.reset()

        cdef size_t b
        for b in range(self.n_proc):
            self.tree.set_value(b, self.get_probability(b))

    def _set_current_process(self, size_t a):
        return self.set_current_process(a)

    cdef double get_probability(self, size_t b) nogil:
        cdef size_t a = self.current_process
        cdef uint64_t joint_count = self.joint_counts.get_cell(a, b)
        return dirmulti_posterior(joint_count, self.denominators[b],
                                  self.current_process_size, self.alpha_prior)

    def _get_probability(self, size_t b):
        return self.get_probability(b)

    cdef void inc_one(self, size_t b) nogil:
        cdef size_t a = self.current_process
        cdef uint64_t joint_count = self.joint_counts.get_cell(a, b)
        self.denominators[b] += 1
        self.joint_counts.set_cell(a, b, joint_count + 1)
        self.tree.set_value(b, self.get_probability(b))

    def _inc_one(self, size_t b):
        return self.inc_one(b)

    cdef void dec_one(self, size_t b) nogil:
        cdef size_t a = self.current_process
        cdef uint64_t joint_count = self.joint_counts.get_cell(a, b)
        self.denominators[b] -= 1
        self.joint_counts.set_cell(a, b, joint_count - 1)
        self.tree.set_value(b, self.get_probability(b))

    def _dec_one(self, size_t b):
        return self.dec_one(b)

    cdef size_t sample(self) nogil:
        return self.tree.sample(rand() * self.tree.get_total())

    def _sample(self):
        return self.sample()
