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

    cdef void set_current_process(self, size_t a) nogil:
        self.current_process = a
        self.current_process_size = self.timestamps.get_stamps(a).shape[0]
        self.tree.reset()

        cdef size_t b
        for b in range(self.n_proc):
            self.tree.set_value(b, self.get_probability(b))

    cdef double get_probability(self, size_t b) nogil:
        cdef size_t a = self.current_process
        cdef uint64_t joint_count = self.joint_counts.get_cell(a, b)
        return dirmulti_posterior(joint_count, self.denominators[b],
                                  self.current_process_size, self.alpha_prior)

    cdef void inc_one(self, size_t b) nogil:
        cdef size_t a = self.current_process
        cdef uint64_t joint_count = self.joint_counts.get_cell(a, b)
        cdef double old_prob = self.tree.get_value(b)
        cdef double new_prob = old_prob + inc(joint_count,
                                              self.denominators[b],
                                              self.current_process_size,
                                              self.alpha_prior, 1)

        self.denominators[b] += 1
        self.joint_counts.set_cell(a, b, joint_count + 1)
        self.tree.set_value(b, new_prob)

    cdef void dec_one(self, size_t b) nogil:
        cdef size_t a = self.current_process
        cdef uint64_t joint_count = self.joint_counts.get_cell(a, b)
        cdef double old_prob = self.tree.get_value(b)
        cdef double new_prob = old_prob + inc(joint_count,
                                              self.denominators[b],
                                              self.current_process_size,
                                              self.alpha_prior, -1)

        self.denominators[b] -= 1
        self.joint_counts.set_cell(a, b, joint_count - 1)
        self.tree.set_value(b, new_prob)

    cdef size_t sample(self) nogil:
        return self.tree.sample(rand())
