# -*- coding: utf8
# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: wraparound=False


import numpy as np

cdef class SloppyCounter(object):

    def __init__(self, size_t n_workers, size_t sloppy_level,
                 uint64_t[::1] global_counts):
        self.sloppy_level = sloppy_level
        cdef size_t n_proc = global_counts.shape[0]
        self.last_seen = np.zeros(shape=(n_workers, n_proc), dtype='uint64')
        self.global_counts = global_counts
        self.updates = np.zeros(n_workers, dtype='uint64')

    cdef void update_counts(self, size_t worker, uint64_t[::1] update) nogil:
        cdef size_t i
        self.updates[worker] += 1
        if self.updates[worker] == self.sloppy_level:
            with gil:
                for i in range(<size_t>update.shape[0]):
                    self.global_counts[i] -= self.last_seen[worker, i]
                    self.global_counts[i] += update[i]
                    update[i] = self.global_counts[i]
            for i in range(<size_t>update.shape[0]):
                self.last_seen[worker, i] = update[i]
            self.updates[worker] = 0
