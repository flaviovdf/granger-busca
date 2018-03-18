# -*- coding: utf8
# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: wraparound=False


from cpython.pythread cimport PyThread_acquire_lock
from cpython.pythread cimport PyThread_allocate_lock
from cpython.pythread cimport PyThread_free_lock
from cpython.pythread cimport PyThread_release_lock
from cpython.pythread cimport NOWAIT_LOCK

import numpy as np


cdef class SloppyCounter(object):

    def __init__(self, size_t n_workers, size_t sloppy_level,
                 uint64_t[::1] global_counts, uint64_t[:, ::1] init_state):
        cdef size_t n_proc = global_counts.shape[0]
        self.sloppy_level = sloppy_level
        self.local_counts = init_state
        self.global_counts = global_counts
        self.delay = np.zeros(n_workers, dtype='uint64')
        self.updates = np.zeros((n_workers, n_proc), dtype='i')
        self.lock = PyThread_allocate_lock()

    def __dealloc__(self):
        if self.lock != NULL:
            PyThread_free_lock(self.lock)

    cdef void inc_one(self, size_t worker, size_t idx) nogil:
        self.updates[worker, idx] += 1

    cdef void dec_one(self, size_t worker, size_t idx) nogil:
        self.updates[worker, idx] -= 1

    cdef uint64_t[::1] get_local_counts(self, size_t worker) nogil:
        return self.local_counts[worker]

    cdef void update_counts(self, size_t worker) nogil:
        cdef size_t i
        self.delay[worker] += 1
        if self.delay[worker] == self.sloppy_level:

            PyThread_acquire_lock(self.lock, NOWAIT_LOCK)
            for i in range(<size_t>self.global_counts.shape[0]):
                self.global_counts[i] += self.updates[worker, i]
                self.local_counts[worker, i] = self.global_counts[i]
            PyThread_release_lock(self.lock)

            for i in range(<size_t>self.global_counts.shape[0]):
                self.updates[worker, i] = 0
            self.delay[worker] = 0
