# -*- coding: utf8
# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: language_level=3
# cython: wraparound=False


from cpython.pythread cimport PyThread_type_lock

from libc.stdint cimport uint64_t


cdef class SloppyCounter(object):
    cdef size_t sloppy_level
    cdef uint64_t[:, ::1] local_state
    cdef uint64_t[:, ::1] local_counts
    cdef uint64_t[::1] global_counts
    cdef uint64_t[::1] delay
    cdef int[:, ::1] updates
    cdef PyThread_type_lock lock

    cdef void update_counts(self, size_t worker) nogil
    cdef void inc_one(self, size_t worker, size_t idx) nogil
    cdef void dec_one(self, size_t worker, size_t idx) nogil
    cdef void get_local_counts(self, size_t worker, uint64_t **at) nogil
