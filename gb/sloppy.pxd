# -*- coding: utf8
# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: wraparound=False


from libc.stdint cimport uint64_t


cdef class SloppyCounter(object):
    cdef size_t sloppy_level
    cdef uint64_t[:, ::1] local_state
    cdef uint64_t[:, ::1] local_counts
    cdef uint64_t[::1] global_counts
    cdef uint64_t[::1] delay
    cdef int[:, ::1] updates

    cdef void update_counts(self, size_t worker) nogil
    cdef void inc_one(self, size_t worker, size_t idx) nogil
    cdef void dec_one(self, size_t worker, size_t idx) nogil
    cdef uint64_t[::1] get_local_counts(self, size_t worker) nogil
