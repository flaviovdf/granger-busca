# -*- coding: utf8
# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: wraparound=False


from libc.stdint cimport uint64_t


cdef class SloppyCounter(object):
    cdef size_t sloppy_level
    cdef uint64_t[:, ::1] last_seen
    cdef uint64_t[::1] global_counts
    cdef uint64_t[::1] updates

    cdef void update_counts(self, size_t worker, uint64_t[::1] update) nogil
