# -*- coding: utf8
# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: wraparound=False


from libcpp.unordered_map cimport unordered_map


cdef class Timestamps(object):

    cdef double[::1] all_stamps
    cdef int[::1] causes
    cdef unordered_map[int, int] start_positions
    cdef int n_stamps

    cdef double[::1] get_stamps(self, int process) nogil
    cdef int[::1] get_causes(self, int process) nogil
