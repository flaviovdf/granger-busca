# -*- coding: utf8
# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: wraparound=False


from libcpp.unordered_map cimport unordered_map


cdef class Timestamps(object):

    cdef double[::1] all_stamps
    cdef size_t[::1] causes
    cdef size_t n_stamps
    cdef unordered_map[size_t, size_t] start_positions

    cdef double[::1] get_stamps(self, size_t process) nogil
    cdef size_t[::1] get_causes(self, size_t process) nogil
    cdef double find_previous(self, size_t process, double t) nogil
