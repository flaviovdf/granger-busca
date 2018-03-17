# -*- coding: utf8
# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: wraparound=False


from gb.collections.table cimport RobinHoodHash


cdef class Timestamps(object):

    cdef double[::1] all_stamps
    cdef size_t[::1] causes
    cdef size_t n_stamps
    cdef size_t[::1] start_positions
    cdef size_t n_proc

    cdef double[::1] get_stamps(self, size_t process) nogil
    cdef size_t[::1] get_causes(self, size_t process) nogil
    cdef double find_previous(self, size_t process, double t) nogil
    cdef size_t num_proc(self) nogil
