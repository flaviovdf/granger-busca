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

    cdef void get_stamps(self, size_t process, double **at) nogil
    cdef void get_causes(self, size_t process, size_t **at) nogil
    cdef size_t get_cause(self, size_t process, size_t i) nogil
    cdef double get_stamp(self, size_t process, size_t i) nogil
    cdef size_t get_size(self, size_t process) nogil
    cdef double find_previous(self, size_t process, double t) nogil
    cdef size_t num_proc(self) nogil
