# -*- coding: utf8
# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: language_level=3
# cython: wraparound=False


cdef class FPTree:
    cdef size_t size
    cdef size_t t_pos
    cdef double[::1] values

    cdef void reset(self) nogil
    cdef void set_value(self, size_t i, double val) nogil
    cdef double get_value(self, size_t i) nogil
    cdef size_t sample(self, double urnd) nogil
    cdef double get_total(self) nogil
