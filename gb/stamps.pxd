# -*- coding: utf8
# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: wraparound=False

from libcpp.unordered_map cimport unordered_map

cdef class Stamps(object):

    cdef float[::1] all_stamps
    cdef unordered_map[int, int] start_positions
    cdef float[::1] get_stamps(self, int process) nogil
