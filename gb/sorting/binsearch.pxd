# -*- coding: utf8
# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: wraparound=False


from libcpp.vector cimport vector


cdef int searchsorted(vector[double] &array, double value, int lower) nogil
