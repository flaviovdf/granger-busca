# -*- coding: utf8

from libcpp.vector cimport vector


cdef int searchsorted(vector[double] &array, double value, int lower) nogil
