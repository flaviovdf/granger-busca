# -*- coding: utf8
# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: wraparound=False


cdef inline double dirmulti_posterior(int n_ab, int n_b, int n,
                                      double prior) nogil:
    return (n_ab + prior) / (n_b + n * prior)
