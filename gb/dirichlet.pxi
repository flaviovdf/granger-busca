# -*- coding: utf8
# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: wraparound=False


cdef inline double dirmulti_posterior(int n_ab, int n_b, int n,
                                      double prior) nogil:
    return (n_ab + prior) / (n_b + n * prior)


cdef inline double numerator(double p, int n_b, int n, double prior) nogil:
    return p * (n_b + n * prior)


cdef inline double renormalize(double p, int n_b, int n, double prior) nogil:
    return p / (n_b + n * prior)


cdef inline double inc(int n_ab, int n_b, int n, double prior,
                       double delta) nogil:
    cdef double b = n_b + n * prior
    cdef double a = n_ab + prior
    return (b * delta - a * delta) / (b * (b + delta))
