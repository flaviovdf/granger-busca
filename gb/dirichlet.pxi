# -*- coding: utf8
# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: wraparound=False


cdef inline double dirmulti_posterior(int joint_count, int denominator,
                                      int n, double prior) nogil:
    return (joint_count + prior) / (denominator + n * prior)


cdef inline double numerator(double p, int denominator, int n,
                             double prior) nogil:
    return p * (denominator + n * prior)


cdef inline double renormalize(double p, int denominator, int n,
                               double prior) nogil:
    return p / (denominator + n * prior)


cdef inline double inc(int joint_count, int denominator, int n, double prior,
                       double delta) nogil:
    cdef double b = denominator + n * prior
    cdef double a = joint_count + prior
    return (b * delta - a * delta) / (b * (b + delta))
