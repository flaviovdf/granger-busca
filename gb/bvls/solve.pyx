# -*- coding: utf8
# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: language_level=3
# cython: wraparound=False


cdef extern from 'bvls.h':
    int bvls(int m, int n, const double *A, const double *b, const double *lb,
             const double *ub, double *x) nogil


cdef int my_bvls(double[:, ::1] A, double[::1] b, double[::1] lb,
                 double[::1] ub, double[::1] result) nogil:
    cdef int n = A.shape[0]
    cdef int m = A.shape[1]
    return bvls(m, n, &A[0, 0], &b[0], &lb[0], &ub[0], &result[0])


def _bvls(double[:, ::1] A, double[::1] b, double[::1] lb,
          double[::1] ub, double[::1] result):
    return my_bvls(A, b, lb, ub, result)
