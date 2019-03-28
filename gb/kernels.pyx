# -*- coding: utf8
# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: language_level=3
# cython: wraparound=False

from gb.bvls.solve cimport my_bvls

from libc.stdio cimport printf
from libc.stdint cimport uint64_t
from libc.stdlib cimport abort

import numpy as np


cdef extern from 'math.h':
    double exp(double) nogil
    double log(double) nogil
    double INFINITY


cdef class AbstractKernel(object):
    cdef void set_current_process(self, size_t process) nogil:
        printf('[gb.kernels] Do not use the AbstractKernel\n')
        abort()
    cdef double background_probability(self, double dt) nogil:
        printf('[gb.kernels] Do not use the AbstractKernel\n')
        abort()
    cdef double cross_rate(self, size_t i, size_t b, double alpha_ab) nogil:
        printf('[gb.kernels] Do not use the AbstractKernel\n')
        abort()
    cdef double mu_rate(self, size_t process) nogil:
        printf('[gb.kernels] Do not use the AbstractKernel\n')
        abort()
    cdef double[::1] get_mu_rates(self) nogil:
        printf('[gb.kernels] Do not use the AbstractKernel\n')
        abort()


cdef class PoissonKernel(AbstractKernel):

    def __init__(self, Timestamps timestamps, size_t n_proc, RNG rng):
        self.timestamps = timestamps
        self.mu = np.zeros(n_proc, dtype='d')
        self.rng = rng

    cdef void set_current_process(self, size_t proc) nogil:
        self.current_process = proc

        cdef size_t n_proc = self.timestamps.num_proc()
        cdef size_t n = self.timestamps.get_size(proc)
        cdef size_t *causes
        self.timestamps.get_causes(proc, &causes)

        cdef double *stamps
        self.timestamps.get_stamps(proc, &stamps)

        cdef size_t i
        cdef double count_background = 0
        cdef double prev_poisson = -1
        cdef double dts = 0.0
        for i in range(n):
            if causes[i] == n_proc:
                if prev_poisson != -1:
                    count_background += 1
                    dts += stamps[i] - prev_poisson
                prev_poisson = stamps[i]

        if dts > 0:
            self.mu[proc] = count_background / dts
        else:
            self.mu[proc] = 0

    cdef double background_probability(self, double dt) nogil:
        cdef double mu_rate = self.mu[self.current_process]
        return mu_rate * dt * exp(-mu_rate * dt)

    cdef double mu_rate(self, size_t process) nogil:
        return self.mu[process]

    cdef double cross_rate(self, size_t i, size_t b, double alpha_ab) nogil:
        return 0.0

    cdef double[::1] get_mu_rates(self) nogil:
        return self.mu


cdef class WoldKernel(AbstractKernel):

    def __init__(self, PoissonKernel poisson,
                 size_t min_pts_regression):

        cdef size_t n_proc = poisson.mu.shape[0]

        self.poisson = poisson
        self.beta = np.zeros(n_proc, dtype='d')
        self.gamma = np.zeros(n_proc, dtype='d')
        self.gamma_sup = np.zeros(n_proc, dtype='d')
        self.gamma_inf = np.zeros(n_proc, dtype='d')
        self.y_vect_regression = np.zeros(n_proc, dtype='d')
        self.x_vect_regression = np.zeros(n_proc, dtype='d')
        self.n_pts_regression = np.zeros(n_proc, dtype='uint64')
        self.prev_delta_vect_regression = np.zeros(n_proc, dtype='d')
        self.min_pts_regression = min_pts_regression

    cdef void set_current_process(self, size_t proc) nogil:
        self.poisson.set_current_process(proc)
        cdef size_t a = proc

        cdef size_t proc_size = self.poisson.timestamps.get_size(a)
        cdef double *stamps
        self.poisson.timestamps.get_stamps(a, &stamps)

        cdef size_t *causes
        self.poisson.timestamps.get_causes(a, &causes)

        cdef double tpp = 0
        cdef double tp = 0
        cdef double t = 0

        cdef size_t b

        for b in range(<size_t>self.beta.shape[0]):
            self.y_vect_regression[b] = 0
            self.x_vect_regression[b] = 0
            self.prev_delta_vect_regression[b] = -1
            self.n_pts_regression[b] = 0

        cdef size_t i
        for i in range(proc_size):
            b = causes[i]
            if b == <size_t>self.beta.shape[0]:
                continue
            t = stamps[i]
            tp = self.poisson.timestamps.find_previous(b, t)
            if self.prev_delta_vect_regression[b] == -1:
                self.prev_delta_vect_regression[b] = t - tp
            else:
                if tp != 0 or \
                    tp == 0. and self.poisson.timestamps.get_stamp(b, 0) == 0:
                    self.n_pts_regression[b] += 1
                    self.y_vect_regression[b] += log(t - tp)
                    self.x_vect_regression[b] += log(self.prev_delta_vect_regression[b])
                    self.prev_delta_vect_regression[b] = t - tp

        cdef size_t n
        for b in range(<size_t>self.beta.shape[0]):
            n = self.n_pts_regression[b]
            self.y_vect_regression[b] = self.y_vect_regression[b] / n
            self.x_vect_regression[b] = self.x_vect_regression[b] / n
            self.prev_delta_vect_regression[b] = -1
            self.gamma_sup[b] = 0
            self.gamma_inf[b] = 0

        cdef double xi
        cdef double yi
        cdef double x_mean
        cdef double y_mean

        for i in range(proc_size):
            b = causes[i]
            if b == <size_t>self.beta.shape[0]:
                continue
            t = stamps[i]
            tp = self.poisson.timestamps.find_previous(b, t)
            if self.prev_delta_vect_regression[b] == -1:
                self.prev_delta_vect_regression[b] = t - tp
            else:
                if tp != 0 or \
                    tp == 0. and self.poisson.timestamps.get_stamp(b, 0) == 0:
                    yi = log(t - tp)
                    xi = log(self.prev_delta_vect_regression[b])
                    y_mean = self.y_vect_regression[b]
                    x_mean = self.x_vect_regression[b]
                    self.gamma_sup[b] += (xi - x_mean) * (yi - y_mean)
                    self.gamma_inf[b] += (xi - x_mean) * (xi - x_mean)
                    self.prev_delta_vect_regression[b] = t - tp

        for b in range(<size_t>self.gamma.shape[0]):
            if self.n_pts_regression[b] >= 2:
                self.gamma[b] = self.gamma_sup[b] / self.gamma_inf[b]
                self.beta[b] = self.y_vect_regression[b]
                self.beta[b] -= (self.x_vect_regression[b] * self.gamma[b])
            if self.gamma[b] < 0:
                self.gamma[b] = 0
            if self.beta[b] < 1:
                self.beta[b] = 1
                # self.gamma[b] = 1.0
                # self.beta[b] = 1.0
            # if self.gamma[b] >= 1:
            #    self.gamma[b] = 1
            # if self.gamma[b] < 0:
            #     self.gamma[b] = 0.000000001
            # if self.beta[b] < 1.0:
            #     self.beta[b] = 1.0
            printf("a=%ld, b=%ld, gamma=%lf, beta=%lf\n", a, b,
                   self.gamma[b], self.beta[b])
        printf("\n")

    cdef double background_probability(self, double dt) nogil:
        return self.poisson.background_probability(dt)

    cdef double mu_rate(self, size_t process) nogil:
        return self.poisson.mu_rate(process)

    cdef double cross_rate(self, size_t i, size_t b, double alpha_ab) nogil:
        cdef size_t a = self.poisson.current_process

        cdef double *stamps
        self.poisson.timestamps.get_stamps(a, &stamps)

        cdef double t = stamps[i]
        cdef double tp
        cdef double tpp
        if i > 0:
            tp = stamps[i-1]
        else:
            tp = 0
        if tp != 0:
            if a == b:
                if i > 1:
                    tpp = stamps[i-2]
                else:
                    tpp = 0
            else:
                tpp = self.poisson.timestamps.find_previous(b, tp)
        else:
            tpp = 0

        cdef double rate = alpha_ab / (self.beta[b] + self.gamma[b]*(tp - tpp))
        return rate


cdef class TruncatedHawkesKernel(WoldKernel):

    cdef double cross_rate(self, size_t i, size_t b, double alpha_ab) nogil:
        cdef size_t a = self.poisson.current_process
        cdef double *stamps
        self.poisson.timestamps.get_stamps(a, &stamps)
        cdef double t = stamps[i]
        cdef double tp
        if a == b:
            if i > 1:
                tp = stamps[i-1]
            else:
                tp = 0
        else:
            tp = self.poisson.timestamps.find_previous(b, t)

        cdef double rate = alpha_ab / (self.beta[b] + t - tp)
        return rate
