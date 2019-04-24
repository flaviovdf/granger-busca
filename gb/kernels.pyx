# -*- coding: utf8
# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: language_level=3
# cython: wraparound=False


from libc.stdio cimport printf
from libc.stdint cimport uint64_t
from libc.stdlib cimport abort

import numpy as np


cdef extern from 'math.h':
    double exp(double) nogil
    double log(double) nogil
    double INFINITY


cdef class AbstractKernel(object):
    cdef void set_current_process(self, size_t proc, double[::1] alphas) nogil:
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

    cdef void set_current_process(self, size_t proc, double[::1] alphas) nogil:
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

    def __init__(self, PoissonKernel poisson):
        cdef size_t n_proc = poisson.mu.shape[0]
        self.poisson = poisson
        self.beta = np.ones(n_proc, dtype='d')*5
        self.saved_betas = np.ones(n_proc*n_proc, dtype='d')*5
        self.dbeta = np.zeros(n_proc, dtype='d')
        self.past_delta = np.zeros(n_proc, dtype='d')

    cdef void gradient_step(self, double[::1] alphas, double *stamps, size_t proc, size_t proc_size) nogil:
        
        cdef double mu = self.mu_rate(proc)
        cdef double tpp = 0
        cdef double tp = 0
        cdef double t = 0
        cdef double delta = 0
        cdef size_t t_idx
        cdef size_t b
        cdef double sum_exct =0
        cdef double t_idxminus1 

        for t_idx in range(proc_size):
            sum_exct = 0
            for b in range(self.beta.shape[0]):
                if (t_idx == 0):
                    self.dbeta[b] = 0
                    self.past_delta[b] = 0
                t = stamps[t_idx]
                tp = self.poisson.timestamps.find_previous(b, t)
                if tp != 0 or (tp == 0. and self.poisson.timestamps.get_stamp(b, 0) == 0 and t != 0):
                    delta = t - tp
                    sum_exct += alphas[b]/(self.beta[b]+delta)
            
            for b in range(self.beta.shape[0]):
                t = stamps[t_idx]
                tp = self.poisson.timestamps.find_previous(b, t)
                if tp != 0 or (tp == 0. and self.poisson.timestamps.get_stamp(b, 0) == 0 and t != 0):
                    delta = t - tp
                    self.dbeta[b] -= alphas[b]/((mu+sum_exct)*(self.beta[b]+delta)**2)
                    if (t_idx > 1 and self.past_delta[b] > 0):
                        self.dbeta[b] += (alphas[b]*(stamps[t_idx]-stamps[t_idx-1]))/(self.beta[b]+self.past_delta[b])**2
                    self.past_delta[b] = delta


    cdef void set_current_process(self, size_t proc, double[::1] alphas) nogil:
        self.poisson.set_current_process(proc, alphas)
        cdef size_t a = proc
        cdef size_t n_proc = self.beta.shape[0]
        cdef size_t proc_size = self.poisson.timestamps.get_size(a)
        cdef double *stamps
        self.poisson.timestamps.get_stamps(a, &stamps)

        cdef size_t *causes
        self.poisson.timestamps.get_causes(a, &causes)

        cdef size_t b 
        for b in range(self.beta.shape[0]):#initialize beta
            self.beta[b] = self.saved_betas[b+a*n_proc]
            #self.beta[b] = 5
        
        cdef size_t it = 0
        cdef size_t max_it = 4000
        cdef double precision = 1e-3
        cdef double past_beta
        cdef double step_size
        cdef double gammab = 0.01
        cdef uint64_t prec_reached = 0
        while not(prec_reached):

            it += 1
            if (it>max_it):
                max_it *= 2
                printf("max iter reached a: %ld\n",a)
                gammab *= 0.5

            prec_reached = 1
            self.gradient_step(alphas, stamps, proc, proc_size)
            for b in range(self.beta.shape[0]):
                past_beta = self.beta[b]
                self.beta[b] = self.beta[b] + gammab*self.dbeta[b]
                if self.beta[b] < 0:
                    self.beta[b] = 0

                step_size = self.beta[b] - past_beta
                step_size = step_size if step_size >= 0 else -step_size
                if step_size > precision:
                    prec_reached = 0

        for b in range(self.beta.shape[0]):
            self.saved_betas[b+a*n_proc] = self.beta[b]
            printf("%ld-b: %lf ",a,self.beta[b])
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

        cdef double rate = alpha_ab / (self.beta[b] + (tp - tpp))
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
