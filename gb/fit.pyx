# -*- coding: utf8
# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: wraparound=False


from gb.collections.table cimport Table

from gb.kernels cimport AbstractKernel
from gb.kernels cimport PoissonKernel
from gb.kernels cimport BuscaKernel
from gb.kernels cimport TruncatedHawkesKernel

from gb.randomkit.random cimport rand

from gb.samplers cimport AbstractSampler
from gb.samplers cimport BaseSampler
from gb.samplers cimport CollapsedGibbsSampler
from gb.samplers cimport FenwickSampler

from gb.stamps cimport Timestamps

from gb.sloppy cimport SloppyCounter

from libc.stdint cimport uint64_t
from libc.stdio cimport printf

import numpy as np


cdef void sample_alpha(size_t proc_a, Timestamps all_stamps,
                       AbstractSampler sampler,  AbstractKernel kernel,
                       uint64_t[::1] num_background) nogil:
    cdef size_t i
    cdef size_t influencer
    cdef size_t new_influencer
    cdef size_t n_proc = num_background.shape[0]

    cdef double[::1] stamps = all_stamps.get_stamps(proc_a)
    cdef size_t[::1] causes = all_stamps.get_causes(proc_a)

    cdef double prev_back_t = 0      # stores last known background time stamp
    cdef double prev_back_t_aux = 0  # every it: prev_back_t = prev_back_t_aux
    for i in range(<size_t>stamps.shape[0]):
        influencer = causes[i]
        if influencer == n_proc:
            num_background[proc_a] -= 1
            prev_back_t_aux = stamps[i] # found a background ts
        else:
            sampler.dec_one(influencer)

        if rand() < kernel.background_probability(stamps[i] - prev_back_t):
            new_influencer = n_proc
        else:
            new_influencer = sampler.sample_for_idx(i, kernel)

        if new_influencer == n_proc:
            num_background[proc_a] += 1
        else:
            sampler.inc_one(new_influencer)
        causes[i] = new_influencer
        prev_back_t = prev_back_t_aux


cdef void sampleone(Timestamps all_stamps, AbstractSampler sampler,
                    AbstractKernel kernel, uint64_t[::1] num_background,
                    size_t n_iter) nogil:

    cdef size_t n_proc = num_background.shape[0]
    cdef size_t proc_a
    for proc_a in range(n_proc):
        kernel.set_current_process(proc_a)
        kernel.update_mu_rate(num_background[proc_a])
        kernel.update_cross_rates()

    for proc_a in range(n_proc):
        sampler.set_current_process(proc_a)
        kernel.set_current_process(proc_a)
        sample_alpha(proc_a, all_stamps, sampler, kernel, num_background)


cdef void cfit(Timestamps all_stamps, AbstractSampler sampler,
               AbstractKernel kernel, uint64_t[::1] num_background,
               size_t n_iter) nogil:
    printf("[logger] Sampler is starting\n")
    printf("[logger]\t n_proc=%ld\n", num_background.shape[0])
    printf("\n")

    cdef size_t iteration
    for iteration in range(n_iter):
        printf("[logger] Iter=%lu. Sampling...\n", iteration)
        sampleone(all_stamps, sampler, kernel, num_background, n_iter)


def fit(Timestamps all_stamps, SloppyCounter sloppy, double alpha_prior,
        size_t n_iter, size_t[::1] workload, int metropolis_walker=True):

    cdef size_t n_proc = all_stamps.num_proc()
    cdef Table causal_counts = Table(n_proc)

    cdef uint64_t[::1] sum_b = np.zeros(n_proc, dtype='uint64', order='C')
    cdef uint64_t[::1] num_background = np.zeros(n_proc, dtype='uint64',
                                                 order='C')

    cdef size_t a, b, i
    cdef uint64_t count
    cdef size_t[::1] causes
    cdef size_t[::1] init_state
    cdef size_t n_stamps = 0
    for a in range(n_proc):
        causes = all_stamps.get_causes(a)
        n_stamps += causes.shape[0]
        init_state = np.random.randint(0, n_proc + 1,
                                       size=causes.shape[0], dtype='uint64')
        for i in range(<size_t>causes.shape[0]):
            b = init_state[i]
            causes[i] = b
            if b == n_proc:
                num_background[a] += 1
            else:
                count = causal_counts.get_cell(a, b)
                causal_counts.set_cell(a, b, count + 1)
                sum_b[b] += 1

    cdef BaseSampler base_sampler = BaseSampler(causal_counts, all_stamps,
                                                sum_b, alpha_prior, 0)
    cdef AbstractSampler sampler
    if metropolis_walker == 1:
        sampler = FenwickSampler(base_sampler, n_proc)
    else:
        sampler = CollapsedGibbsSampler(base_sampler, n_proc)

    cdef PoissonKernel poisson = PoissonKernel(all_stamps, n_proc)
    cdef AbstractKernel kernel = BuscaKernel(poisson, all_stamps,
                                             n_proc, n_stamps)

    cfit(all_stamps, sampler, kernel, num_background, n_iter)

    Alpha = {}
    curr_state = {}
    for a in range(n_proc):
        Alpha[a] = {}
        causes = all_stamps.get_causes(a)
        curr_state[a] = np.array(causes)
        for b in causes:
            if b != n_proc:
                if b not in Alpha[a]:
                    Alpha[a][b] = 0
                Alpha[a][b] += 1

    return Alpha, np.array(poisson.get_mu_rates()), \
        np.array(kernel.get_beta_rates()), np.array(num_background), curr_state
