# -*- coding: utf8
# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: wraparound=False


from cython cimport parallel

from gb.kernels cimport AbstractKernel
from gb.kernels cimport PoissonKernel
from gb.kernels cimport TruncatedHawkesKernel
from gb.kernels cimport WoldKernel

from gb.randomkit.random cimport RNG

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
                       AbstractSampler sampler,  AbstractKernel kernel) nogil:
    cdef size_t i
    cdef size_t influencer
    cdef size_t new_influencer
    cdef size_t n_proc = all_stamps.num_proc()

    cdef size_t n = all_stamps.get_size(proc_a)
    cdef double *stamps
    all_stamps.get_stamps(proc_a, &stamps)
    cdef size_t *causes
    all_stamps.get_causes(proc_a, &causes)

    cdef double prev_back_t = 0      # stores last known background time stamp
    cdef double prev_back_t_aux = 0  # every it: prev_back_t = prev_back_t_aux
    cdef double dt_poisson
    cdef double dt_wold
    cdef size_t candidate
    cdef double candidate_prob
    for i in range(n):
        influencer = causes[i]
        if influencer == n_proc:
            prev_back_t_aux = stamps[i] # found a background ts
        else:
            sampler.dec_one(influencer)

        dt_poisson = stamps[i] - prev_back_t
        if i > 0:
            dt_wold = stamps[i] - stamps[i-1]
        else:
            dt_wold = stamps[i]

        if sampler.is_background(kernel.mu_rate(proc_a), dt_poisson,
                                 candidate_prob, dt_wold):
            new_influencer = n_proc
        else:
            sampler.sample_for_idx(i, kernel, &candidate, &candidate_prob)
            new_influencer = candidate
            sampler.inc_one(new_influencer)

        causes[i] = new_influencer
        prev_back_t = prev_back_t_aux


cdef void do_work(Timestamps all_stamps, SloppyCounter sloppy,
                  AbstractSampler sampler, AbstractKernel kernel,
                  size_t n_iter, size_t worker_id, size_t[::1] workload) nogil:

    cdef size_t iteration
    cdef size_t proc_a, i
    for iteration in range(n_iter):
        for i in range(<size_t>workload.shape[0]):
            proc_a = workload[i]
            sampler.set_current_process(proc_a)
            kernel.set_current_process(proc_a)
            sample_alpha(proc_a, all_stamps, sampler, kernel)
        sloppy.update_counts(worker_id)


def fit(Timestamps all_stamps, SloppyCounter sloppy, double alpha_prior,
        double[::1] beta, size_t n_iter, size_t worker_id,
        size_t[::1] workload, int metropolis_walker=True):

    cdef RNG rng = RNG()
    cdef size_t n_proc = all_stamps.num_proc()
    cdef BaseSampler base_sampler = BaseSampler(all_stamps, sloppy, worker_id,
                                                alpha_prior, rng)
    cdef AbstractSampler sampler
    if metropolis_walker == 1:
        sampler = FenwickSampler(base_sampler, n_proc)
    else:
        sampler = CollapsedGibbsSampler(base_sampler, n_proc)

    cdef PoissonKernel poisson = PoissonKernel(all_stamps, n_proc, rng)
    cdef AbstractKernel kernel = WoldKernel(poisson, beta)

    printf("Worker %lu starting\n", worker_id)
    with nogil:
        do_work(all_stamps, sloppy, sampler, kernel, n_iter, worker_id,
                workload)
    printf("Worker %lu done\n", worker_id)

    cdef dict Alpha = {}
    cdef dict curr_state = {}

    cdef size_t a, b, i, j
    cdef size_t n
    cdef size_t *causes
    cdef uint64_t[::1] num_background = np.zeros(n_proc, dtype='uint64')

    for i in range(<size_t>workload.shape[0]):
        a = workload[i]
        n = all_stamps.get_size(a)
        all_stamps.get_causes(a, &causes)
        Alpha[a] = {}
        curr_state[a] = np.zeros(n, dtype='uint64', order='C')
        for j in range(n):
            b = causes[j]
            curr_state[a][j] = b
            if b != n_proc:
                if b not in Alpha[a]:
                    Alpha[a][b] = 0
                Alpha[a][b] += 1
            else:
                num_background[a] += 1

    return Alpha, np.asanyarray(poisson.get_mu_rates()), \
        np.asanyarray(num_background), curr_state
