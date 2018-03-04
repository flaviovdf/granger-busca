# -*- coding: utf8
# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: wraparound=False


from gb.collections.table cimport Table
from gb.randomkit.random cimport rand
from gb.stamps cimport Timestamps
from gb.sparsefp cimport FenwickSampler

from libc.stdint cimport uint64_t
from libc.stdio cimport printf

from libcpp.algorithm cimport sort as stdsort
from libcpp.vector cimport vector
cdef extern from "<algorithm>" namespace "std" nogil:
    void nth_element(vector.iterator first, vector.iterator nth,
                     vector.iterator last)

import numpy as np


cdef double E = 2.718281828459045


cdef extern from 'math.h':
    double exp(double) nogil


cdef void update_mu_rate(size_t proc_a, Timestamps all_stamps,
                         double count_background, double[::1] mu_rates) nogil:

    cdef double[::1] timestamps_proc_a = all_stamps.get_stamps(proc_a)
    cdef size_t n_events = timestamps_proc_a.shape[0]
    cdef double T = timestamps_proc_a[n_events-1]
    cdef double rate
    if T == 0:
        rate = 0
    else:
        rate = count_background / T
    mu_rates[proc_a] = rate


cdef void update_beta_rate(size_t proc_a, Timestamps all_stamps,
                           double[::1] beta_rates) nogil:

    cdef vector[double] all_deltas
    cdef size_t n_proc = beta_rates.shape[0]
    cdef size_t proc_b, i
    cdef double ti, tp
    cdef double max_ti = 0
    cdef double[::1] stamps_b
    cdef size_t[::1] state_b
    cdef size_t n_elements = 0
    for proc_b in range(n_proc):
        stamps_b = all_stamps.get_stamps(proc_b)
        state_b = all_stamps.get_causes(proc_b)
        for i in range(<size_t>stamps_b.shape[0]):
            if state_b[i] == proc_a:
                ti = stamps_b[i]
                if ti > max_ti:
                    max_ti = ti
                tp = all_stamps.find_previous(proc_a, ti)
                all_deltas.push_back(ti - tp)
                n_elements += 1

    if n_elements >= 1:
        stdsort(all_deltas.begin(), all_deltas.end())
        beta_rates[proc_a] = all_deltas[all_deltas.size() // 2]
        if n_elements % 2 == 0:
            beta_rates[proc_a] += all_deltas[(all_deltas.size() // 2)-1]
            beta_rates[proc_a] = beta_rates[proc_a] / 2
    else:
        beta_rates[proc_a] = max_ti


cdef double busca_probability(size_t i, size_t proc_a, size_t proc_b,
                              Timestamps all_stamps, double alpha_ba,
                              double beta_rate) nogil:
    cdef double[::1] stamps = all_stamps.get_stamps(proc_a)
    cdef double t = stamps[i]
    cdef double tp
    cdef double tpp
    if i > 0:
        tp = stamps[i-1]
    else:
        tp = 0
    if tp != 0:
        if proc_a == proc_b:
            if i > 1:
                tpp = stamps[i-2]
            else:
                tpp = 0
        else:
            tpp = all_stamps.find_previous(proc_b, tp)
    else:
        tpp = 0

    cdef double rate = alpha_ba / (beta_rate/E + tp - tpp)
    return rate



cdef size_t metropolis_step(size_t i, size_t proc_a, Timestamps all_stamps,
                            FenwickSampler sampler, double[::1] mu_rates,
                            double[::1] beta_rates,
                            double background_delta) nogil:

    cdef size_t n_proc = mu_rates.shape[0]

    cdef double[::1] stamps = all_stamps.get_stamps(proc_a)
    cdef size_t[::1] causes = all_stamps.get_causes(proc_a)

    cdef double mu_rate = mu_rates[proc_a]
    cdef double ti = stamps[i]
    cdef double mu_prob = mu_rate * background_delta * \
            exp(-mu_rate*background_delta)

    if rand() < mu_prob:
        return n_proc

    cdef size_t candidate = sampler.sample()
    cdef size_t proc_b = causes[i]
    if proc_b == n_proc:
        return candidate

    cdef double alpha_ba = sampler.get_probability(proc_b)
    cdef double alpha_ca = sampler.get_probability(candidate)

    cdef double p_b = busca_probability(i, proc_a, proc_b, all_stamps,
                                        alpha_ba, beta_rates[proc_b])
    cdef double p_c = busca_probability(i, proc_a, candidate, all_stamps,
                                        alpha_ca, beta_rates[candidate])

    cdef int choice
    if rand() < min(1, (p_c / p_b)):
        choice = candidate
    else:
        choice = proc_b
    return choice


cdef void sample_alpha(size_t proc_a, Timestamps all_stamps,
                       FenwickSampler sampler, uint64_t[::1] num_background,
                       double[::1] mu_rates, double[::1] beta_rates) nogil:
    cdef size_t i
    cdef size_t influencer
    cdef size_t new_influencer
    cdef size_t n_proc = mu_rates.shape[0]

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

        new_influencer = metropolis_step(i, proc_a, all_stamps, sampler,
                                         mu_rates, beta_rates,
                                         stamps[i] - prev_back_t)

        if new_influencer == n_proc:
            num_background[proc_a] += 1
        else:
            sampler.inc_one(new_influencer)
        causes[i] = new_influencer
        prev_back_t = prev_back_t_aux


cdef void sampleone(Timestamps all_stamps, FenwickSampler sampler,
                    uint64_t[::1] num_background, double[::1] mu_rates,
                    double[::1] beta_rates, size_t n_iter) nogil:

    cdef size_t n_proc = mu_rates.shape[0]
    cdef size_t a
    printf("[logger]\t Learning mu.\n")
    for a in range(n_proc):
        update_mu_rate(a, all_stamps, num_background[a], mu_rates)

    printf("[logger]\t Learning beta.\n")
    for a in range(n_proc):
        update_beta_rate(a, all_stamps, beta_rates)

    printf("[logger]\t Sampling Alpha.\n")
    for a in range(n_proc):
        sampler.set_current_process(a)
        sample_alpha(a, all_stamps, sampler, num_background, mu_rates,
                     beta_rates)


cdef void cfit(Timestamps all_stamps, FenwickSampler sampler,
               uint64_t[::1] num_background, double[::1] mu_rates,
               double[::1] beta_rates, size_t n_iter) nogil:
    printf("[logger] Sampler is starting\n")
    printf("[logger]\t n_proc=%ld\n", mu_rates.shape[0])
    printf("\n")

    cdef size_t iteration, a, b
    for iteration in range(n_iter):
        printf("[logger] Iter=%lu. Sampling...\n", iteration)
        sampleone(all_stamps, sampler, num_background, mu_rates, beta_rates,
                  n_iter)


def fit(dict all_timestamps, double alpha_prior, size_t n_iter):

    cdef size_t n_proc = len(all_timestamps)
    cdef Timestamps all_stamps = Timestamps(all_timestamps)
    cdef Table causal_counts = Table(n_proc)

    cdef uint64_t[::1] sum_b = np.zeros(n_proc, dtype='uint64', order='C')
    cdef uint64_t[::1] num_background = np.zeros(n_proc, dtype='uint64',
                                                 order='C')

    cdef size_t a, b
    cdef uint64_t count
    cdef size_t[::1] causes
    for a in range(n_proc):
        causes = all_stamps.get_causes(a)
        causes[:] = np.random.randint(n_proc + 1, causes.shape[0])
        for b in causes:
            if b == n_proc:
                num_background[a] += 1
            else:
                count = causal_counts.get_cell(a, b)
                causal_counts.set_cell(a, b, count + 1)
                sum_b[b] += 1

    cdef FenwickSampler sampler = FenwickSampler(causal_counts, all_stamps,
                                                 sum_b, alpha_prior, 0)

    cdef double[::1] mu_rates = np.zeros(n_proc, dtype='d', order='C')
    cdef double[::1] beta_rates = np.zeros(n_proc, dtype='d', order='C')

    cfit(all_stamps, sampler, num_background, mu_rates, beta_rates, n_iter)

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

    return Alpha, np.array(mu_rates), np.array(beta_rates), num_background, \
        curr_state
