# -*- coding: utf8
# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: wraparound=False

from cython cimport parallel

from gb.collections.fptree cimport FPTree

from gb.randomkit.random cimport rand
from gb.randomkit.random cimport gamma as rgamma

from gb.sorting.binsearch cimport searchsorted

from libc.stdio cimport printf

from libcpp.algorithm cimport sort as stdsort
from libcpp.unordered_map cimport unordered_map as map
from libcpp.map cimport pair
from libcpp.vector cimport vector

import numpy as np


cdef double E = 2.718281828459045


cdef extern from 'math.h':
    double exp(double) nogil


cdef inline double dirmulti_posterior(map[int, map[int, int]] &Alpha_ab,
                                      int[::1] sum_b, int proc_a, int proc_b,
                                      double prior) nogil:
    '''
    Computes the posterior for the counts.

    Parameters
    ----------
    Alpha_ab: libstd map of ints -> map[int, int]
        Counts for each time b cross-excited a
    sum_b: int array
        Sum for a given row in Alpha_ab
    proc_a: int
        ID for process a
    proc_b: int
        ID for process b
    prior: double
        Prior for the process
    '''
    cdef int n = sum_b.shape[0]
    cdef int nba
    if Alpha_ab.count(proc_a) == 0:
        nba = 0
    elif Alpha_ab[proc_a].count(proc_b) == 0:
        nba = 0
    else:
        nba = Alpha_ab[proc_a][proc_b]
    return (nba + prior) / (sum_b[proc_b] + n * prior)


cdef inline int sample_background(double prob_background) nogil:
    '''Simply samples a random number and checks if < prob_background'''
    return rand() < prob_background


cdef inline double find_previous(vector[double] &timestamps, double t) nogil:
    '''
    Finds the closest timestamp with value less than the one informed.

    Parameters
    ----------
    timestamps: vector[double]
        the process to search in
    t: double
        the reference value to find tp < t such that t - tp is minimal
    '''

    cdef int tp_idx = max(searchsorted(timestamps, t, 0) - 1, 0)
    cdef double tp = timestamps[tp_idx]
    if tp >= t:
        tp = 0
    return tp


cdef inline double busca_probability(int i, int proc_a, int proc_b,
                                     map[int, vector[double]] &all_timestamps,
                                     double alpha_ba,
                                     double beta_proc_b) nogil:

    cdef double t = all_timestamps[proc_a][i]
    cdef double tp
    cdef double tpp
    if i > 0:
        tp = all_timestamps[proc_a][i-1]
    else:
        tp = 0
    if tp != 0:
        if proc_a == proc_b:
            if i > 1:
                tpp = all_timestamps[proc_a][i-2]
            else:
                tpp = 0
        else:
            tpp = find_previous(all_timestamps[proc_b], tp)
    else:
        tpp = 0

    cdef double rate = alpha_ba / (beta_proc_b/E + tp - tpp)
    return rate


cdef inline int metropolis_walk_step(int proc_a, int i, double prev_back_t,
                                     map[int, vector[double]] &all_timestamps,
                                     vector[int] &curr_state_proc_a,
                                     map[int, map[int, int]] &Alpha_ab,
                                     int[::1] sum_b, double alpha_prior,
                                     double[::1] mu_rates,
                                     double[::1] beta_rates,
                                     FPTree fptree) nogil:

    cdef double mu_rate = mu_rates[proc_a]
    cdef double ti = all_timestamps[proc_a][i]
    cdef double tp = prev_back_t
    cdef double dt = ti - tp
    cdef double mu_prob = mu_rate * dt * exp(-mu_rate*dt)

    cdef int candidate = fptree.sample(rand()*fptree.get_total())
    cdef double q_c = fptree.get_value(candidate)
    cdef double a_ca = dirmulti_posterior(Alpha_ab, sum_b, proc_a, candidate,
                                          alpha_prior)
    cdef double p_c = busca_probability(i, proc_a, candidate, all_timestamps,
                                        a_ca, beta_rates[candidate])

    cdef int curr_influencer_b = curr_state_proc_a[i]
    cdef double a_ba = dirmulti_posterior(Alpha_ab, sum_b, proc_a,
                                          curr_influencer_b, alpha_prior)
    cdef double q_b = 0
    cdef double p_b = 0
    if curr_influencer_b != -1:
        q_b = fptree.get_value(curr_influencer_b)
        p_b = busca_probability(i, proc_a, curr_influencer_b, all_timestamps,
                                a_ba, beta_rates[curr_influencer_b])


    cdef int choice
    cdef double busca_rate_choice
    if curr_influencer_b == -1 or rand() < min(1, (p_c * q_b) / (p_b * q_c)):
        choice = candidate
        busca_rate_choice = p_c / a_ca
    else:
        choice = curr_influencer_b
        busca_rate_choice = p_b / a_ba

    if sample_background(mu_prob):
        return -1
    else:
        return choice


cdef inline double inc(int n, int nb, int nba, double alpha_prior,
                       double delta) nogil:
    cdef double b = nb + n * alpha_prior
    cdef double a = nba + alpha_prior
    return (b * delta - a * delta) / (b * (b + delta))


cdef void sample_alpha(int proc_a, map[int, vector[double]] &all_timestamps,
                       vector[int] &curr_state,
                       map[int, map[int, int]] &Alpha_ab, int[::1] sum_b,
                       double[::1] mu_rates, double[::1] beta_rates,
                       double alpha_prior, FPTree fptree,
                       map[int, vector[int]] workload) nogil:

    cdef int i
    cdef int influencer
    cdef int new_influencer
    cdef int n_proc = mu_rates.shape[0]

    cdef double prev_back_t = 0      # stores last known background time stamp
    cdef double prev_back_t_aux = 0  # every it: prev_back_t = prev_back_t_aux
    cdef int nba
    cdef int nb

    for i in workload[proc_a]:
        influencer = curr_state[i]
        if influencer == -1:
            prev_back_t_aux = all_timestamps[proc_a][i] # found a background ts
        else:
            nba = Alpha_ab[proc_a][influencer]
            nb = sum_b[influencer]
            Alpha_ab[proc_a][influencer] -= 1
            sum_b[influencer] -= 1
            fptree.set_value(influencer,
                             fptree.get_value(influencer) - \
                             inc(n_proc, nb, nba, alpha_prior, -1))

        new_influencer = metropolis_walk_step(proc_a, i, prev_back_t,
                                              all_timestamps, curr_state,
                                              Alpha_ab, sum_b, alpha_prior,
                                              mu_rates, beta_rates, fptree)
        if new_influencer != -1:
            if Alpha_ab[proc_a].count(new_influencer) == 0:
                Alpha_ab[proc_a][new_influencer] = 0
            nba = Alpha_ab[proc_a][new_influencer]
            nb = sum_b[new_influencer]
            Alpha_ab[proc_a][new_influencer] += 1
            sum_b[new_influencer] += 1
            fptree.set_value(new_influencer,
                             fptree.get_value(new_influencer) - \
                             inc(n_proc, nb, nba, alpha_prior, -1))

        curr_state[i] = new_influencer
        prev_back_t = prev_back_t_aux


cdef void update_mu_rate(int proc_a, vector[double] &timestamps_proc_a,
                         vector[int] &curr_state_proc_a,
                         double count_background, double[::1] mu_rates) nogil:
    cdef int n_events = timestamps_proc_a.size()
    cdef double T = timestamps_proc_a[n_events-1]
    cdef double rate
    if T == 0:
        rate = 0
    else:
        rate = count_background / T
    mu_rates[proc_a] = rate


cdef void update_beta_rate(int proc_a,
                           map[int, vector[double]] &all_timestamps,
                           map[int, vector[int]] &curr_state_all,
                           map[int, map[int, int]] &Alpha_ba,
                           double alpha_prior, int[::1] sum_b,
                           double[::1] beta_rates) nogil:

    cdef vector[double] all_deltas
    cdef int n_proc = beta_rates.shape[0]
    cdef int proc_b, i
    cdef double ti, tp
    cdef double max_ti = 0
    cdef int n_elements = 0
    for proc_b in range(n_proc):
        for i in range(<int>all_timestamps[proc_b].size()):
            if curr_state_all[proc_b][i] == proc_a:
                ti = all_timestamps[proc_b][i]
                if ti > max_ti:
                    max_ti = ti
                tp = find_previous(all_timestamps[proc_a], ti)
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


def mstep(map[int, vector[double]] &all_timestamps,
          map[int, vector[int]] &curr_state, double alpha_prior):

    cdef int n_proc = all_timestamps.size()

    cdef int a, b
    cdef map[int, map[int, int]] Alpha_ab
    cdef int[::1] num_background = np.zeros(n_proc, dtype='i')
    cdef int[::1] sum_b = np.zeros(n_proc, dtype='i')
    for a in range(n_proc):
        Alpha_ab[a] = map[int, int]()
        for b in curr_state[a]:
            if b != -1:
                if Alpha_ab[a].count(b) == 0:
                    Alpha_ab[a][b] = 0
                Alpha_ab[a][b] += 1
                sum_b[b] += 1
            else:
                num_background[a] += 1

    cdef double[::1] mu_rates = np.zeros(n_proc, dtype='d', order='C')
    cdef double[::1] beta_rates = np.zeros(n_proc, dtype='d', order='C')
    for a in parallel.prange(n_proc, schedule='static', nogil=True):
        update_mu_rate(a, all_timestamps[a], curr_state[a], num_background[a],
                       mu_rates)
    for a in parallel.prange(n_proc, schedule='static', nogil=True):
        update_beta_rate(a, all_timestamps, curr_state, Alpha_ab, alpha_prior,
                         sum_b, beta_rates)
    return Alpha_ab, num_background, sum_b, mu_rates, beta_rates


cdef int estep(map[int, vector[double]] &all_timestamps,
               map[int, vector[int]] &curr_state, double[::1] mu_rates,
               double[::1] beta_rates, double alpha_prior, int[::1] sum_b,
               FPTree fptree, map[int, vector[int]] &workload) nogil:

    cdef int n_proc = all_timestamps.size()
    cdef int a, b
    cdef map[int, map[int, int]] Alpha_ab
    for a in range(n_proc):
        Alpha_ab[a] = map[int, int]()
        for b in curr_state[a]:
            if b != -1:
                if Alpha_ab[a].count(b) == 0:
                    Alpha_ab[a][b] = 0
                Alpha_ab[a][b] += 1
                sum_b[b] += 1

    cdef pair[int, int] b_pair
    for a in range(n_proc):
        fptree.reset()
        for b_pair in Alpha_ab[a]:
            b = b_pair.first
            fptree.set_value(b, (Alpha_ab[a][b] + alpha_prior) / \
                                 (sum_b[b] + n_proc * alpha_prior))
        sample_alpha(a, all_timestamps, curr_state[a], Alpha_ab, sum_b,
                     mu_rates, beta_rates, alpha_prior, fptree, workload)


def fit(dict all_timestamps, double alpha_prior, int n_iter, int burn_in,
        dict start_state, dict workload_dict):

    cdef int n_proc = len(all_timestamps)

    cdef map[int, vector[int]] curr_state
    cdef map[int, map[int, vector[int]]] workload
    cdef map[int, vector[double]] all_timestamps_map

    cdef int a
    for a in range(n_proc):
        all_timestamps_map[a] = all_timestamps[a]
        curr_state[a] = np.asanyarray(start_state[a], dtype='i')

    cdef int n_jobs = len(workload)
    cdef int job
    for job in range(n_jobs):
        workload[job] = map[int, vector[int]]()
        for a in range(n_proc):
            workload[job][a] = np.asanyarray(workload_dict[job][a], dtype='i')

    cdef map[int, map[int, int]] Alpha_ab
    cdef int[::1] num_background
    cdef int[::1] sum_b
    cdef double[::1] mu_rates
    cdef double[::1] beta_rates
    cdef int[:,::1] SB = np.zeros(shape=(n_jobs, n_proc), dtype='i', order='C')

    cdef int i
    for i in range(n_iter):
        Alpha_ab, num_background, mu_rates, beta_rates = \
            mstep(all_timestamps_map, curr_state, alpha_prior)
        for job in parallel.prange(n_jobs, schedule='static', nogil=True):
            with gil:
                # the gil will be released as soon as the func starts
                # we need it just to create the FPTree obj
                estep(all_timestamps_map, curr_state, mu_rates, beta_rates,
                      alpha_prior, SB[job], FPTree(), workload[job])

    # One final mstep to get parameters
    Alpha_ab, num_background, mu_rates, beta_rates = \
        mstep(all_timestamps_map, curr_state, alpha_prior)

    cdef pair[int, int] b_pair
    cdef int b
    cdef map[int, map[int, int]] Alpha_ba
    for a in range(n_proc):
        if Alpha_ab.count(a) == 0:
            continue
        for b_pair in Alpha_ab[a]:
            b = b_pair.first
            if Alpha_ba.count(b) == 0:
                Alpha_ba[b] = map[int, int]()
            Alpha_ba[b][a] += Alpha_ab[a][b]

    return Alpha_ba, np.asarray(mu_rates), np.asarray(beta_rates), \
        np.asarray(num_background), curr_state
