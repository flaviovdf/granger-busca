# -*- coding: utf8
# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: wraparound=False

import numpy as np

from gb.randomkit.random cimport rand
from gb.randomkit.random cimport gamma as rgamma

from libc.stdio cimport printf

from libcpp.algorithm cimport sort as stdsort
from libcpp.unordered_map cimport unordered_map as map
from libcpp.map cimport pair
from libcpp.vector cimport vector


cdef double E = 2.718281828459045


cdef extern from 'math.h':
    double exp(double) nogil


cdef inline double dirmulti_posterior(map[int, map[int, int]] &Alpha_ba,
                                      int[::1] sum_b, int proc_b, int proc_a,
                                      double prior) nogil:
    '''
    Computes the posterior for the counts.

    Parameters
    ----------
    Alpha_ba: libstd map of ints -> map[int, int]
        Counts for each time b cross-excited a
    sum_b: int array
        Sum for a given row in Alpha_ba
    proc_b: int
        ID for process b
    proc_a: int
        ID for process a
    prior: double
        Prior for the process
    '''
    cdef int n = sum_b.shape[0]
    cdef int nba
    if Alpha_ba.count(proc_b) == 0:
        nba = 0
    elif Alpha_ba[proc_b].count(proc_a) == 0:
        nba = 0
    else:
        nba = Alpha_ba[proc_b][proc_a]
    return (nba + prior) / (sum_b[proc_b] + n * prior)


cdef inline int sample_background(double prob_background) nogil:
    '''Simply samples a random number and checks if < prob_background'''
    return rand() < prob_background


cdef inline int searchsorted(vector[double] &array, double value, int lower,
                             int left) nogil:
    '''
    Finds the first element in the array where the given is OR should have been
    in the given array. This is simply a binary search, but if the element is
    not found we return the index where it should have been at (to the left,
    or right given the parameter).

    Parameters
    ----------
    array: vector of doubles
    value: double to look for
    lower: int to start search from [lower, n)
    left:  int to determine if return the left or right index when the search
           fails
    '''

    cdef int upper = array.size() - 1  # closed interval
    cdef int half = 0
    cdef int idx = -1

    while upper >= lower:
        half = lower + ((upper - lower) // 2)
        if value == array[half]:
            idx = half
            break
        elif value > array[half]:
            lower = half + 1
        else:
            upper = half - 1

    if idx == -1:  # Element not found, return where it should be
        if left:
            idx = upper # counter intuitive but upper is now < lower
        else:
            idx = lower

    return idx


cdef inline double find_previous(vector[double] &timestamps, double t,
                                 int lower, int *store_idx) nogil:
    '''
    Finds the closest timestamp with value less than the one informed.

    Parameters
    ----------
    timestamps: vector[double]
        the process to search in
    t: double
        the reference value to find tp < t such that t - tp is minimal
    lower: int
        where to start the search from [0, n_stamps)
    '''

    cdef int tp_idx = searchsorted(timestamps, t, lower, 1)

    if tp_idx == lower - 1: # t is less than all timestamps, fall back to lower
        tp_idx = lower
    store_idx[0] = tp_idx

    cdef double tp = timestamps[tp_idx]
    if tp >= t: # can occur when no timestamp < t exists and lower == 0.
        tp = 0
    return tp


cdef inline double busca_probability(int i, int proc_a, int proc_b,
                                     map[int, vector[double]] &all_timestamps,
                                     double alpha_ba, double beta_proc_b,
                                     map[int, int] &lower_stamps) nogil:

    cdef double t = all_timestamps[proc_a][i]
    cdef double tp
    cdef double tpp
    cdef int store_idx
    cdef int lower = 0
    if lower_stamps.count(proc_b) > 0:
        lower = lower_stamps[proc_b]

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
            tpp = find_previous(all_timestamps[proc_b], tp, lower, &store_idx)
            lower_stamps[proc_b] = store_idx
    else:
        tpp = 0

    cdef double rate = alpha_ba / (beta_proc_b/E + tp - tpp)
    return rate


cdef inline void populate_busca_vect(int i, int proc_a,
                                     map[int, vector[double]] &all_timestamps,
                                     map[int, map[int, int]] &Alpha_ba,
                                     int[::1] sum_b, double alpha_prior,
                                     double[::1] beta_rates,
                                     vector[double] &prob_b,
                                     map[int, int] &lower_stamps) nogil:

    cdef double a_ba
    cdef int n_proc = <int>all_timestamps.size()
    cdef int proc_b
    cdef int lower

    for proc_b in range(n_proc):
        a_ba = dirmulti_posterior(Alpha_ba, sum_b, proc_b, proc_a, alpha_prior)
        prob_b[proc_b] = busca_probability(i, proc_a, proc_b, all_timestamps,
                                           a_ba, beta_rates[proc_b],
                                           lower_stamps)
        if proc_b > 0:
            prob_b[proc_b] += prob_b[proc_b-1]


cdef inline int sample_one_timeindex(int proc_a, int i, double prev_back_t,
                                     int[::1] num_background,
                                     map[int, vector[double]] &all_timestamps,
                                     vector[int] &curr_state_proc_a,
                                     map[int, map[int, int]] &Alpha_ba,
                                     int[::1] sum_b, double alpha_prior,
                                     double[::1] mu_rates,
                                     double[::1] beta_rates,
                                     map[int, map[int, int]] &search_lower,
                                     vector[double] &prob_b) nogil:

    cdef double mu_rate = mu_rates[proc_a]
    cdef double ti = all_timestamps[proc_a][i]
    cdef double tp = prev_back_t
    cdef double dt = ti - tp
    cdef double mu_prob = mu_rate * dt * exp(-mu_rate*dt)

    populate_busca_vect(i, proc_a, all_timestamps, Alpha_ba, sum_b,
                        alpha_prior, beta_rates, prob_b, search_lower[proc_a])

    cdef int n_proc = <int>all_timestamps.size()
    if sample_background(mu_prob / (mu_prob + prob_b[n_proc-1])):
        return -1
    else:
        return searchsorted(prob_b, prob_b[n_proc-1] * rand(), 0, 0)


cdef void sample_alpha(int proc_a, map[int, vector[double]] &all_timestamps,
                       vector[int] &curr_state, int[::1] num_background,
                       map[int, map[int, int]] &Alpha_ba, int[::1] sum_b,
                       double[::1] mu_rates, double[::1] beta_rates,
                       double alpha_prior, vector[double] &prob_b,
                       map[int, map[int, int]] &search_lower) nogil:

    cdef int i
    cdef int influencer
    cdef int new_influencer
    cdef double prev_back_t = 0      # stores last known background time stamp
    cdef double prev_back_t_aux = 0  # every it: prev_back_t = prev_back_t_aux
    for i in range(<int>all_timestamps[proc_a].size()):
        influencer = curr_state[i]
        if influencer == -1:
            num_background[proc_a] -= 1
            prev_back_t_aux = all_timestamps[proc_a][i] # found a background ts
        else:
            Alpha_ba[influencer][proc_a] -= 1
            sum_b[influencer] -= 1

        new_influencer = sample_one_timeindex(proc_a, i, prev_back_t,
                                              num_background, all_timestamps,
                                              curr_state, Alpha_ba, sum_b,
                                              alpha_prior, mu_rates,
                                              beta_rates, search_lower, prob_b)
        if new_influencer == -1:
            num_background[proc_a] += 1
        else:
            if Alpha_ba.count(new_influencer) == 0:
                Alpha_ba[new_influencer] = map[int, int]()
            if Alpha_ba[new_influencer].count(proc_a) == 0:
                Alpha_ba[new_influencer][proc_a] = 0
            Alpha_ba[new_influencer][proc_a] += 1
            sum_b[new_influencer] += 1

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
    cdef int proc_b, i, lower, store_idx
    cdef double ti, tp
    cdef double max_ti = 0
    cdef int n_elements = 0
    for proc_b in range(n_proc):
        lower = 0
        for i in range(<int>all_timestamps[proc_b].size()):
            if curr_state_all[proc_b][i] == proc_a:
                ti = all_timestamps[proc_b][i]
                if ti > max_ti:
                    max_ti = ti
                tp = find_previous(all_timestamps[proc_a], ti, lower,
                                   &store_idx)
                lower = store_idx
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


cdef void sampleone(map[int, vector[double]] &all_timestamps,
                    map[int, vector[int]] &curr_state, int[::1] num_background,
                    double[::1] mu_rates, double[::1] beta_rates,
                    map[int, map[int, int]] &Alpha_ba, double alpha_prior,
                    int[::1] sum_b, vector[double] &prob_b,
                    map[int, map[int, int]] &search_lower) nogil:

    cdef int n_proc = all_timestamps.size()
    cdef int a

    printf("[logger]\t Learning mu.\n")
    for a in range(n_proc):
        update_mu_rate(a, all_timestamps[a], curr_state[a],
                       num_background[a], mu_rates)
    printf("[logger]\t Learning beta.\n")
    for a in range(n_proc):
        update_beta_rate(a, all_timestamps, curr_state, Alpha_ba, alpha_prior,
                         sum_b, beta_rates)
        search_lower[a].clear()

    printf("[logger]\t Sampling Alpha.\n")
    for a in range(n_proc):
        sample_alpha(a, all_timestamps, curr_state[a], num_background,
                     Alpha_ba, sum_b, mu_rates, beta_rates, alpha_prior,
                     prob_b, search_lower)


cdef int cfit(map[int, vector[double]] &all_timestamps,
              map[int, vector[int]] &curr_state, int[::1] num_background,
              double[::1] mu_rates, double[::1] beta_rates,
              map[int, map[int, int]] &Alpha_ba, double alpha_prior,
              int[::1] sum_b, vector[double] &prob_b,
              map[int, map[int, int]] &search_lower,
              double[::1] mu_rates_final, double[::1] beta_rates_final,
              map[int, map[int, int]] &Alpha_ba_final, int n_iter,
              int burn_in) nogil:

    printf("[logger] Sampler is starting\n")
    printf("[logger]\t n_proc=%ld\n", mu_rates.shape[0])
    printf("[logger]\t alpha_prior=%lf\n", alpha_prior)
    printf("\n")

    cdef int iteration, b, a
    cdef int num_good = 0
    for iteration in range(n_iter):
        printf("[logger] Iter=%d. Sampling...\n", iteration)
        sampleone(all_timestamps, curr_state, num_background, mu_rates,
                  beta_rates, Alpha_ba, alpha_prior, sum_b, prob_b,
                  search_lower)
        if iteration >= burn_in:
            printf("[logger]\t Averaging after burn in...\n")
            num_good += 1
            for b in range(mu_rates.shape[0]):
                mu_rates_final[b] += mu_rates[b]
                beta_rates_final[b] += beta_rates[b]
                for a in range(mu_rates.shape[0]):
                    if Alpha_ba.count(b) == 0:
                        continue
                    if Alpha_ba[b].count(a) == 0:
                        continue
                    if Alpha_ba_final.count(b) == 0:
                        Alpha_ba_final[b] = map[int, int]()
                    if Alpha_ba_final[b].count(a) == 0:
                        Alpha_ba_final[b][a] = 0
                    Alpha_ba_final[b][a] += Alpha_ba[b][a]
        printf("[logger] Iter done!\n")
    return num_good

def fit(dict all_timestamps, double alpha_prior, int n_iter, int burn_in):

    cdef int n_proc = len(all_timestamps)

    cdef map[int, vector[int]] curr_state
    cdef map[int, map[int, int]] Alpha_ba
    cdef map[int, vector[double]] all_timestamps_map
    cdef map[int, map[int, int]] search_lower

    cdef int[::1] sum_b = np.zeros(n_proc, dtype='i', order='C')
    cdef int[::1] num_background = np.zeros(n_proc, dtype='i', order='C')

    cdef int a, b
    for a in range(n_proc):
        search_lower[a] = map[int, int]()
        all_timestamps_map[a] = all_timestamps[a]
        curr_state[a] = np.random.randint(-1, n_proc, len(all_timestamps[a]))
        for b in curr_state[a]:
            if b == -1:
                num_background[a] += 1
            else:
                if Alpha_ba.count(b) == 0:
                    Alpha_ba[b] = map[int, int]()
                if Alpha_ba[b].count(a) == 0:
                    Alpha_ba[b][a] = 0
                Alpha_ba[b][a] += 1
                sum_b[b] += 1

    cdef double[::1] mu_rates = np.zeros(n_proc, dtype='d', order='C')
    cdef double[::1] beta_rates = np.zeros(n_proc, dtype='d', order='C')
    cdef vector[double] prob_b = np.zeros(n_proc, dtype='d', order='C')

    cdef double[::1] mu_rates_final = np.zeros(n_proc, dtype='d', order='C')
    cdef double[::1] beta_rates_final = np.zeros(n_proc, dtype='d', order='C')
    cdef map[int, map[int, int]] Alpha_ba_final

    cdef int n_good = cfit(all_timestamps_map, curr_state, num_background,
                           mu_rates, beta_rates, Alpha_ba, alpha_prior,
                           sum_b, prob_b, search_lower, mu_rates_final,
                           beta_rates_final, Alpha_ba_final, n_iter, burn_in)

    return Alpha_ba_final, np.asarray(mu_rates_final), \
        np.asarray(beta_rates_final), np.asarray(num_background), curr_state, \
        n_good
