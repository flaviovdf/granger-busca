# -*- coding: utf8
# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: wraparound=False

import numpy as np

from gb.collections.fptree cimport FPTree
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


cdef inline double dirmulti_posterior(map[int, map[int, int]] &Alpha_ab,
                                      int[::1] sum_b, int proc_b, int proc_a,
                                      double prior) nogil:
    '''
    Computes the posterior for the counts.

    Parameters
    ----------
    Alpha_ab: libstd map of ints -> map[int, int]
        Counts for each time b cross-excited a
    sum_b: int array
        Sum for a given row in Alpha_ab
    proc_b: int
        ID for process b
    proc_a: int
        ID for process a
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


cdef inline double busca_rate(double alpha_ba, double tp, double tpp,
                              double beta_proc_b) nogil:
    '''Computes the busca rate given alpha and dt'''
    cdef double rate = alpha_ba / (beta_proc_b/E + tp - tpp)
    return rate


cdef inline void busca_probability(int proc_b, int proc_a,
                                   map[int, map[int, int]] &Alpha_ab,
                                   int[::1] sum_b, double alpha_prior,
                                   double[::1] beta_rates,
                                   double tp, double tpp,
                                   double *a_ba, double *rate) nogil:

    a_ba[0] = dirmulti_posterior(Alpha_ab, sum_b, proc_b, proc_a, alpha_prior)
    rate[0] = busca_rate(a_ba[0], tp, tpp, beta_rates[proc_b])


cdef inline int metropolis_walk_step(int proc_a, int curr_influencer_b,
                                     double t, double[::1] last_t,
                                     double dt_background,
                                     map[int, map[int, int]] &Alpha_ab,
                                     int[::1] sum_b, double alpha_prior,
                                     double[::1] mu_rates,
                                     double[::1] beta_rates,
                                     FPTree fptree) nogil:

    cdef double mu_rate = mu_rates[proc_a]
    cdef double mu_prob = mu_rate * dt_background * exp(-mu_rate*dt_background)

    cdef int candidate = fptree.sample(rand())
    cdef double q_c = fptree.get_value(candidate)
    cdef double a_ca = 0
    cdef double p_c = 0
    busca_probability(candidate, proc_a, Alpha_ab, sum_b, alpha_prior,
                      beta_rates, t, last_t[candidate], &a_ca, &p_c)

    cdef double q_b = 0
    cdef double a_ba = 0
    cdef double p_b = 0
    if curr_influencer_b != -1:
        q_b = fptree.get_value(curr_influencer_b)
        busca_probability(curr_influencer_b, proc_a, Alpha_ab, sum_b,
                          alpha_prior, beta_rates, t,
                          last_t[curr_influencer_b], &a_ba, &p_b)

    cdef double busca_rate_c = p_c / a_ca
    if sample_background(mu_prob / (mu_prob + busca_rate_c)):
        return -1
    elif curr_influencer_b == -1:
        return candidate
    else:
        if rand() < min(1, (p_c * q_b) / (p_b * q_c)):
            return candidate
        else:
            return curr_influencer_b


cdef int sample_alpha(int proc_a, double t, int influencer,
                      double[::1] last_t, int[::1] num_background,
                      double dt_background,
                      map[int, map[int, int]] &Alpha_ab, int[::1] sum_b,
                      double[::1] mu_rates, double[::1] beta_rates,
                      double alpha_prior, FPTree fptree) nogil:

    cdef int new_influencer
    if influencer == -1:
        num_background[proc_a] -= 1
    else:
        Alpha_ab[proc_a][influencer] -= 1
        sum_b[influencer] -= 1

    new_influencer = metropolis_walk_step(proc_a, influencer, t, last_t,
                                          dt_background, Alpha_ab, sum_b,
                                          alpha_prior, mu_rates, beta_rates,
                                          fptree)
    if new_influencer == -1:
        num_background[proc_a] += 1
    else:
        if Alpha_ab[proc_a].count(new_influencer) == 0:
            Alpha_ab[proc_a][new_influencer] = 0
        Alpha_ab[proc_a][new_influencer] += 1
        sum_b[new_influencer] += 1
        fptree.set_value(new_influencer,
                         fptree.get_value(new_influencer) + 1)

    if influencer != -1:
        fptree.set_value(influencer, fptree.get_value(influencer) - 1)
    return new_influencer


cdef void update_mu_rate(double[::1] last_t_background,
                         double count_background, double[::1] mu_rates) nogil:

    cdef int a
    for a in range(last_t_background.shape[0]):
        if last_t_background[a] <= 0:
            mu_rates[a] = 0
        else:
            mu_rates[a] = count_background[a] / last_t_background[a]


cdef void update_beta_rate(double[::1] timestamps, int[::1] process_id,
                           int[::1] curr_state, double[::1] last_t,
                           map[int, map[int, int]] &Alpha_ab,
                           double alpha_prior, int[::1] sum_b,
                           double[::1] beta_rates) nogil:

    cdef map[int, vector[double]] all_deltas
    cdef int n_proc = beta_rates.shape[0]
    cdef int proc_a, proc_b, i
    cdef double t, dt

    for proc_a in range(last_t.shape[0]):
        last_t[proc_a] = 0

    for i in range(timestamps.shape[0]):
        t = timestamps[i]
        proc_a = process_id[i]
        proc_b = curr_state[i]
        if proc_b == -1:
            continue

        dt = t - last_t[proc_b]
        if all_deltas.count(proc_a) == 0:
            all_deltas[proc_a] = vector[double]()
        all_deltas[proc_a].push_back(dt)
        last_t[proc_a] = t

    cdef int n
    for proc_a in all_deltas:
        n = all_deltas[proc_a]
        if n >= 1:
            stdsort(all_deltas[proc_a].begin(), all_deltas[proc_a].end())
            beta_rates[proc_a] = all_deltas[proc_a][n // 2]
            if n % 2 == 0:
                beta_rates[proc_a] += all_deltas[proc_a][(n // 2)-1]
                beta_rates[proc_a] = beta_rates[proc_a] / 2
        else:
            beta_rates[proc_a] = last_t[proc_a]


cdef void sampleone(double[::1] timestamps, int[::1] process_id,
                    double[::1] last_t, double[::1] last_t_background,
                    int[::1] curr_state, int[::1] num_background,
                    double[::1] mu_rates, double[::1] beta_rates,
                    map[int, map[int, int]] &Alpha_ab, double alpha_prior,
                    int[::1] sum_b, vector[FPTree] &fptrees,
                    double[::1] last_t, double[::1] last_t_background) nogil:

    printf("[logger]\t Learning mu.\n")
    update_mu_rate(timestamps, process_id, curr_state, num_background,
                   mu_rates)

    printf("[logger]\t Learning beta.\n")
    update_beta_rate(timestamps, process_id, curr_state, Alpha_ab, alpha_prior,
                     sum_b, beta_rates)

    printf("[logger]\t Sampling Alpha.\n")
    cdef int i, a, b, new_b
    cdef double t

    for a in range(last_t.shape[0]):
        last_t[a] = 0
        last_t_background[a] = 0

    for i in range(timestamps.shape[0]):
        t = timestamps[i]
        a = process_id[i]
        new_b = sample_alpha(a, t, curr_state[i], num_background,
                             t - last_t_background[a], Alpha_ab, sum_b,
                             mu_rates, beta_rates, alpha_prior, fptrees[a])
        curr_state[i] = new_b
        if curr_state[i] == -1:
            last_t_background[i] = t
        last_t[a] = t


cdef int cfit(double[::1] timestamps, int[::1] process_id, int[::1] curr_state,
              int[::1] num_background, double[::1] mu_rates,
              double[::1] beta_rates, map[int, map[int, int]] &Alpha_ab,
              double alpha_prior, int[::1] sum_b, FPTree fptree,
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
        sampleone(timestamps, process_id, curr_state, num_background, mu_rates,
                  beta_rates, Alpha_ab, alpha_prior, sum_b, fptree)
        if iteration >= burn_in:
            printf("[logger]\t Averaging after burn in...\n")
            num_good += 1
            for a in range(mu_rates.shape[0]):
                mu_rates_final[a] += mu_rates[a]
                beta_rates_final[a] += beta_rates[a]
                for b in Alpha_ab[a]:
                    if Alpha_ba_final.count(b) == 0:
                        Alpha_ba_final[b] = map[int, int]()
                    if Alpha_ba_final[b].count(a) == 0:
                        Alpha_ba_final[b][a] = 0
                    Alpha_ba_final[b][a] += Alpha_ab[a][b]
        printf("[logger] Iter done!\n")
    return num_good

def fit(double[::1] timestamps, int[::1] process_id, int n_proc,
        double alpha_prior, int n_iter, int burn_in):

    cdef int n_stamps = timestamps.shape[0]
    cdef int[::1] curr_state = np.zeros(n_stamps, dtype='i', order='C')
    cdef int[::1] sum_b = np.zeros(n_proc, dtype='i', order='C')
    cdef int[::1] num_background = np.zeros(n_proc, dtype='i', order='C')

    cdef map[int, map[int, int]] Alpha_ab
    cdef int i, a, b
    for i in range(n_stamps):
        a = process_id[i]
        b = np.random.randint(-1, n_proc)
        curr_state[i] = b
        if Alpha_ab.count(a) == 0:
            Alpha_ab[a] = map[int, int]()
        if b == -1:
            num_background[a] += 1
        else:
            if Alpha_ab[a].count(b) == 0:
                Alpha_ab[a][b] = 0
            Alpha_ab[a][b] += 1
            sum_b[b] += 1

    cdef double[::1] mu_rates = np.zeros(n_proc, dtype='d', order='C')
    cdef double[::1] beta_rates = np.zeros(n_proc, dtype='d', order='C')

    cdef double[::1] mu_rates_final = np.zeros(n_proc, dtype='d', order='C')
    cdef double[::1] beta_rates_final = np.zeros(n_proc, dtype='d', order='C')
    cdef map[int, map[int, int]] Alpha_ba_final

    cdef int n_good = cfit(timestamps, process_id, curr_state, num_background,
                           mu_rates, beta_rates, Alpha_ab, alpha_prior,
                           sum_b, fptree, search_lower, mu_rates_final,
                           beta_rates_final, Alpha_ba_final, n_iter, burn_in)

    return Alpha_ba_final, np.asarray(mu_rates_final), \
        np.asarray(beta_rates_final), np.asarray(num_background), curr_state, \
        n_good
