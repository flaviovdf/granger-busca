# -*- coding: utf8

from bisect import bisect

import numpy as np


class GrangeBuscaSimulator(object):

    def __init__(self, mu_rates, Alpha_ba, Beta_ba=None, thinning=False):
        self.mu_rates = np.asanyarray(mu_rates)
        self.Alpha_ba = np.asanyarray(Alpha_ba)
        if Beta_ba is not None:
            self.Beta_ba = np.asanyarray(Beta_ba)
        else:
            self.Beta_ba = np.ones(shape=self.Alpha_ba.shape)
        self.past = [[] for i in range(self.Alpha_ba.shape[0])]
        self.integrals = [[] for i in range(self.Alpha_ba.shape[0])]
        self.upper_bound = 0.0
        for proc_a in range(self.Alpha_ba.shape[0]):
            self.upper_bound += self.mu_rates[proc_a]
            for proc_b in range(self.Alpha_ba.shape[0]):
                self.upper_bound += self.Alpha_ba[proc_b, proc_a] / \
                    self.Beta_ba[proc_b, proc_a]
        self.thinning = thinning
        self.t = 0

    def total_intensity(self, t):
        lambdas_t = np.zeros(self.mu_rates.shape[0], dtype='d')
        for proc_a in range(self.Alpha_ba.shape[0]):
            lambdas_t[proc_a] = self.mu_rates[proc_a]
            if len(self.past[proc_a]) == 0:
                continue

            tp = self.past[proc_a][-1]
            assert tp <= t
            for proc_b in range(self.Alpha_ba.shape[0]):
                if len(self.past[proc_b]) == 0:
                    continue

                tpp_idx = bisect(self.past[proc_b], tp)
                if tpp_idx == len(self.past[proc_b]):
                    tpp_idx -= 1
                tpp = self.past[proc_b][tpp_idx]
                while tpp >= tp and tpp_idx > 0:
                    tpp_idx -= 1
                    tpp = self.past[proc_b][tpp_idx]
                if tpp >= tp:
                    continue
                busca_rate = self.Alpha_ba[proc_b, proc_a]
                busca_rate /= (self.Beta_ba[proc_b, proc_a] + tp - tpp)
                lambdas_t[proc_a] += busca_rate
        return lambdas_t

    def simulate(self, forward):
        t = self.t
        max_time = t + forward
        while t < max_time:
            lambdas_t = self.total_intensity(t)
            sum_lambdas_t = lambdas_t.cumsum()
            if self.thinning:
                dt = np.random.exponential(1.0 / self.upper_bound)
            else:
                dt = np.random.exponential(1.0 / sum_lambdas_t[-1])

            t = t + dt
            if t > max_time:
                break

            if self.thinning:
                if np.random.rand() < (sum_lambdas_t[-1] / self.upper_bound):
                    continue

            i = 0
            u = np.random.rand() * sum_lambdas_t[-1]
            while i < self.Alpha_ba.shape[0]:
                if sum_lambdas_t[i] >= u:
                    break
                i += 1
            if len(self.past[i]) > 1:
                self.integrals[i].append(lambdas_t[i] * (t - self.past[i][-1]))
            self.past[i].append(t)
        self.t = t
        return self.past
