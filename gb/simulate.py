# -*- coding: utf8

import numpy as np


class GrangeBuscaSimulator(object):

    def __init__(self, mu_rates, Alpha_ba):
        self.mu_rates = np.asanyarray(mu_rates)
        self.Alpha_ba = np.asanyarray(Alpha_ba)
        self.past = [[] for i in range(self.Alpha_ba.shape[0])]
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
                tpp_idx = np.searchsorted(self.past[proc_b], tp)
                if tpp_idx >= len(self.past[proc_b]):
                    tpp_idx = -1
                tpp = self.past[proc_b][tpp_idx]
                if tpp >= tp:
                    continue

                busca_rate = self.Alpha_ba[proc_b, proc_a]
                busca_rate /= (tp - tpp)
                lambdas_t[proc_a] += busca_rate
        return lambdas_t

    def simulate(self, forward):
        t = self.t
        max_time = t + forward
        while t < max_time:
            lambdas_t = self.total_intensity(t)
            sum_lambdas_t = lambdas_t.cumsum()
            dt = np.random.exponential(1.0 / sum_lambdas_t[-1])
            t = t + dt
            if t > max_time:
                break

            i = 0
            u = np.random.rand() * sum_lambdas_t[-1]
            while i < self.Alpha_ba.shape[0]:
                if sum_lambdas_t[i] >= u:
                    break
                i += 1
            self.past[i].append(t)
        self.t = t
        return self.past
