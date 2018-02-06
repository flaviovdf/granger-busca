# -*- coding: utf8

from gb.sampler import fit as gibbs_fit
from gb.mhw import fit as metropolis_fit

from scipy import sparse as sp

import numpy as np
import os
import time


class GrangerBusca(object):

    def __init__(self, alpha_p, num_iter, burn_in, metropolis=True,
                 n_jobs=None):
        self.alpha_p = alpha_p
        self.num_iter = num_iter
        self.burn_in = burn_in
        self.metropolis = metropolis
        if n_jobs is None:
            n_jobs = os.cpu_count() - 1
        self.n_jobs = n_jobs

        self.timestamps = None
        self.time_range = None
        self.num_stamps = None
        self.training_time = None
        self.Alpha_ = None
        self.mu_ = None
        self.beta_ = None
        self.back_ = None
        self.curr_state_ = None

    def _init_timestamps(self, timestamps):
        min_all = np.float('inf')
        max_all = 0
        copy_of_timestamps = {}
        n_proc = len(timestamps)
        for i in range(n_proc):
            assert (np.asanyarray(timestamps[i]) >= 0).all()
            min_all = min(min_all, np.min(timestamps[i]))
            max_all = max(max_all, np.max(timestamps[i]))
            tis = np.sort(timestamps[i])
            copy_of_timestamps[i] = np.asanyarray(tis, dtype='d', order='C')

        timestamps = copy_of_timestamps
        max_all -= min_all
        num_stamps = 0
        curr_state_ = {}
        for i in range(n_proc):
            timestamps[i] -= min_all
            num_stamps += len(timestamps[i])
            curr_state_[i] = np.random.randint(-1, n_proc, len(timestamps[i]))

        self.curr_state_ = curr_state_
        self.timestamps = timestamps
        self.time_range = max_all
        self.num_stamps = num_stamps

    def parallel_metropolis(self):
        workload = np.zeros(shape=(self.num_stamps, 2), dtype='i')
        slice_size = len(workload) // self.n_jobs
        i = 0
        for a in self.timestamps:
            for idx in range(len(self.timestamps[a])):
                workload[i][0] = a
                workload[i][1] = idx
                i += 1

        np.random.shuffle(workload)
        workload_dicts = {}
        st = 0
        ed = slice_size
        for job in range(self.n_jobs):
            workload_dict = {}
            for a in range(len(self.timestamps)):
                workload_dict[a] = []
            workload_dicts[job] = workload_dict

            if job == self.n_jobs - 1:
                ed = len(workload)
            for slice_ in workload[st:ed]:
                a = slice_[0]
                idx = slice_[1]
                workload_dict[a].append(idx)
            st = ed
            ed = ed + slice_size

        return metropolis_fit(self.timestamps, self.alpha_p, self.num_iter, 0,
                              self.curr_state_, workload_dicts)

    def fit(self, timestamps):
        self._init_timestamps(timestamps)

        now = time.time()
        if self.metropolis:
            R = self.parallel_metropolis()
            n = 1
        else:
            R = gibbs_fit(self.timestamps, self.alpha_p, self.num_iter,
                          self.burn_in)
            n = R[-1]
        dt = time.time() - now

        self.Alpha_ = R[0]
        self.mu_ = R[1]
        self.beta_ = R[2]
        self.back_ = R[3]
        self.curr_state_ = R[4]
        self.training_time = dt

        vals = []
        rows = []
        cols = []
        for row in self.Alpha_:
            for col in self.Alpha_[row]:
                rows.append(row)
                cols.append(col)
                vals.append(self.Alpha_[row][col] / n)
        self.Alpha_ = sp.csr_matrix((vals, (rows, cols)), dtype='i')
        self.mu_ = self.mu_ / n
        self.beta_ = self.beta_ / n
