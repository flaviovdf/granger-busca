# -*- coding: utf8

from gb.fit import fit as cyfit

from scipy import sparse as sp

import numpy as np
import time


class GrangerBusca(object):

    def __init__(self, alpha_p, num_iter, burn_in, metropolis=True):
        self.alpha_p = alpha_p
        self.num_iter = num_iter
        self.burn_in = burn_in
        self.metropolis = metropolis

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
        for i in range(n_proc):
            timestamps[i] -= min_all

        self.n_proc = n_proc
        self.timestamps = timestamps
        self.time_range = max_all

    def fit(self, timestamps):
        self._init_timestamps(timestamps)

        now = time.time()
        self.Alpha_, self.mu_, self.beta_, self.back_, self.curr_state_ = \
            cyfit(self.timestamps, self.alpha_p, self.num_iter,
                  int(self.metropolis))
        dt = time.time() - now
        self.training_time = dt

        vals = []
        rows = []
        cols = []
        for col in self.Alpha_:
            for row in self.Alpha_[col]:
                rows.append(row)
                cols.append(col)
                vals.append(self.Alpha_[col][row])
        self.Alpha_ = sp.csr_matrix((vals, (rows, cols)), dtype='d')
        self.mu_ = self.mu_
        self.beta_ = self.beta_
