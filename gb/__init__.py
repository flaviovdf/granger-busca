# -*- coding: utf8

from concurrent import futures

from gb.fit import fit as cyfit
from gb.scheduler import greedy_schedule
from gb.sloppy import SloppyCounter
from gb.stamps import Timestamps

from scipy import sparse as sp

import numpy as np
import os
import time


def to_csr(sparse_dict, n_proc, dtype='d'):
    vals = []
    rows = []
    cols = []
    for col in sparse_dict:
        for row in sparse_dict[col]:
            rows.append(row)
            cols.append(col)
            vals.append(sparse_dict[col][row])
    return sp.csr_matrix((vals, (rows, cols)), shape=(n_proc, n_proc),
                         dtype=dtype)


class GrangerBusca(object):

    def __init__(self, alpha_prior, num_iter, metropolis=True, sloppy=1,
                 beta_strategy=1, num_jobs=None):

        if isinstance(beta_strategy, str):
            if beta_strategy != 'busca':
                raise AttributeError('Please choose "busca" or a number ' +
                                     'for the beta_strategy attribute')

        self.alpha_prior = alpha_prior
        self.num_iter = num_iter
        self.sloppy = sloppy
        if num_jobs is None:
            num_jobs = os.cpu_count()
        if num_jobs == 1:
            sloppy = 2 * num_iter
        self.num_jobs = num_jobs
        self.metropolis = metropolis
        self.beta_strategy = beta_strategy

    def _init_timestamps(self, timestamps, sort):
        min_all = np.float('inf')
        max_all = 0
        copy_of_timestamps = {}
        n_proc = len(timestamps)
        for i in range(n_proc):
            tis = np.asanyarray(timestamps[i], dtype='d', order='C').copy()
            min_all = min(min_all, tis.min())
            max_all = max(max_all, tis.max())
            if sort:
                tis.sort()
            copy_of_timestamps[i] = tis

        timestamps = copy_of_timestamps
        max_all -= min_all
        medians = []
        for i in range(n_proc):
            timestamps[i] -= min_all
            medians.append(np.median(np.ediff1d(timestamps[i])))

        self.n_proc = n_proc
        self.timestamps = timestamps
        self.time_range = max_all
        self.medians = np.array(medians)

    def _init_state(self, static_schedule):
        state_keeper = Timestamps(self.timestamps)

        n_proc = len(self.timestamps)
        n_workers = len(static_schedule)

        local_state = np.zeros((n_workers, n_proc), dtype='uint64', order='C')
        global_state = np.zeros(n_proc, dtype='uint64', order='C')

        for a in range(n_proc):
            causes = state_keeper._get_causes(a)
            init_state = np.random.randint(0, n_proc + 1, size=causes.shape[0],
                                           dtype='uint64')
            causes[:] = init_state
            unique, counts = np.unique(init_state[init_state != n_proc],
                                       return_counts=True)
            counts = np.array(counts, dtype='uint64')
            global_state[unique] += counts
            local_state[:, unique] += counts

        return state_keeper, SloppyCounter(n_workers, self.sloppy,
                                           global_state, local_state)

    def fit(self, timestamps, sort=False):
        now = time.time()

        self._init_timestamps(timestamps, sort)
        schedule, _ = greedy_schedule(self.timestamps, self.num_jobs)
        state_keeper, sloppy_counter = self._init_state(schedule)

        if self.beta_strategy == 'busca':
            self.beta_ = self.medians / np.e
        else:
            self.beta_ = np.ones(self.medians.shape[0], dtype='d', order='C')
            self.beta_ = self.beta_ * self.beta_strategy

        def parallel_fit(worker_id):
            return cyfit(state_keeper, sloppy_counter, self.alpha_prior,
                         self.beta_, self.num_iter, worker_id,
                         schedule[worker_id], int(self.metropolis))

        self.Alpha_ = {}
        self.mu_ = np.zeros(len(self.timestamps), dtype='d')
        self.back_ = np.zeros(len(self.timestamps), dtype='uint64')
        self.curr_state_ = {}
        with futures.ThreadPoolExecutor(max_workers=self.num_jobs) as executor:
            jobs = [executor.submit(parallel_fit, worker_id)
                    for worker_id in range(self.num_jobs)]
            for job in jobs:
                Alpha_, mu_, back_, curr_state_ = job.result()
                self.Alpha_.update(Alpha_)
                self.mu_ += mu_
                self.back_ += back_
                self.curr_state_.update(curr_state_)

        self.Alpha_ = to_csr(self.Alpha_, self.n_proc, dtype='uint64')
        dt = time.time() - now
        self.training_time = dt
