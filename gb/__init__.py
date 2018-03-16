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


def to_csr(sparse_dict, dtype='d'):
    vals = []
    rows = []
    cols = []
    for col in sparse_dict:
        for row in sparse_dict[col]:
            rows.append(row)
            cols.append(col)
            vals.append(sparse_dict[col][row])
    return sp.csr_matrix((vals, (rows, cols)), dtype=dtype)


class GrangerBusca(object):

    def __init__(self, alpha_prior, num_iter, metropolis=True, sloppy=2,
                 num_jobs=None):
        self.alpha_prior = alpha_prior
        self.num_iter = num_iter
        self.sloppy = sloppy
        if num_jobs is None:
            num_jobs = os.cpu_count()
        if num_jobs == 1:
            sloppy = 2 * num_iter
        self.num_jobs = num_jobs
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

    def _init_state(self, static_schedule, proc2worker):
        state_keeper = Timestamps(self.timestamps)

        n_proc = len(self.timestamps)
        n_workers = len(static_schedule)

        local_state = np.zeros((n_workers, n_proc), dtype='uint64', order='C')
        global_state = np.zeros(n_proc, dtype='uint64', order='C')

        for a in range(n_proc):
            causes = state_keeper._get_causes(a)
            init_state = np.random.randint(0, n_proc + 1, size=causes.shape[0],
                                           dtype='uint64')
            for b in init_state[init_state != n_proc]:
                global_state[b] += 1
                worker_id = proc2worker[b]
                local_state[worker_id, b] += 1

        return state_keeper, SloppyCounter(n_workers, self.sloppy,
                                           global_state, local_state)

    def fit(self, timestamps):
        self._init_timestamps(timestamps)

        now = time.time()
        schedule, inverse_schedule = greedy_schedule(self.timestamps,
                                                     self.num_jobs)
        state_keeper, sloppy_counter = self._init_state(schedule,
                                                        inverse_schedule)

        def parallel_fit(worker_id):
            return cyfit(state_keeper, sloppy_counter, self.alpha_prior,
                         self.num_iter, worker_id, schedule[worker_id],
                         int(self.metropolis))

        self.Alpha_ = {}
        self.mu_ = np.zeros(len(self.timestamps), dtype='d')
        self.Beta_ = {}
        self.back_ = np.zeros(len(self.timestamps), dtype='i')
        self.curr_state_ = {}
        with futures.ThreadPoolExecutor(max_workers=self.num_jobs) as executor:
            jobs = [executor.submit(parallel_fit, worker_id)
                    for worker_id in range(self.num_jobs)]
            futures.wait(jobs)
            for job in jobs:
                Alpha_, mu_, Beta_, back_, curr_state_ = job.result()
                self.Alpha_.update(Alpha_)
                self.mu_ += mu_
                self.Beta_.update(Beta_)
                self.back_ += back_
                self.curr_state_.update(curr_state_)

        self.Alpha_ = to_csr(self.Alpha_)
        self.Beta_ = to_csr(self.Beta_)

        dt = time.time() - now
        self.training_time = dt
