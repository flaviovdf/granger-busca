# -*- coding: utf8


import heapq
import numpy as np


def define_worker_for_each_process(static_schedule):
    '''Given a static schedule generates the reverse mapping'''
    proc2worker = {}
    for worker_id in static_schedule:
        for proc in static_schedule[worker_id]:
            proc2worker[proc] = worker_id
    return proc2worker


def greedy_schedule(timestamps, n_workers):
    '''
    Implements the Longest Processing Time Scheduling Strategy. At each
    iteration, the largest process is scheduled to the worker with the least
    load.

    Parameters
    ----------
    timestamps: dict of lists
        The timestaps of each process
    n_workers: int
        The number of workers

    Returns
    -------
    A dict with the assignment for each worker
    '''
    schedule = {}
    heap = []
    for worker in range(n_workers):
        schedule[worker] = []
        heapq.heappush(heap, (0, worker))

    for n, proc in sorted(((len(timestamps[k]), k) for k in timestamps),
                          reverse=True):
        load, worker = heapq.heappop(heap)
        schedule[worker].append(proc)
        heapq.heappush(heap, (load + n, worker))

    rv = dict((k, np.array(v, dtype='uint64')) for k, v in schedule.items())
    return rv, define_worker_for_each_process(rv)
