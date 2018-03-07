# -*- coding: utf8
# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: wraparound=False


from gb.sorting.binsearch cimport searchsorted

import numpy as np


cdef class Timestamps(object):

    def __init__(self, dict process_stamps):
        cdef size_t n_proc = len(process_stamps)

        self.n_stamps = 0
        cdef size_t proc_a
        for proc_a in range(n_proc):
            self.n_stamps += len(process_stamps[proc_a])

        self.all_stamps = np.zeros(n_proc + self.n_stamps, dtype='d')
        self.causes = np.zeros(n_proc + self.n_stamps, dtype='uint64') + n_proc

        cdef size_t pos = 0
        cdef size_t n
        cdef double[::1] stamps
        self.start_positions = RobinHoodHash()
        for proc_a in range(n_proc):
            stamps = np.asanyarray(process_stamps[proc_a], dtype='d')
            self.start_positions.set(proc_a, pos)

            n = stamps.shape[0]
            self.all_stamps[pos] = n
            self.causes[pos] = n

            pos += 1
            self.all_stamps[pos:pos+n] = stamps
            pos += n

    cdef double[::1] get_stamps(self, size_t process) nogil:
        cdef size_t pos = self.start_positions.get(process)
        cdef size_t size = self.causes[pos]
        return self.all_stamps[pos+1:pos+1+size]

    def _get_stamps(self, size_t process):
        return self.get_stamps(process)

    cdef size_t[::1] get_causes(self, size_t process) nogil:
        cdef size_t pos = self.start_positions.get(process)
        cdef size_t size = self.causes[pos]
        return self.causes[pos+1:pos+1+size]

    def _get_causes(self, size_t process):
        return self.get_causes(process)

    cdef double find_previous(self, size_t process, double t) nogil:
        cdef double[::1] timestamps = self.get_stamps(process)
        cdef size_t tp_idx = searchsorted(timestamps, t, 0)
        if tp_idx > 0:
            tp_idx = tp_idx - 1
        cdef double tp = timestamps[tp_idx]
        if tp >= t:
            tp = 0
        return tp

    def _find_previous(self, size_t process, double t):
        return self.find_previous(process, t)
