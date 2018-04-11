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
        self.n_proc = n_proc
        self.n_stamps = 0
        cdef size_t proc_a
        for proc_a in range(n_proc):
            self.n_stamps += len(process_stamps[proc_a])

        self.all_stamps = np.zeros(n_proc + self.n_stamps, dtype='d')
        self.causes = np.zeros(n_proc + self.n_stamps, dtype='uint64') + n_proc

        cdef size_t pos = 0
        cdef size_t n
        cdef double[::1] stamps
        self.start_positions = np.zeros(n_proc, dtype='uint64')
        for proc_a in range(n_proc):
            stamps = np.asanyarray(process_stamps[proc_a], dtype='d')
            self.start_positions[proc_a] = pos

            n = stamps.shape[0]
            self.all_stamps[pos] = n
            self.causes[pos] = n

            pos += 1
            self.all_stamps[pos:pos+n] = stamps
            pos += n

    cdef void get_stamps(self, size_t process, double **at) nogil:
        cdef size_t pos = self.start_positions[process]
        at[0] = &self.all_stamps[pos+1]

    cdef double get_stamp(self, size_t process, size_t i) nogil:
        cdef size_t pos = self.start_positions[process]
        return self.all_stamps[pos+1+i]

    cdef void get_causes(self, size_t process, size_t **at) nogil:
        cdef size_t pos = self.start_positions[process]
        at[0] = &self.causes[pos+1]

    cdef size_t get_cause(self, size_t process, size_t i) nogil:
        cdef size_t pos = self.start_positions[process]
        return self.causes[pos+1+i]

    cdef double find_previous(self, size_t process, double t) nogil:
        cdef size_t n = self.get_size(process)
        cdef double *timestamps
        self.get_stamps(process, &timestamps)
        cdef size_t tp_idx = searchsorted(timestamps, n, t, 0)
        if tp_idx > 0:
            tp_idx = tp_idx - 1
        cdef double tp = timestamps[tp_idx]
        if tp >= t:
            tp = 0
        return tp

    cdef size_t get_size(self, size_t process) nogil:
        cdef size_t pos = self.start_positions[process]
        return self.causes[pos]

    cdef size_t num_proc(self) nogil:
        return self.n_proc

    def _get_stamps(self, size_t process):
        cdef size_t n = self.get_size(process)
        cdef double *at
        self.get_stamps(process, &at)
        return <double[:n]> at

    def _get_stamp(self, size_t process, size_t i):
        return self.get_stamp(process, i)

    def _get_causes(self, size_t process):
        cdef size_t n = self.get_size(process)
        cdef size_t *at
        self.get_causes(process, &at)
        return <size_t[:n]> at

    def _get_cause(self, size_t process, size_t i):
        return self.get_cause(process, i)

    def _find_previous(self, size_t process, double t):
        return self.find_previous(process, t)

    def _get_size(self, size_t process):
        return self.get_size(process)

    def _num_proc(self):
        return self.num_proc()
