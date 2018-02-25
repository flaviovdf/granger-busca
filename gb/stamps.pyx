# -*- coding: utf8
# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: wraparound=False


import numpy as np


cdef class Timestamps(object):

    def __init__(self, dict process_stamps):
        cdef int n_proc = len(process_stamps)

        self.n_stamps = 0
        cdef int proc_a
        for proc_a in range(n_proc):
            self.n_stamps += len(process_stamps[proc_a])

        self.all_stamps = np.zeros(n_proc + self.n_stamps, dtype='d')
        self.causes = np.zeros(n_proc + self.n_stamps, dtype='i') - 1

        cdef int pos = 0
        cdef int n
        cdef double[::1] stamps
        for proc_a in range(n_proc):
            stamps = np.asanyarray(process_stamps[proc_a], dtype='d')
            self.start_positions[proc_a] = pos

            n = stamps.shape[0]
            self.all_stamps[pos] = n
            self.causes[pos] = n

            pos += 1
            self.all_stamps[pos:pos+n] = stamps
            pos += n

    cdef double[::1] get_stamps(self, int process) nogil:
        cdef int pos = self.start_positions[process]
        cdef int size = self.causes[pos]
        return self.all_stamps[pos+1:pos+1+size]

    def _get_stamps(self, int process):
        return self.get_stamps(process)

    cdef int[::1] get_causes(self, int process) nogil:
        cdef int pos = self.start_positions[process]
        cdef int size = self.causes[pos]
        return self.causes[pos+1:pos+1+size]

    def _get_causes(self, int process):
        return self.get_causes(process)
