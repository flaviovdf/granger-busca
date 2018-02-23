# -*- coding: utf8
# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: wraparound=False

cdef class Timestamps(object):

    def __init__(dict process_stamps, int n_stamps):
        cdef int n_proc = len(process_stamps)

        self.all_stamps = np.zeros(n_proc + n_stamps, dtype='f')
        cdef int proc_a
        cdef int pos = 0
        cdef double[::1] stamps
        for proc_a in range(n_proc):
            stamps = process_stamps[proc_a]
            self.start_positions[proc_a] = pos
            self.all_stamps[pos] = stamps.shape[0]
            pos += 1
            self.all_stamps[pos:pos+stamps.shape[0]] = stamps
            pos += pos + stamps.shape[0]

    cdef double[::1] get_stamps(self, int process) nogil:
        cdef int pos = self.start_positions[process]
        cdef int size = self.all_stamps
        return self.all_stamps[pos+1:pos+1+size]
