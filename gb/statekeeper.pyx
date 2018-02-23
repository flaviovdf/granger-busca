# -*- coding: utf8
# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: wraparound=False

from gb.collections cimport BitSet
from gb.collections cimport FPTree


cdef inline double dirmulti_posterior(int joint_count, int denominator,
                                      int n, double prior) nogil:
    return (joint_count + prior) / (denominator + n * prior)


cdef inline double inc(int joint_count, int denominator, int n, double prior,
                       double delta) nogil:
    cdef double b = denominator + n * prior
    cdef double a = joint_count + prior
    return (b * delta - a * delta) / (b * (b + delta))


cdef class SparseStateKeeper(object):

    def __init__(self, int size, double alpha_prior):
        self.alpha_prior = alpha_prior
        self.non_zero = 0
        self.non_zero_idx.resize(size)
        self.bit_set = BitSet(size)
        self.tree = FPTree(size)

    cdef void pack(self, double[::1] joint_counts,
                   double[::1] denominators) nogil:
        cdef int i
        cdef int j = 0
        self.non_zero = 0
        for i in range(self.size):
            if joint_counts[i] != 0:
                self.non_zero_idx[j] = i
                self.non_zero += 1
                j += 1
        self.tree.reset(self.non_zero)
        for i in range(self.non_zero):
            self.tree.set_value(i, prob)

    cdef void repack(self, vector[double] denominators) nogil:
        self.denominators =
        pass

    cdef int get_count(self, int i) nogil:
        cdef int joint_count = 0
        if self.bit_set.get(i) == 0:
            joint_count = self.counts(j)
        return joint_count

    cdef double get_probability(self, int i) nogil:
        cdef int joint_count = self.get_count(i)
        return dirmulti_posterior()

    cdef void inc_one(self, int i) nogil:
        if self.counts[i] == 0:
            self.needs_pack = True
        cdef double old = self.tree.get_value()
        self.tree.set_value(, old + inc())

    cdef void dec_one(self, int i) nogil:
        if self.counts[i] == 1:
            self.needs_pack = True
        cdef double old = self.tree.get_value()
        self.tree.set_value(, old - inc())

    cdef int sample(self) nogil:
        cdef double non_zero_prob = min(1, tree.get_total())
        if rand() < (1 - non_zero_prob): # Rare chance we sample a zero count
            return self.bit_set.next(<int>(rand() * self.size))

        cdef int i = self.tree.sample(rand())
        return self.non_zero_idx[i]
