# -*- coding: utf8
# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: wraparound=False


include 'dirichlet.pxi'

from gb.randomkit.random cimport rand

from libcpp.map cimport pair


cdef class SparseFenwickSampler(object):

    def __init__(self, int n, unordered_map[int, int] joint_counts,
                 int[::1] denominators, double alpha_prior):
        cdef int n_proc = denominators.shape[0]
        self.n = n
        self.alpha_prior = alpha_prior
        self.bit_set = BitSet(n_proc)

        cdef int i
        cdef int j = 0
        self.non_zero = 0
        for i in range(self.size):
            if joint_counts.count(i) != 0:
                self.non_zero_idx[j] = i
                self.reverse_idx[i] = j
                self.non_zero += 1
                j += 1
            else:
                self.bit_set.add(i)

        self.load = self.non_zero
        self.tree = FPTree()
        cdef double prob
        cdef int joint
        for i in range(self.non_zero):
            if joint_counts.count(i) == 0:
                joint = 0
            else:
                joint = joint_counts[i]
            prob = dirmulti_posterior(joint, denominators[i], self.n,
                                      alpha_prior)
            self.tree.set_value(i, prob)
        self.current_denominators = denominators

    cdef void renormalize(self, int[::1] denominators) nogil:
        cdef double old_prob
        cdef double new_prob
        cdef pair[int, int] entry
        cdef int joint_count
        cdef int i
        cdef int j
        for entry in self.reverse_idx:
            i = entry.first
            j = entry.second
            old_prob = dirmulti_posterior(joint_count,
                                          self.current_denominators[i],
                                          self.n, self.alpha_prior)
            old_prob = numerator(old_prob, self.current_denominators[i],
                                 self.n, self.alpha_prior)
            prob = renormalize(old_prob, denominators[i], self.n,
                               self.alpha_prior)
            self.tree.set_value(j, prob)
        self.current_denominators = denominators

    cdef double get_probability(self, int i) nogil:
        cdef int joint_count
        cdef int j
        if self.bit_set.get(i) == 0:
            j = self.reverse_idx[i]
            joint_count = self.joint_counts[j]
        else:
            joint_count = 0
        return dirmulti_posterior(joint_count, self.current_denominators[i],
                                  self.n, self.alpha_prior)

    cdef void inc_one(self, int i) nogil:
        cdef double change
        cdef int j = self.reverse_idx[i]
        if self.joint_counts[j] == 0:
            self.bit_set.remove(i)
            self.needs_resize = True

        if self.joint_counts[j] >= 1:
            change = inc(joint_count, self.current_denominators[i],
                         self.n, self.alpha_prior, +1)
            self.joint_counts[j] += 1
            self.denominators[i] += 1
            self.tree.set_value(j, old + change)
        else:
            if self.new_counts.get(i) == 0:
                self.new_counts[i] = 0
            self.new_counts[i] += 1

    cdef void dec_one(self, int i) nogil:
        cdef double change
        cdef int j = self.reverse_idx[i]
        if self.joint_counts[j] == 1:
            self.bit_set.add(i)
            self.needs_resize = True
        if self.joint_counts[j] >= 1:
            change = inc(joint_count, self.current_denominators[i],
                         self.n, self.alpha_prior, -1)
            self.joint_counts[j] -= 1
            self.denominators[i] -= 1
            self.tree.set_value(j, old + change)

    cdef int sample(self) nogil:
        cdef double non_zero_prob = min(1, tree.get_total())
        if rand() < (1 - non_zero_prob): # Rare chance we sample a zero count
            return <int>(rand() * self.size)

        cdef int j = self.tree.sample(rand())
        return self.non_zero_idx[j]

    # cdef void resize(self) nogil:
    #     if self.non_zero > self.load:
    #         self.load = self.non_zero
    #         self.grow()
    #     else:
    #         self.shrink()
    #         if self.non_zero < (1-LOAD_FACTOR) * self.load: # free memory
    #             self.freeup_mem()
