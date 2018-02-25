# -*- coding: utf8
# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: wraparound=False


include 'dirichlet.pxi'


from gb.collections cimport BitSet
from gb.collections cimport FPTree

cdef class SparseFenwickSampler(object):

    def __init__(self, int n, double[::1] joint_counts,
                 double[::1] denominators, double alpha_prior):
        self.n = n
        self.alpha_prior = alpha_prior
        self.bit_set = BitSet(n_proc)
        self.tree = FPTree(n_proc)

        cdef int i
        cdef int j = 0
        self.non_zero = 0
        for i in range(self.size):
            if joint_counts[i] != 0:
                self.non_zero_idx[j] = i
                self.reverse_idx[i] = j
                self.non_zero += 1
                j += 1
            else:
                self.bit_set.add(i)

        self.load = self.non_zero
        self.tree.reset(self.non_zero)
        cdef double prob
        for i in range(self.non_zero):
            prob = dirmulti_posterior(joint_counts[i], denominators[i],
                                      self.n, alpha_prior)
            self.tree.set_value(i, prob)
        self.current_denominators = denominators

    cdef void renormalize(self, double[::1] denominators) nogil:
        cdef double old_prob
        cdef double new_prob
        cdef pair[int, int] entry
        for entry in self.reverse_idx:
            old_prob = self.tree.get_value(j)
            prob = old_prob * \
                (self.n * alpha_prior + self.current_denominators[i]) / \
                (self.n * alpha_prior + denominators[i])
            tree.set_value(j, prob)
        self.current_denominators = denominators

    cdef double get_probability(self, int i) nogil:
        cdef int j
        if self.bit_set.get(i) == 0:
            j = self.reverse_idx[i]
            return tree.get_value(j)
        else:
            return dirmulti_posterior(0, self.current_denominators[i],
                                      self.n, alpha_prior)

    cdef void inc_one(self, int i) nogil:
        if self.bit_set.get(i) == 1:
            self.bit_set.remove(i)
            self.needs_resize = True
        cdef double old = self.tree.get_value()
        # self.tree.set_value(, old + inc())

    cdef void dec_one(self, int i) nogil:
        if self.bit_set.get(i) == 0:
            self.bit_set.add(i)
            self.needs_resize = True
        cdef double old = self.tree.get_value()
        # self.tree.set_value(, old - inc())

    cdef int sample(self) nogil:
        cdef double non_zero_prob = min(1, tree.get_total())
        if rand() < (1 - non_zero_prob): # Rare chance we sample a zero count
            return self.bit_set.next(<int>(rand() * self.size))

        cdef int i = self.tree.sample(rand())
        return self.non_zero_idx[i]

    cdef void resize(self) nogil:
        if self.non_zero > self.load:
            self.load = self.non_zero
            self.grow()
        else:
            self.shrink()
            if self.non_zero < (1-LOAD_FACTOR) * self.load: # free memory
                self.freeup_mem()
