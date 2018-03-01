# -*- coding: utf8
# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: wraparound=False


'''
Fenwick Tree Sampling Implementation. Ported from the Nomad LDA paper:
http://bigdata.ices.utexas.edu/publication/nomad-lda/

Copyright (c) 2014-2015 The NOMAD-LDA Project. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither name of copyright holders nor the names of its contributors may be
   used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''


from libcpp.vector cimport vector


cdef class FPTree:

    def __init__(self, size_t size):
        self._build(size)

    cdef void _build(self, size_t size) nogil:
        self.size = size
        cdef size_t t_pos = 1
        while t_pos < size:
            t_pos *= 2
        cdef double init_val = 0.0
        self.values.resize(2 * t_pos)
        cdef size_t i
        for i in range(1, self.values.size()):
            self.values[i] = 0.0
        # values[0] == T --> where the probabilities start
        # values[1] will be the root of the FPTree
        self.values[0] = t_pos

    cdef void reset(self) nogil:
        cdef size_t i
        for i in range(1, <size_t>self.values.size()):
            self.values[i] = 0.0

    def _reset(self):
        self.reset()

    cdef double get_value(self, size_t i) nogil:
        cdef size_t t_pos = <size_t>self.values[0]
        return self.values[i + t_pos]

    def _get_value(self, size_t i):
        return self.get_value(i)

    cdef void set_value(self, size_t i, double value) nogil:
        if value < 0: value = 0
        cdef size_t t_pos = <size_t>self.values[0]
        cdef size_t pos = i + t_pos
        value -= self.values[pos]
        while pos > 0:
            self.values[pos] += value
            pos >>= 1

    def _set_value(self, size_t i, double value):
        self.set_value(i, value)

    cdef size_t sample(self, double urnd) nogil:
        # urnd: uniformly random number between [0, tree_total]
        cdef size_t t_pos = <size_t> self.values[0]
        cdef size_t pos = 1
        while pos < t_pos:
            pos <<= 1
            if urnd >= self.values[pos]:
                urnd -= self.values[pos]
                pos += 1
        return pos - t_pos

    def _sample(self, double urnd):
        return self.sample(urnd)

    cdef double get_total(self) nogil:
        return self.values[1]

    def _get_total(self):
        return self.get_total()
