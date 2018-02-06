# -*- coding: utf8
# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: wraparound=False

'''
Alias Tree Sampling Implementation. Ported from the Nomad LDA paper:
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


cdef class AliasTable:

    cdef void build(self, vector[double] p):
        cdef int size = p.size()
        if size:
            self.alias.resize(size)
            self.prob.resize(size)
            self.P.resize(size)
            self.L.resize(size)
            self.S.resize(size)
        else:
            return

        cdef double sum_ = 0.0
        cdef int i
        for i in range(size):
            sum_ += p[i]
        for i in range(size):
            p[i] = p[i] / sum_
        self.original = p
        for i in range(size):
            self.P[i] = p[i] * size / sum_

        cdef int nS = 0
        cdef int nL = 0
        for i in range(size):
            if self.P[i] < 1:
                self.S[nS] = i
                nS += 1
            else:
                self.L[nL] = i
                nL += 1

        cdef int a, g
        while nS and nL:
            nS -= 1
            a = self.S[nS] # Schwarz's l

            nL -= 1
            g = self.L[nL] # Schwarz's g

            self.prob[a] = self.P[a]
            self.alias[a] = g
            self.P[g] = self.P[g] + self.P[a] - 1
            if self.P[g] < 1:
                self.S[nS] = g
                nS += 1
            else:
                self.L[nL] = g
                nL += 1

        while nL:
            nL -= 1
            self.prob[self.L[nL]] = 1
        while nS:
            nS -= 1
            self.prob[self.S[nS]] = 1

    cdef int sample(self, double urnd1, double urnd2) nogil:
        cdef int idx = <int>(urnd1 * self.size);
        if urnd2 < self.prob[idx]:
            return idx
        else:
            return self.alias[idx]
