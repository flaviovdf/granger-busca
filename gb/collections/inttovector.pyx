# -*- coding: utf8
# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: language_level=3
# cython: wraparound=False


from libc.stdlib cimport abort
from libc.stdlib cimport free
from libc.stdlib cimport malloc
from libc.stdlib cimport realloc


cdef class IntToVector(object):

    def __cinit__(self, size_t n_proc, size_t init_capacity):
        self.vectors = <vector *> malloc(n_proc * sizeof(vector))
        if self.vectors == NULL:
            raise MemoryError()

        cdef size_t i
        for i in range(n_proc):
            self.vectors[i].data = \
                <double *> malloc(init_capacity * sizeof(double))
            if self.vectors[i].data == NULL:
                raise MemoryError()
            self.vectors[i].capacity = init_capacity
            self.vectors[i].size = 0
        self.n_proc = n_proc

    def __dealloc__(self):
        cdef size_t i
        if self.vectors != NULL:
            for i in range(self.n_proc):
                if self.vectors[i].data != NULL:
                    free(self.vectors[i].data)
            free(self.vectors)

    cdef void reset(self) nogil:
        cdef size_t i
        for i in range(self.n_proc):
            self.vectors[i].size = 0

    cdef void push_back(self, size_t i, double value) nogil:
        if self.vectors[i].size == self.vectors[i].capacity:
            self.vectors[i].capacity *= 2
            self.vectors[i].data = \
                <double *> realloc(self.vectors[i].data,
                                   self.vectors[i].capacity * sizeof(double))
            if self.vectors[i].data == NULL:
                abort()
        self.vectors[i].data[self.vectors[i].size] = value
        self.vectors[i].size += 1

    cdef size_t get_size(self, size_t i) nogil:
        return self.vectors[i].size

    cdef double *get_values(self, size_t i) nogil:
        return self.vectors[i].data
