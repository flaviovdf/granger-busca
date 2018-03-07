# -*- coding: utf8
# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: wraparound=False


from libc.stdint cimport uint64_t


ctypedef struct entry_t:
    size_t key
    uint64_t value
    size_t dib


ctypedef struct rh_hash_t:
    size_t inserted
    size_t n_to_prime
    size_t capacity
    double load_factor
    entry_t *data


cdef class Table(object):
    cdef rh_hash_t *rows
    cdef size_t n_rows

    cdef void set_cell(self, size_t row, size_t col, uint64_t value) nogil
    cdef uint64_t get_cell(self, size_t row, size_t col) nogil


cdef class RobinHoodHash(object):
    cdef rh_hash_t *table

    cdef void set(self, size_t key, uint64_t value) nogil
    cdef uint64_t get(self, size_t key) nogil
    cdef size_t size(self) nogil
