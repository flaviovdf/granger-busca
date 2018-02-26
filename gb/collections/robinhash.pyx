# -*- coding: utf8
# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: wraparound=False


# Decent enough hash function from:
# https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
# Author is Thomas Mueller
cdef inline uint32 so_hash(uint32 x):
    x = ((x >> 16) ^ x) * <uint32>0x45d9f3b
    x = ((x >> 16) ^ x) * <uint32>0x45d9f3b
    x = (x >> 16) ^ x
    return x


cdef class RobinHoodHash(object):

    def __init__(self, int initial_capacity=256, double load_factor=0.95):
        self.size = 0
        self.data.resize(initial_capacity, 0)
        self.load_factor = load_factor
        self.capacity = initial_capacity

    cdef int insert(self, uint32_t key, uint32_t value) nogil:
        if key == UINT32_MAX:
            return 0

        key += 1 # 0 means empty
        cdef uint32 hash = so_hash(key)
        cdef uint32 location = hash % self.capacity

        return 1

    cdef int remove(self, uint32_t key, uint32_t value) nogil:
        if key == UINT32_MAX:
            return 0

        pass

    cdef int size(self) nogil:
        return self.size
