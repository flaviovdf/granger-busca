# -*- coding: utf8
# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: wraparound=False


# Decent enough hash function from Stack Overflow
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
        self.initial_capacity = initial_capacity
        self.capacity = initial_capacity

    cdef int update(self, uint32_t key, uint32_t delta):
        if key == UINT32_MAX:
            return -1

        cdef uint64_t mask = 0xffff0000
        cdef uint32_t h = so_hash(key + 1)
        cdef uint64_t old_entry
        cdef uint32_t old_val
        cdef uint32_t loc = h % self.capacity
        while True:
            if self.data[loc] == 0:
                old_val = 0
                break
            elif <uint32_t>(data[loc] >> 32) == key:
                old_val = <uint32_t>(self.data[loc] & mask)
                self.probe_count[loc] += 1
                break
            else:
                old_entry = self.data[loc]
                if self.probe_count
                break
            loc = (loc + 1) % self.capacity


        cdef uint64_t new_val = old_val + delta
        if new_val == UINT32_MAX:
            return -1
        elif new_val == 0:
            self.data[loc] = 0
            return 0

        cdef uint64_t entry = (key << 32) | new_val
        self.data[loc] = entry

    cdef int inc(self, uint32_t key) nogil:
        return self.update(key, 1)

    cdef int dec(self, uint32_t key) nogil:
        return self.update(key, -1)

    cdef int size(self) nogil:
        return self.size
