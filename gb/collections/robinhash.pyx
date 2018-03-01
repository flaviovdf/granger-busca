# -*- coding: utf8
# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: wraparound=False


from libc.string cimport memset


# Decent enough hash function from Stack Overflow
# Author is Thomas Mueller
cdef inline size_t so_hash(size_t x) nogil:
    x = (x ^ (x >> 30)) * <size_t>(0xbf58476d1ce4e5b9)
    x = (x ^ (x >> 27)) * <size_t>(0x94d049bb133111eb)
    x = x ^ (x >> 31)
    return x


cdef inline size_t prime_by_n(size_t n) nogil:
    return 5 * (n * n + n) + 1


cdef class RobinHoodHash(object):

    def __init__(self, double load_factor=0.95):
        self.inserted = 0
        self.n_to_prime = 1
        self.load_factor = load_factor
        self._resize(prime_by_n(self.n_to_prime))

    cdef void _resize(self, size_t new_size) nogil:
        cdef entry_t zero
        zero.key = 0
        zero.value = 0
        zero.dib = 0
        cdef vector[entry_t] copy = vector[entry_t](self.data)

        if new_size < self.data.size():
            self.data.resize(new_size, zero)
            self.data.shrink_to_fit()
        else:
            self.data.resize(new_size, zero)
        memset(self.data.data(), 0, self.data.size() * sizeof(entry_t))

        cdef entry_t entry
        self.inserted = 0
        for entry in copy:
            if entry.key == 0:
                continue
            self.set(entry.key - 1, entry.value)

    cdef void set(self, size_t key, size_t value) nogil:
        cdef size_t kp1 = key + 1
        cdef size_t h = so_hash(kp1)

        cdef entry_t entry, tmp
        entry.key = kp1
        entry.value = value
        entry.dib = 0

        cdef size_t loc
        cdef size_t i = 0
        self.inserted += 1
        while True:
            loc = (h + i) % self.data.size()
            if self.data[loc].key == 0:
                self.data[loc] = entry
                break
            elif self.data[loc].key == kp1:
                self.data[loc].value = value
                self.inserted -= 1
                break
            elif self.data[loc].dib < entry.dib:
                tmp = self.data[loc]
                self.data[loc] = entry
                entry = tmp
            entry.dib += 1
            i += 1

        cdef double load = (<double>self.inserted) / self.data.size()
        cdef size_t new_n
        if load > self.load_factor:
            new_n = self.n_to_prime + 1
            while prime_by_n(new_n) < 2 * self.data.size():
                new_n += 1
            self.n_to_prime = new_n
            self._resize(prime_by_n(new_n))

    cdef size_t get(self, size_t key) nogil:
        cdef size_t kp1 = key + 1
        cdef size_t h = so_hash(kp1)
        cdef size_t i = 0
        cdef size_t loc
        for i in range(self.data.size()):
            loc = (h + i) % self.data.size()
            if self.data[loc].key == 0:
                return 0
            elif self.data[loc].key == kp1:
                return self.data[loc].value
        return 0

    cdef size_t shrink_to_fit(self) nogil:
        cdef size_t n_removals = 0
        cdef size_t i
        for i in range(self.data.size()):
            if self.data[i].key != 0 and self.data[i].value == 0:
                n_removals += 1
                self.data[i].key = 0
                self.data[i].value = 0
                self.data[i].dib = 0

        cdef size_t new_n = 1
        if n_removals > 0:
            while prime_by_n(new_n) < (self.inserted - n_removals):
                new_n += 1
            if new_n < self.n_to_prime:
                self.n_to_prime = new_n
                self._resize(prime_by_n(new_n))

    def _get(self, size_t key):
        return self.get(key)

    def _set(self, size_t key, size_t value):
        self.set(key, value)

    cdef size_t size(self) nogil:
        return self.inserted

    def _size(self):
        return self.size()

    def _capacity(self):
        return self.data.size()
