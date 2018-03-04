# -*- coding: utf8
# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: wraparound=False


from libc.stdlib cimport abort
from libc.stdlib cimport free
from libc.stdlib cimport malloc
from libc.stdlib cimport realloc

from libc.stdio cimport printf

from libc.string cimport memset
from libc.string cimport memcpy


# Decent enough hash function from Stack Overflow
# Author is Thomas Mueller
cdef inline size_t so_hash(size_t x) nogil:
    x = (x ^ (x >> 30)) * <size_t>(0xbf58476d1ce4e5b9)
    x = (x ^ (x >> 27)) * <size_t>(0x94d049bb133111eb)
    x = x ^ (x >> 31)
    return x


cdef inline size_t prime_by_n(size_t n) nogil:
    return 5 * (n * n + n) + 1


cdef void rh_resize(rh_hash_t *table, size_t new_size) nogil:
    cdef entry_t *copy = NULL
    if table.data == NULL:
        table.data = <entry_t *> malloc(new_size*sizeof(entry_t))
    else:
        copy = <entry_t *> malloc(table.capacity * sizeof(entry_t))
        if copy == NULL:
            printf("[gb] memory error!\n")
            abort()
        memcpy(copy, table.data, table.capacity * sizeof(entry_t))
        table.data = <entry_t *> realloc(table.data, new_size*sizeof(entry_t))

    if table.data == NULL:
        printf("[gb] memory error!\n")
        abort()

    memset(table.data, 0, new_size * sizeof(entry_t))

    cdef entry_t entry
    cdef size_t i
    cdef size_t old_size = table.capacity
    table.inserted = 0
    table.capacity = new_size
    if copy != NULL:
        for i in range(old_size):
            entry = copy[i]
            if entry.key == 0:
                continue
            rh_set(table, entry.key - 1, entry.value)
        free(copy)


cdef void rh_init(rh_hash_t *table, double load_factor) nogil:
    table.inserted = 0
    table.n_to_prime = 1
    table.load_factor = load_factor
    table.data = NULL
    rh_resize(table, prime_by_n(table.n_to_prime))


cdef void rh_set(rh_hash_t *table, size_t key, uint64_t value) nogil:
    cdef size_t kp1 = key + 1
    cdef size_t h = so_hash(kp1)

    cdef entry_t entry, tmp
    entry.key = kp1
    entry.value = value
    entry.dib = 0

    cdef size_t loc
    cdef size_t i = 0
    table.inserted += 1
    while True:
        loc = (h + i) % table.capacity
        if table.data[loc].key == 0:
            table.data[loc] = entry
            break
        elif table.data[loc].key == kp1:
            table.data[loc].value = value
            table.inserted -= 1
            break
        elif table.data[loc].dib < entry.dib:
            tmp = table.data[loc]
            table.data[loc] = entry
            entry = tmp
        entry.dib += 1
        i += 1

    cdef double load = (<double>table.inserted) / table.capacity
    cdef size_t new_n
    if load > table.load_factor:
        new_n = table.n_to_prime + 1
        while prime_by_n(new_n) < 2 * table.capacity:
            new_n += 1
        table.n_to_prime = new_n
        rh_resize(table, prime_by_n(new_n))


cdef uint64_t rh_get(rh_hash_t *table, size_t key) nogil:
    cdef size_t kp1 = key + 1
    cdef size_t h = so_hash(kp1)
    cdef size_t i = 0
    cdef size_t loc
    for i in range(table.inserted):
        loc = (h + i) % table.capacity
        if table.data[loc].key == 0:
            return 0
        elif table.data[loc].key == kp1:
            return table.data[loc].value
    return 0


cdef size_t rh_size(rh_hash_t *table) nogil:
    return table.inserted


cdef class Table(object):

    def __cinit__(self, size_t n_rows, double load_factor=0.95):
        self.n_rows = n_rows
        self.rows = <rh_hash_t *> malloc(n_rows * sizeof(rh_hash_t))
        if self.rows == NULL:
            raise MemoryError()
        cdef size_t i
        for i in range(n_rows):
            rh_init(&self.rows[i], load_factor)

    def __dealloc__(self):
        cdef size_t i
        if self.rows != NULL:
            for i in range(self.n_rows):
                if self.rows[i].data != NULL:
                    free(self.rows[i].data)
            free(self.rows)

    cdef uint64_t get_cell(self, size_t row, size_t col) nogil:
        return rh_get(&self.rows[row], col)

    def _get_cell(self, size_t row, size_t col):
        return self.get_cell(row, col)

    cdef void set_cell(self, size_t row, size_t col, uint64_t value) nogil:
        rh_set(&self.rows[row], col, value)

    def _set_cell(self, size_t row, size_t col, uint64_t value):
        return self.set_cell(row, col, value)


cdef class RobinHoodHash(object):

    def __cinit__(self, double load_factor=0.95):
        self.table = <rh_hash_t *> malloc(sizeof(rh_hash_t))
        if self.table == NULL:
            raise MemoryError('Robin Hash out of mem')

        rh_init(self.table, load_factor)

    def __dealloc__(self):
        if self.table != NULL:
            if self.table.data != NULL:
                free(self.table.data)
            free(self.table)

    cdef void set(self, size_t key, uint64_t value) nogil:
        rh_set(self.table, key, value)

    cdef uint64_t get(self, size_t key) nogil:
        return rh_get(self.table, key)

    def _get(self, size_t key):
        return self.get(key)

    def _set(self, size_t key, size_t value):
        self.set(key, value)

    cdef size_t size(self) nogil:
        return rh_size(self.table)

    def _size(self):
        return self.size()
