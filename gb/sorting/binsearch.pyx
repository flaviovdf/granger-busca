# -*- coding: utf8
# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: wraparound=False


cdef size_t searchsorted(double[::1] array, double value, size_t lower) nogil:
    '''
    Finds the first element in the array where the given is OR should have been
    in the given array. This is simply a binary search, but if the element is
    not found we return the index where it should have been at.

    Parameters
    ----------
    array: vector of doubles
    value: double to look for
    lower: size_t to start search from [lower, n)
    '''

    cdef size_t n = array.shape[0]
    if n == 0: return 0

    cdef size_t upper = n - 1  # closed interval
    cdef size_t half = 0
    cdef size_t idx = n

    while upper >= lower:
        half = lower + ((upper - lower) // 2)
        if value == array[half]:
            idx = half
            break
        elif value > array[half]:
            lower = half + 1
        elif half > 0:
            upper = half - 1
        else:
            break

    # Element not found, return where it should be
    if idx == n:
        idx = lower

    return idx


def _searchsorted(double[::1] array, double value, size_t lower=0):
    return searchsorted(array, value, lower)
