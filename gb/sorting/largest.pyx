# -*- coding: utf8
# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: language_level=3
# cython: wraparound=False


cdef inline void swap(double *a, double *b) nogil:
    cdef double aux = a[0]
    a[0] = b[0]
    b[0] = aux


cdef inline double quickselect(double *arr, size_t n, size_t k) nogil:
    cdef size_t i, ir, j, l, mid
    cdef double a, temp

    l = 0
    ir = n - 1
    while True:
        if ir <= l + 1:
            if ir == l+1 and arr[ir] < arr[l]:
                swap(&arr[l], &arr[ir])
            return arr[k]
        else:
            mid = (l + ir) >> 1
            swap(&arr[mid], &arr[l+1])
            if arr[l] > arr[ir]:
                swap(&arr[l], &arr[ir])
            if arr[l+1] > arr[ir]:
                swap(&arr[l+1], &arr[ir])
            if arr[l] > arr[l+1]:
                swap(&arr[l], &arr[l+1])
            i = l + 1
            j = ir
            a = arr[l+1]
            while True:
                i += 1
                while arr[i] < a:
                    i += 1
                j -= 1
                while (arr[j] > a):
                    j -=1
                if j < i:
                    break
                swap(&arr[i], &arr[j]);
            arr[l+1] = arr[j]
            arr[j] = a
            if j >= k:
                ir = j - 1
            if j <= k:
                l = i


cdef double quick_median(double *array, size_t n) nogil:
    cdef size_t mid = n // 2
    quickselect(array, n, mid)

    cdef size_t i
    cdef double max_val = array[0]
    if n % 2 != 0:
         return array[mid]
    else:
         for i in range(1, mid):
             if array[i] > max_val:
                 max_val = array[i]
         return (array[mid] + max_val) / 2


def _median(double[::1] array):
    return quick_median(&array[0], array.shape[0])
