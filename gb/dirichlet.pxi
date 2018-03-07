# -*- coding: utf8
# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: wraparound=False


cdef inline double dirmulti_posterior(int n_ab, int n_b, int n,
                                      double prior) nogil:
    return (n_ab + prior) / (n_b + n * prior)


cdef inline double busca_probability(size_t i, size_t proc_a, size_t proc_b,
                                     Timestamps all_stamps, double alpha_ba,
                                     double beta_rate) nogil:
    cdef double E = 2.718281828459045
    cdef double[::1] stamps = all_stamps.get_stamps(proc_a)
    cdef double t = stamps[i]
    cdef double tp
    cdef double tpp
    if i > 0:
        tp = stamps[i-1]
    else:
        tp = 0
    if tp != 0:
        if proc_a == proc_b:
            if i > 1:
                tpp = stamps[i-2]
            else:
                tpp = 0
        else:
            tpp = all_stamps.find_previous(proc_b, tp)
    else:
        tpp = 0

    cdef double rate = alpha_ba / (beta_rate/E + tp - tpp)
    return rate
