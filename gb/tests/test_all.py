# -*- coding: utf8


from gb import GrangerBusca

try:
    import tick.simulation as hk
    hk.HawkesKernelExp
except (ImportError, AttributeError):
    import tick.hawkes as hk

import numpy as np


def test_metropolis():
    # Simulation of a 10-dimensional Hawkes process
    beta = 1.
    mu = 0.01  # 0.01
    d = 10
    T = 1e5

    mus = mu * np.ones(d)
    Alpha = np.zeros((d, d))
    Beta = np.zeros((d, d))
    for i in range(5):
        for j in range(5):
            if i >= j:
                Alpha[i][j] = 1.
                Beta[i][j] = 100*beta

    for i in range(5, 10):
        for j in range(5, 10):
            if i >= j:
                Alpha[i][j] = 1.
                Beta[i][j] = 100*beta
    Alpha /= 6
    kernels = [[hk.HawkesKernelExp(a, b) for (a, b) in zip(a_list, b_list)]
               for (a_list, b_list) in zip(Alpha, Beta)]
    h = hk.SimuHawkes(kernels=kernels, baseline=list(mus), end_time=T)
    h.simulate()
    granger_model = GrangerBusca(alpha_prior=1.0/len(h.timestamps),
                                 num_iter=300, metropolis=True, num_jobs=1)
    granger_model.fit(h.timestamps)
    P = np.array(granger_model.Alpha_.toarray() > 0, dtype='i').T
    T = np.array(Alpha > 0, dtype='i')
    assert (((P - T) == 0).sum() / P.size) > 0.5


def test_metropolis_multithread():
    # Simulation of a 10-dimensional Hawkes process
    beta = 1.
    mu = 0.01  # 0.01
    d = 10
    T = 1e5

    mus = mu * np.ones(d)
    Alpha = np.zeros((d, d))
    Beta = np.zeros((d, d))
    for i in range(5):
        for j in range(5):
            if i >= j:
                Alpha[i][j] = 1.
                Beta[i][j] = 100*beta

    for i in range(5, 10):
        for j in range(5, 10):
            if i >= j:
                Alpha[i][j] = 1.
                Beta[i][j] = 100*beta
    Alpha /= 6
    kernels = [[hk.HawkesKernelExp(a, b) for (a, b) in zip(a_list, b_list)]
               for (a_list, b_list) in zip(Alpha, Beta)]
    h = hk.SimuHawkes(kernels=kernels, baseline=list(mus), end_time=T)
    h.simulate()
    granger_model = GrangerBusca(alpha_prior=1.0/len(h.timestamps),
                                 num_iter=300, metropolis=True, num_jobs=4)
    granger_model.fit(h.timestamps)
    P = np.array(granger_model.Alpha_.toarray() > 0, dtype='i').T
    T = np.array(Alpha > 0, dtype='i')
    assert (((P - T) == 0).sum() / P.size) > 0.5


def test_gibbs():
    # Simulation of a 10-dimensional Hawkes process
    beta = 1.
    mu = 0.01  # 0.01
    d = 10
    T = 1e5

    mus = mu * np.ones(d)
    Alpha = np.zeros((d, d))
    Beta = np.zeros((d, d))
    for i in range(5):
        for j in range(5):
            if i >= j:
                Alpha[i][j] = 1.
                Beta[i][j] = 100*beta

    for i in range(5, 10):
        for j in range(5, 10):
            if i >= j:
                Alpha[i][j] = 1.
                Beta[i][j] = 100*beta
    Alpha /= 6
    kernels = [[hk.HawkesKernelExp(a, b) for (a, b) in zip(a_list, b_list)]
               for (a_list, b_list) in zip(Alpha, Beta)]
    h = hk.SimuHawkes(kernels=kernels, baseline=list(mus), end_time=T)
    h.simulate()
    granger_model = GrangerBusca(alpha_prior=1.0/len(h.timestamps),
                                 num_iter=300, metropolis=False, num_jobs=1)
    granger_model.fit(h.timestamps)
    P = np.array(granger_model.Alpha_.toarray() > 0, dtype='i').T
    T = np.array(Alpha > 0, dtype='i')
    assert (((P - T) == 0).sum() / P.size) > 0.5
