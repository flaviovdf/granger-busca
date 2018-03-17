# -*- coding: utf8

from gb import GrangerBusca

import tick.simulation as hk
import numpy as np

# Simulation of a 10-dimensional Hawkes process
beta = 1.
mu = 0.01  # 0.01
d = 10
T = 1e5
H = 10
n_days = 20

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
granger_model = GrangerBusca(alpha_prior=1.0/len(h.timestamps), num_iter=300,
                             num_jobs=4, metropolis=True)
granger_model.fit(h.timestamps)
print(granger_model.mu_)
print(np.array(granger_model.Beta_.toarray()))
np.set_printoptions(precision=2)
print(np.array(granger_model.Alpha_.toarray(), dtype='i').T)
print(np.array(Alpha > 0, dtype='i'))
