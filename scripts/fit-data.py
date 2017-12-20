# -*- coding: utf8

from gb import GrangerBusca

import numpy as np

timestamps = []
with open('ticks.dat') as data:
    for l in data:
        timestamps.append([float(x) for x in l.split()[1:]])

granger_model = GrangerBusca(alpha_p=1.0/len(timestamps), num_iter=300,
                             burn_in=200)
granger_model.fit(timestamps)
print(granger_model.back_)
print(granger_model.mu_)
print(granger_model.beta_)
np.set_printoptions(precision=2)
print(granger_model.Alpha_.toarray().T)
