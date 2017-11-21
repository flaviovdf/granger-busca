# coding: utf-8

from gb import GrangerBusca

import numpy as np


eps = 0.02
timestamps = [
    np.arange(100),
    np.arange(100) + eps,
    np.arange(200, 300),
    np.arange(200, 300) + eps,
]
granger_model = GrangerBusca(alpha_p=1.0/4, num_iter=200, burn_in=100)
granger_model.fit(timestamps)
print(granger_model.mu_)
print(granger_model.Alpha_.toarray())
