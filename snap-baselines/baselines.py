#-*- coding: utf8

from hyperopt import fmin
from hyperopt import hp
from hyperopt import tpe

from tick.hawkes import HawkesADM4
from tick.hawkes import HawkesCumulantMatching
from tick.hawkes import HawkesSumGaussians

from time import time

import experiments
import numpy as np
import sys

def run_cumulants(data):
    def objective(h):
        model = HawkesCumulantMatching(h, max_iter=300)
        model.fit(data)
        return model.objective(model.adjacency)
    best = fmin(objective,
                space=hp.uniform('h', 1, 100),
                algo=tpe.suggest,
                max_evals=20)
    half_width = best['h']
    model = HawkesCumulantMatching(half_width, max_iter=300, verbose=True)
    model.fit(data)
    return model


def run_adm4(data):
    def objective(d):
        model = HawkesADM4(d, max_iter=300, n_threads=20)
        model.fit(data)
        return -model.score()
    best = fmin(objective,
                space=hp.uniform('d', 1e-2, 1e2),
                algo=tpe.suggest,
                max_evals=20)
    decay = best['d']
    print(decay)
    model = HawkesADM4(decay, max_iter=300, n_threads=16, verbose=True)
    model.fit(data)
    return model


models = {'adm4':HawkesADM4,
          'cumu':HawkesCumulantMatching,
          'sumgauss':HawkesSumGaussians}

name = sys.argv[1]
path = sys.argv[2]
top = None
if len(sys.argv) >= 4:
    top = sys.argv[3]
    top = int(top)

timestamps, graph, ids = experiments.get_graph_stamps(path, top)
realizations = []
realizations.append([np.array(timestamps[i]) for i in range(len(timestamps))])

hawkes = models[name]
begin=time()
if hawkes == HawkesCumulantMatching:
    model = run_cumulants(realizations)
elif hawkes == HawkesSumGaussians:
    model = hawkes(10, n_threads=20, max_iter=300, verbose=True)
    model.fit(realizations)
else:
    model = run_adm4(realizations)

print(name, path, top, time()-begin)

if top is not None:
    path = path + '-' + str(top)
print(model.get_kernel_norms().shape)
np.savez_compressed(name + '-optim-' + path + '.npz',
                    G=model.get_kernel_norms())
