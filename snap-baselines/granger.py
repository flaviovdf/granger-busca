#-*- coding: utf8

from gb import GrangerBusca
from gb.gbio import save_model

from time import time

import experiments
import glob
import gzip
import numpy as np
import sys

name = sys.argv[1]
path = sys.argv[2]
top = None
if len(sys.argv) >= 4:
    top = sys.argv[3]
    top = int(top)
start = time()
timestamps, graph, ids = experiments.get_graph_stamps(path, top)
granger_model = GrangerBusca(alpha_prior=1.0/len(timestamps), num_iter=300,
                             beta_strategy=1)
granger_model.fit(timestamps)
print(path, granger_model.training_time)
print(path, 'with pre processing', time() - start)
if top is not None:
    path = path + '-' + str(top)
save_model(path + '.busca.model', granger_model)
