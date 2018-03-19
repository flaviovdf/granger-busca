# -*- coding: utf8

from gb import GrangerBusca
from gb.gbio import save_model

timestamps = []
with open('ticks.dat') as data:
    for l in data:
        timestamps.append([float(x) for x in l.split()[1:]])

granger_model = GrangerBusca(alpha_prior=1.0/len(timestamps), num_iter=300,
                             num_jobs=20, sloppy=2)

granger_model.fit(timestamps)
print(granger_model.training_time)
save_model('first_model.npz', granger_model)
