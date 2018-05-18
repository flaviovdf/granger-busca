# -*- coding: utf8

from gb import gbio
from scipy import sparse as sp
from sklearn.preprocessing import normalize

import experiments
import glob
import scipy.stats as ss
import numpy as np

P = []


def precision10(A_true, A_pred, k=10):
    res = 0.0
    tmp = 0
    for i in range(A_true.shape[0]):
        x = set(A_true[i].argsort()[::-1][:k])
        y = set(A_pred[i].argsort()[::-1][:k])
        res += len(x.intersection(y)) / 10
        tmp += 1
    return res / tmp


def rank_corr(A_true, A_pred):
    res = 0.
    tmp = 0
    for (x, y) in zip(A_true, A_pred):
        corr = ss.kendalltau(x, y)[0]
        P.append(ss.kendalltau(x, y)[1])
        if not np.isnan(corr):
            res += corr
            tmp += 1
    if tmp == 0:
        return 0
    return corr / tmp


def rel_err(X, Y):
    X_not_zero = X != 0
    average = 0.
    average += np.sum(np.abs(Y) * (Y == 0))
    average += np.sum(np.abs(X - Y)[X_not_zero] / np.abs(X)[X_not_zero])
    average /= X.size
    return average


results_folder = './'
avg_error = {}
kendall = {}
precision = {}
for path in glob.glob('*.gz'):
    timestamps, graph, ids = experiments.get_graph_stamps(path, 100)
    vals = []
    rows = []
    cols = []
    for src in graph:
        for dst in graph[src]:
            assert src in ids and dst in ids
            rows.append(ids[src])
            cols.append(ids[dst])
            vals.append(graph[src][dst])
    GT = sp.csr_matrix((vals, (rows, cols)), dtype='d').toarray()
    GT = normalize(GT, 'l1')

    name = path.split('/')[-1]
    model_paths = glob.glob(results_folder + '*100*.npz')

    for model_path in model_paths:
        if name in model_path and '100' in model_path:
            if ('cumu' in model_path or 'adm4' in model_path) \
                    and 'opt' not in model_path:
                continue

            if name not in avg_error:
                avg_error[name] = {}
            if name not in kendall:
                kendall[name] = {}
            if name not in precision:
                precision[name] = {}

            data = np.load(model_path)
            if len(data.keys()) == 1:
                G = data['G'].T
            else:
                model = gbio.load_model(model_path)
                G = model.Alpha_.toarray()
                G = normalize(G, 'l1')

            if 'cum' in model_path:
                mname = 'cumulants'
            elif 'adm4' in model_path:
                mname = 'adm4'
            elif 'sumg' in model_path:
                mname = 'gaussian'
            else:
                mname = 'granger'
            try:
                avg_error[name][mname] = rel_err(G, GT)
                kendall[name][mname] = rank_corr(GT, G)
                precision[name][mname] = precision10(GT, G)
            except:
                pass

print('Kendall')
print("{:<40}\t{:<10}\t{:<10}".format('Dataset', 'Method', 'Value'))
for data, score in sorted(kendall.items()):
    for method, value in score.items():
        print("{:<40}\t{:<10}\t{:<10}".format(data, method, value))
    print()
print()
print()
print('Avg Error mean(True - Pred)')
for data, score in sorted(avg_error.items()):
    for method, value in score.items():
        print("{:<40}\t{:<10}\t{:<10}".format(data, method, value))
    print()
print()
print('Precision@10')
for data, score in sorted(precision.items()):
    for method, value in score.items():
        print("{:<40}\t{:<10}\t{:<10}".format(data, method, value))
    print()
