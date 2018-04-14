# -*- coding: utf8

from gb import GrangerBusca

from scipy import sparse as sp

import numpy as np


def save_model(filename, granger_model):
    state = {}
    for key, val in granger_model.curr_state_.items():
        state['id_{}'.format(key)] = np.asanyarray(val)

    np.savez_compressed(filename,
                        Alpha_data=granger_model.Alpha_.data,
                        Alpha_indices=granger_model.Alpha_.indices,
                        Alpha_indptr=granger_model.Alpha_.indptr,
                        Alpha_shape=granger_model.Alpha_.shape,
                        alpha_prior=granger_model.alpha_prior,
                        beta=granger_model.beta_,
                        mu_=granger_model.mu_,
                        num_iter=granger_model.num_iter,
                        metropolis=granger_model.metropolis,
                        num_jobs=granger_model.num_jobs,
                        sloppy=granger_model.sloppy,
                        **state)


def load_model(filename):
    model = GrangerBusca(0, 0, 0)
    loader = np.load(filename)
    model.Alpha_ = sp.csr_matrix((loader['Alpha_data'],
                                 loader['Alpha_indices'],
                                 loader['Alpha_indptr']),
                                 shape=loader['Alpha_shape'])
    model.alpha_prior = loader['alpha_prior']
    model.mu_ = loader['mu_']
    model.num_iter = loader['num_iter']
    model.metropolis = loader['metropolis']
    model.num_jobs = loader['num_jobs']
    model.sloppy = loader['sloppy']
    model.beta_ = loader['beta']
    state = {}
    for id_ in range(model.mu_.shape[0]):
        state[id_] = loader['id_{}'.format(id_)]
    model.curr_state_ = state
    return model
