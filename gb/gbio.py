# -*- coding: utf8

from gb import GrangerBusca

from scipy import sparse as sp

import numpy as np


def save_model(filename, granger_model):
    state = {}
    for key, val in granger_model.curr_state_.items():
        state['id_{}'.format(key)] = np.array(val)

    np.savez_compressed(filename,
                        Alpha_data=granger_model.Alpha_.data,
                        Alpha_indices=granger_model.Alpha_.indices,
                        Alpha_indptr=granger_model.Alpha_.indptr,
                        Alpha_shape=granger_model.Alpha_.shape,
                        alpha_p=granger_model.alpha_p,
                        beta_=granger_model.beta_,
                        burn_in=granger_model.burn_in,
                        mu_=granger_model.mu_,
                        num_iter=granger_model.num_iter,
                        **state)


def load_model(filename):
    model = GrangerBusca(0, 0, 0)
    loader = np.load(filename)
    model.Alpha_ = sp.csr_matrix((loader['Alpha_data'],
                                 loader['Alpha_indices'],
                                 loader['Alpha_indptr']),
                                 shape=loader['Alpha_shape'])
    model.alpha_p = loader['alpha_p']
    model.beta_ = loader['beta_']
    model.burn_in = loader['burn_in']
    model.mu_ = loader['mu_']
    model.num_iter = loader['num_iter']
    state = {}
    for id_ in range(model.mu_.shape[0]):
        state[id_] = loader['id_{}'.format(id_)]
    model.curr_state_ = state
    return model
