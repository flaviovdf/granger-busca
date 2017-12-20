import numpy as np

from gb import simulate
from matplotlib import pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF

import math
import random


def SFP(n, mu, rho=1):
    # first inter-event time
    deltat = mu
    # list of inter-event times
    Deltat = []
    for i in range(1, n):
        # Poisson Process which Beta=deltat+mu/e
        deltat = -(deltat+(mu**rho)/math.e)
        deltat = deltat * math.log(random.random())
        Deltat.append(deltat**(1/rho))
    return Deltat


sim = simulate.GrangeBuscaSimulator([0.0], [10], [[1]])
ticks = sim.simulate(5000000)[0]
ticks2 = SFP(len(ticks), 10)
ticks2 = np.array(ticks2)

vals = np.ediff1d(ticks)
ecdf = ECDF(vals)
x_ticks = np.unique(vals)
plt.loglog(x_ticks, (1-ecdf(x_ticks)), label='busca')

vals = ticks2
ecdf = ECDF(vals)
x_ticks = np.unique(vals)
plt.loglog(x_ticks, (1-ecdf(x_ticks)), label='sfp')
plt.legend()
# plt.tight_layout(pad=0)
plt.xlabel(r'$\delta(t) - x$')
plt.ylabel(r'$P[X > x]$')
plt.savefig('compare.pdf')
plt.close()
