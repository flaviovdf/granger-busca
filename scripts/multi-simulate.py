import matplotlib
matplotlib.use('PDF')

import numpy as np

from gb import simulate
from matplotlib import pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF

Alpha = [[0.25, 0.25, 0.25, 0.25],
         [0, 1, 0, 0],
         [0, 0, 0.2, 0.8],
         [0, 1, 0, 0]]
sim = simulate.GrangeBuscaSimulator([0.001]*4, Alpha)
ticks = sim.simulate(10000)

counting = [np.arange(len(ticks[i])) for i in range(len(ticks))]

# plt.subplot(211)
# for i in range(len(ticks)):
#     plt.plot(ticks[i], counting[i], label='%d' % i)
# plt.legend()

# plt.subplot(212)
for i in range(len(ticks)):
    vals = np.ediff1d(ticks[i])
    ecdf = ECDF(vals)
    x_ticks = np.unique(vals)
    plt.loglog(x_ticks, (1-ecdf(x_ticks)), label='%d' % i)
plt.legend()
# plt.tight_layout(pad=0)
plt.xlabel(r'$\delta(t) - x$')
plt.ylabel(r'$P[X > x]$')
plt.savefig('multi.pdf')
plt.close()
