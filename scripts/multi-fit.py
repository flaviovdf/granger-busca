from gb import GrangerBusca
from gb import simulate

Alpha = [[0.5, 0.5, 0, 0],
         [0,   1,   0, 0],
         [0,   0,   0.5, 0.5],
         [0,   0,   0, 1]]
sim = simulate.GrangeBuscaSimulator([0.01]*4, Alpha)
ticks = sim.simulate(50000)

granger_model = GrangerBusca(alpha_prior=1.0/len(ticks), num_iter=300)
granger_model.fit(ticks)
print(granger_model.mu_)
print(granger_model.back_)
print(granger_model.Alpha_.toarray())
