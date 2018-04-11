from gb import GrangerBusca
from gb import simulate

Alpha = [[1]]
sim = simulate.GrangeBuscaSimulator([0.01], Alpha)
ticks = sim.simulate(500000)

granger_model = GrangerBusca(alpha_prior=1.0, num_iter=300)
granger_model.fit(ticks)
print(granger_model.mu_)
print(granger_model.back_)
print(granger_model.Alpha_.toarray())
