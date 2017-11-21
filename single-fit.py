from gb import GrangerBusca
from gb import simulate

Alpha = [[1]]
sim = simulate.GrangeBuscaSimulator([0.0], [20], Alpha)
ticks = sim.simulate(5000000)

granger_model = GrangerBusca(alpha_p=1.0, num_iter=300, burn_in=100)
granger_model.fit(ticks)
print(granger_model.mu_)
print(granger_model.back_)
print(granger_model.beta_)
print(granger_model.Alpha_.toarray())
