"""Modelo leaky integrate-and-fire (LIF) vetorizado em NumPy.

Simples e barato o bastante pra rodar milhares de neuronios em tempo real
(30-60Hz) sem depender de Brian2/NEST.
"""
import numpy as np


class LIFPopulation:
    def __init__(self, n, tau_m=10.0, v_rest=0.0, v_reset=0.0, v_threshold=1.0,
                 refractory_steps=2, dt=1.0):
        self.n = n
        self.tau_m = tau_m
        self.v_rest = v_rest
        self.v_reset = v_reset
        self.v_threshold = v_threshold
        self.refractory_steps = refractory_steps
        self.dt = dt

        self.v = np.full(n, v_rest, dtype=np.float64)
        self.refractory_timer = np.zeros(n, dtype=np.int32)
        self.spikes = np.zeros(n, dtype=bool)

    def reset(self):
        self.v[:] = self.v_rest
        self.refractory_timer[:] = 0
        self.spikes[:] = False

    def step(self, input_current):
        """Avanca um passo dt. Retorna array booleano de spikes."""
        active = self.refractory_timer <= 0
        dv = (self.dt / self.tau_m) * (-(self.v - self.v_rest) + input_current)
        self.v[active] += dv[active]

        self.spikes = active & (self.v >= self.v_threshold)
        self.v[self.spikes] = self.v_reset
        self.refractory_timer[self.spikes] = self.refractory_steps

        self.refractory_timer[~active] -= 1
        # neuronios fora do periodo refratario e sem esse decremento continuam com timer<=0
        return self.spikes
