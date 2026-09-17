"""Optimized vectorized leaky integrate-and-fire population."""
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
        self._dv = np.zeros(n, dtype=np.float64)

    def reset(self):
        self.v.fill(self.v_rest)
        self.refractory_timer.fill(0)
        self.spikes.fill(False)

    def step(self, input_current):
        active = self.refractory_timer <= 0
        np.subtract(self.v, self.v_rest, out=self._dv)
        self._dv *= -1.0
        self._dv += input_current
        self._dv *= self.dt / self.tau_m
        self.v[active] += self._dv[active]

        self.spikes = active & (self.v >= self.v_threshold)
        self.v[self.spikes] = self.v_reset
        self.refractory_timer[self.spikes] = self.refractory_steps
        self.refractory_timer[~active] -= 1
        return self.spikes
