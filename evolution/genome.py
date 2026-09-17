"""Genome for the FlyPong evolutionary tournament.

Only hyperparameters are heritable. Learned synaptic weights are never
carried between generations (Lamarckian inheritance of learned weights
would confound "the genome is good" with "this individual got lucky
during its own lifetime"), so a fresh brain is built from each genome at
the start of every match.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import random


@dataclass
class Genome:
    plastic_lr: float
    synthetic_w_init: float
    dopamine_baseline: float
    reward_sensitivity: float
    weight_scale: float
    noise_sigma: float
    action_threshold: float

    # (min, max) used both to sample a fresh random genome and to clamp
    # a value after mutation, so a genome can never drift outside the
    # range the rest of the simulation was validated against.
    BOUNDS = {
        "plastic_lr": (0.001, 0.05),
        "synthetic_w_init": (5.0, 30.0),
        "dopamine_baseline": (0.0, 0.5),
        "reward_sensitivity": (0.5, 3.0),
        "weight_scale": (0.5, 2.0),
        "noise_sigma": (0.0, 0.3),
        "action_threshold": (0.3, 0.7),
    }

    @classmethod
    def random(cls, rng: random.Random) -> "Genome":
        return cls(**{name: rng.uniform(*bounds) for name, bounds in cls.BOUNDS.items()})

    def _clamp(self, name: str, value: float) -> float:
        lo, hi = self.BOUNDS[name]
        return min(hi, max(lo, value))

    def mutate(self, rng: random.Random, rate: float = 0.3, strength: float = 0.2) -> "Genome":
        """Return a mutated copy; self is left untouched.

        Each gene independently has probability `rate` of being perturbed
        by Gaussian noise scaled to its own magnitude (so a gene near 0.01
        and one near 30.0 both get proportionally sized steps), then
        clamped back into BOUNDS.
        """
        values = asdict(self)
        for name, value in values.items():
            if rng.random() < rate:
                delta = rng.gauss(0.0, strength * abs(value) + 1e-6)
                values[name] = self._clamp(name, value + delta)
        return Genome(**values)

    def crossover(self, other: "Genome", rng: random.Random) -> "Genome":
        """Uniform crossover: each gene independently comes from self or other."""
        a, b = asdict(self), asdict(other)
        return Genome(**{name: (a[name] if rng.random() < 0.5 else b[name]) for name in a})

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Genome":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})
