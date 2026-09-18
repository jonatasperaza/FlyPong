"""Mutation operators for the Fly Evolution Engine.

Extracted from Genome.mutate() for modularity. Each function
operates on a Genome independently, making it easy to mix and
match mutation strategies.
"""
from __future__ import annotations

from evolution.genome import Genome


def gaussian_mutation(genome: Genome, rng, rate: float = 0.3, strength: float = 0.2) -> Genome:
    """Apply Gaussian perturbation to each gene independently.

    Each gene has probability `rate` of being mutated by
    `rng.gauss(0, strength * |gene| + epsilon)`, then clamped
    back into BOUNDS.
    """
    from dataclasses import asdict
    values = asdict(genome)
    for name, value in values.items():
        if rng.random() < rate:
            delta = rng.gauss(0.0, strength * abs(value) + 1e-6)
            lo, hi = Genome.BOUNDS[name]
            values[name] = min(hi, max(lo, value + delta))
    return Genome(**values)


def uniform_mutation(genome: Genome, rng, rate: float = 0.3) -> Genome:
    """Replace mutated genes with uniform random values within BOUNDS."""
    from dataclasses import asdict
    values = asdict(genome)
    for name, value in values.items():
        if rng.random() < rate:
            lo, hi = Genome.BOUNDS[name]
            values[name] = rng.uniform(lo, hi)
    return Genome(**values)


def bitflip_mutation(genome: Genome, rng, rate: float = 0.05) -> Genome:
    """Not applicable to continuous genomes; returns genome unchanged.

    Provided for API compatibility with discrete-genome strategies.
    """
    return genome


def adaptive_mutation(genome: Genome, rng, generation: int, max_generations: int, base_rate: float = 0.3, base_strength: float = 0.2) -> Genome:
    """Mutation rate and strength adapt over generations.

    Rate decreases and strength decreases as the population converges,
    allowing fine-tuning in later generations.
    """
    progress = generation / max(max_generations, 1)
    rate = base_rate * (1.0 - progress * 0.5)  # decreases from base_rate to base_rate*0.5
    strength = base_strength * (1.0 - progress * 0.5)  # decreases similarly
    return gaussian_mutation(genome, rng, rate=rate, strength=strength)


def mutate(genome: Genome, rng, rate: float = 0.3, strength: float = 0.2) -> Genome:
    """Convenience wrapper: delegates to gaussian_mutation."""
    return gaussian_mutation(genome, rng, rate=rate, strength=strength)
