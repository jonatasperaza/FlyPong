"""Reproduction operators for the Fly Evolution Engine — Etapa 2.

Supports three inheritance modes:
  - GENOME_ONLY: standard evolution (only hyperparameters inherited)
  - LAMARCKIAN: genome + learned synaptic weights inherited
  - EVOLVE_LEARN: genome seeds a brain that then learns (plasticity during life)

Extracted from the inline breeding logic in evolve.py.
"""
from __future__ import annotations

import random
from typing import Optional
import numpy as np

from evolution.genome import Genome
from evolution.individual import Individual, InheritanceMode
from evolution.mutation import gaussian_mutation
from evolution.selection import truncation_select


def crossover(parent_a: Genome, parent_b: Genome, rng: random.Random) -> Genome:
    """Uniform crossover: each gene independently comes from parent_a or parent_b."""
    a, b = parent_a.to_dict(), parent_b.to_dict()
    return Genome(**{name: (a[name] if rng.random() < 0.5 else b[name]) for name in a})


def _inherit_weights(parent: Individual, child_genome: Genome, rng: random.Random, mode: InheritanceMode) -> tuple[Genome, Optional[np.ndarray]]:
    """Determine which genome and weights the child inherits.

    For LAMARCKIAN mode, the child inherits both the genome (via crossover)
    and the parent's learned synaptic weights (possibly perturbed).
    For EVOLVE_LEARN mode, the child gets the genome but starts with
    fresh/perturbed weights that will need to be learned anew.
    For GENOME_ONLY mode, only the genome matters.
    """
    child_weights = None

    if mode == InheritanceMode.LAMARCKIAN and parent.synaptic_weights is not None:
        # Inherit learned weights, with small perturbation
        child_weights = parent.synaptic_weights.copy()
        # Add Gaussian noise to inherited weights (mutation of weights)
        noise = rng.normal(0, 0.1, size=child_weights.shape)
        child_weights = child_weights + noise
        child_weights = np.clip(child_weights, -10.0, 10.0)  # reasonable bounds

    elif mode == InheritanceMode.EVOLVE_LEARN and parent.synaptic_weights is not None:
        # Start with parent's weights but they'll be overwritten by plasticity
        # during the child's lifetime. Use parent weights as initialization.
        child_weights = parent.synaptic_weights.copy()
        # Light perturbation so the child isn't identical to parent
        noise = rng.normal(0, 0.05, size=child_weights.shape)
        child_weights = child_weights + noise

    return child_genome, child_weights


def breed_genome_only(population: list[Individual], rng: random.Random, target_size: int, immigration_rate: float = 0.05, mutation_rate: float = 0.3, mutation_strength: float = 0.2) -> list[Individual]:
    """Standard evolution: only genome (hyperparameters) inherited."""
    survivors = truncation_select(population, fraction=0.5, rng=rng)
    if not survivors:
        survivors = population[: max(1, target_size // 2)]

    n_immigrants = max(1, round(target_size * immigration_rate))
    n_children = max(0, target_size - n_immigrants)

    children: list[Individual] = []
    for _ in range(n_children):
        if len(survivors) >= 2:
            parent_a = rng.choice(survivors)
            parent_b = rng.choice(survivors)
        else:
            parent_a = parent_b = survivors[0] if survivors else population[0]
        child_genome = crossover(parent_a.genome, parent_b.genome, rng)
        child_genome = gaussian_mutation(child_genome, rng, rate=mutation_rate, strength=mutation_strength)
        children.append(Individual.from_genome(child_genome, mode=InheritanceMode.GENOME_ONLY))

    immigrants = [Individual.from_genome(Genome.random(rng), mode=InheritanceMode.GENOME_ONLY) for _ in range(n_immigrants)]
    return (children + immigrants)[:target_size]


def breed_lamarckian(population: list[Individual], rng: random.Random, target_size: int, immigration_rate: float = 0.05, mutation_rate: float = 0.3, mutation_strength: float = 0.2) -> list[Individual]:
    """Lamarckian reproduction: genome + learned synaptic weights inherited.

    The child receives:
    1. Crossover of parent genomes + mutation
    2. Parent's learned synaptic weights (perturbed)
    3. The child doesn't need to re-learn — the weights carry over
    """
    survivors = truncation_select(population, fraction=0.5, rng=rng)
    if not survivors:
        survivors = population[: max(1, target_size // 2)]

    n_immigrants = max(1, round(target_size * immigration_rate))
    n_children = max(0, target_size - n_immigrants)

    children: list[Individual] = []
    for _ in range(n_children):
        if len(survivors) >= 2:
            parent_a = rng.choice(survivors)
            parent_b = rng.choice(survivors)
        else:
            parent_a = parent_b = survivors[0] if survivors else population[0]

        child_genome = crossover(parent_a.genome, parent_b.genome, rng)
        child_genome = gaussian_mutation(child_genome, rng, rate=mutation_rate, strength=mutation_strength)
        child_genome, child_weights = _inherit_weights(parent_a, child_genome, rng, InheritanceMode.LAMARCKIAN)

        child = Individual.from_genome(child_genome, mode=InheritanceMode.LAMARCKIAN)
        child.synaptic_weights = child_weights
        if child_weights is not None:
            child.weight_mean = float(np.mean(child_weights))
            child.weight_std = float(np.std(child_weights))
        children.append(child)

    immigrants = [Individual.from_genome(Genome.random(rng), mode=InheritanceMode.LAMARCKIAN) for _ in range(n_immigrants)]
    return (children + immigrants)[:target_size]


def breed_evolve_learn(population: list[Individual], rng: random.Random, target_size: int, immigration_rate: float = 0.05) -> list[Individual]:
    """Evolution + learning reproduction.

    The child receives:
    1. Crossover of parent genomes + mutation (as standard)
    2. Parent's learned weights as initialization (but will be overwritten
       by plasticity during the child's lifetime)
    
    This models: the genome gives the brain structure, but the brain
    must still learn during its own lifetime. The inherited weights
    serve as a "warm start" for learning.
    """
    survivors = truncation_select(population, fraction=0.5, rng=rng)
    if not survivors:
        survivors = population[: max(1, target_size // 2)]

    n_immigrants = max(1, round(target_size * immigration_rate))
    n_children = max(0, target_size - n_immigrants)

    children: list[Individual] = []
    for _ in range(n_children):
        if len(survivors) >= 2:
            parent_a = rng.choice(survivors)
            parent_b = rng.choice(survivors)
        else:
            parent_a = parent_b = survivors[0] if survivors else population[0]

        child_genome = crossover(parent_a.genome, parent_b.genome, rng)
        child_genome = gaussian_mutation(child_genome, rng, rate=0.3, strength=0.2)
        child_genome, child_weights = _inherit_weights(parent_a, child_genome, rng, InheritanceMode.EVOLVE_LEARN)

        child = Individual.from_genome(child_genome, mode=InheritanceMode.EVOLVE_LEARN)
        child.synaptic_weights = child_weights
        children.append(child)

    immigrants = [Individual.from_genome(Genome.random(rng), mode=InheritanceMode.EVOLVE_LEARN) for _ in range(n_immigrants)]
    return (children + immigrants)[:target_size]


def breed(population: list[Individual], rng: random.Random, target_size: int, immigration_rate: float = 0.05, mutation_rate: float = 0.3, mutation_strength: float = 0.2, mode: InheritanceMode = InheritanceMode.GENOME_ONLY) -> list[Individual]:
    """Create the next generation using the specified inheritance mode.

    This is the main entry point — delegates to the appropriate breeding
    function based on the inheritance mode.
    """
    if mode == InheritanceMode.LAMARCKIAN:
        return breed_lamarckian(population, rng, target_size, immigration_rate, mutation_rate, mutation_strength)
    elif mode == InheritanceMode.EVOLVE_LEARN:
        return breed_evolve_learn(population, rng, target_size, immigration_rate)
    else:
        return breed_genome_only(population, rng, target_size, immigration_rate, mutation_rate, mutation_strength)


def breed_with_elitism(population: list[Individual], rng: random.Random, target_size: int, immigration_rate: float = 0.05, elite_count: int = 2, mode: InheritanceMode = InheritanceMode.GENOME_ONLY) -> list[Individual]:
    """Breed next generation preserving top `elite_count` individuals unchanged."""
    next_gen = []

    if elite_count > 0 and population:
        sorted_pop = sorted(population, key=lambda ind: ind.compute_fitness(), reverse=True)
        elite = sorted_pop[:min(elite_count, len(sorted_pop))]
        next_gen.extend(elite)

    remaining = target_size - len(next_gen)
    if remaining > 0:
        bred = breed(population, rng, remaining, immigration_rate=immigration_rate, mode=mode)
        next_gen.extend(bred)

    return next_gen[:target_size]


def create_random_population(size: int, rng: random.Random, generation: int = 0, mode: InheritanceMode = InheritanceMode.GENOME_ONLY) -> list[Individual]:
    """Create an initial random population of Individuals."""
    return [Individual.from_genome(Genome.random(rng), generation=generation, id=i, mode=mode) for i in range(size)]
