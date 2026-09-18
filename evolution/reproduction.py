"""Reproduction operators for the Fly Evolution Engine.

Extracted from the inline breeding logic in evolve.py. Provides
crossover, mutation, and population replenishment as composable
functions.
"""
from __future__ import annotations

import random
from typing import Callable

from evolution.genome import Genome
from evolution.individual import Individual
from evolution.mutation import gaussian_mutation
from evolution.selection import truncation_select


def crossover(parent_a: Genome, parent_b: Genome, rng: random.Random) -> Genome:
    """Uniform crossover: each gene independently comes from parent_a or parent_b."""
    a, b = parent_a.to_dict(), parent_b.to_dict()
    return Genome(**{name: (a[name] if rng.random() < 0.5 else b[name]) for name in a})


def breed(population: list[Individual], rng: random.Random, target_size: int, immigration_rate: float = 0.05, mutation_rate: float = 0.3, mutation_strength: float = 0.2) -> list[Individual]:
    """Create the next generation from survivors via crossover + mutation.

    Strategy:
    1. Select top survivors (truncation selection, top 50% by default)
    2. Breed children via crossover + mutation
    3. Add random immigrants to maintain diversity

    Returns a new list of Individuals ready for the next generation.
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
        children.append(Individual.from_genome(child_genome))

    immigrants = [Individual.from_genome(Genome.random(rng)) for _ in range(n_immigrants)]
    return (children + immigrants)[:target_size]


def breed_with_elitism(population: list[Individual], rng: random.Random, target_size: int, immigration_rate: float = 0.05, elite_count: int = 2) -> list[Individual]:
    """Breed next generation preserving top `elite_count` individuals unchanged."""
    next_gen = []

    if elite_count > 0 and population:
        sorted_pop = sorted(population, key=lambda ind: ind.compute_fitness(), reverse=True)
        elite = sorted_pop[:min(elite_count, len(sorted_pop))]
        next_gen.extend(elite)

    remaining = target_size - len(next_gen)
    if remaining > 0:
        bred = breed(population, rng, remaining, immigration_rate=immigration_rate)
        next_gen.extend(bred)

    return next_gen[:target_size]


def create_random_population(size: int, rng: random.Random, generation: int = 0) -> list[Individual]:
    """Create an initial random population of Individuals."""
    return [Individual.from_genome(Genome.random(rng), generation=generation, id=i) for i in range(size)]
