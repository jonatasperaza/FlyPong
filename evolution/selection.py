"""Selection strategies for the Fly Evolution Engine.

Provides tournament selection, roulette wheel, rank-based, and
truncation selection strategies. All strategies operate on a list
of (entity_id, score) pairs or Individual objects.
"""
from __future__ import annotations

import random
from typing import Callable, Sequence, TypeVar

from evolution.individual import Individual

T = TypeVar("T")


def tournament_select(population: Sequence[Individual], k: int = 3, rng: random.Random | None = None) -> Individual:
    """Tournament selection: pick k random individuals, return the best."""
    if rng is None:
        rng = random.Random()
    contestants = rng.sample(list(population), min(k, len(population)))
    return max(contestants, key=lambda ind: ind.compute_fitness())


def roulette_select(population: Sequence[Individual], rng: random.Random | None = None) -> Individual:
    """Fitness-proportionate (roulette wheel) selection.

    Handles negative fitness by shifting all scores to be positive.
    """
    if rng is None:
        rng = random.Random()
    scores = [ind.compute_fitness() for ind in population]
    min_score = min(scores) if scores else 0.0
    shifted = [s - min_score + 1e-6 for s in scores]
    total = sum(shifted)
    if total <= 0:
        return rng.choice(list(population))
    pick = rng.uniform(0, total)
    cumulative = 0.0
    for ind, s in zip(population, shifted):
        cumulative += s
        if cumulative >= pick:
            return ind
    return population[-1]


def rank_select(population: Sequence[Individual], rng: random.Random | None = None) -> Individual:
    """Rank-based selection: probability proportional to rank, not raw score.

    Best individual gets highest probability, worst gets lowest.
    This reduces the dominance of super-individuals early on.
    """
    if rng is None:
        rng = random.Random()
    ranked = sorted(population, key=lambda ind: ind.compute_fitness(), reverse=True)
    n = len(ranked)
    if n == 0:
        raise ValueError("Cannot select from empty population")
    # Linear ranking probabilities: n/n*(n+1)/2, ..., 1/n*(n+1)/2
    total = n * (n + 1) / 2
    pick = rng.uniform(0, total)
    cumulative = 0.0
    for i, ind in enumerate(ranked):
        cumulative += (n - i) / total
        if cumulative >= pick:
            return ind
    return ranked[-1]


def truncation_select(population: Sequence[Individual], fraction: float = 0.5, rng: random.Random | None = None) -> list[Individual]:
    """Select the top `fraction` of the population by fitness."""
    if rng is None:
        rng = random.Random()
    sorted_pop = sorted(population, key=lambda ind: ind.compute_fitness(), reverse=True)
    n = max(1, int(len(sorted_pop) * fraction))
    return sorted_pop[:n]


def select(population: Sequence[Individual], strategy: str = "tournament", **kwargs) -> Individual:
    """Dispatch to the named selection strategy."""
    strategies: dict[str, Callable] = {
        "tournament": lambda pop, rng: tournament_select(pop, rng=rng, **kwargs),
        "roulette": lambda pop, rng: roulette_select(pop, rng=rng, **kwargs),
        "rank": lambda pop, rng: rank_select(pop, rng=rng, **kwargs),
    }
    rng = kwargs.pop("rng", None)
    if strategy not in strategies:
        raise ValueError(f"Unknown selection strategy: {strategy}")
    return strategies[strategy](list(population), rng)


def select_many(population: Sequence[Individual], count: int, strategy: str = "tournament", rng: random.Random | None = None, **kwargs) -> list[Individual]:
    """Select `count` individuals using the named strategy, with replacement."""
    if rng is None:
        rng = random.Random()
    return [select(population, strategy=strategy, rng=rng, **kwargs) for _ in range(count)]
