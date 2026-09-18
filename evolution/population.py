"""Population management for the Fly Evolution Engine.

Manages a collection of Individuals, providing methods for
ranking, selection pressure application, and replacement strategies.
"""
from __future__ import annotations

import random
from typing import Callable

from evolution.genome import Genome
from evolution.individual import Individual


class Population:
    """Manages a population of Individuals through evolutionary generations."""

    def __init__(self, individuals: list[Individual] | None = None, generation: int = 0):
        self.individuals: list[Individual] = individuals or []
        self.generation = generation
        self._id_counter = len(self.individuals)

    @classmethod
    def random(cls, size: int, rng: random.Random, generation: int = 0) -> "Population":
        """Create a random population."""
        individuals = [
            Individual.from_genome(Genome.random(rng), generation=generation, id=i)
            for i in range(size)
        ]
        return cls(individuals, generation=generation)

    @property
    def size(self) -> int:
        return len(self.individuals)

    @property
    def is_empty(self) -> bool:
        return len(self.individuals) == 0

    def add(self, individual: Individual) -> None:
        """Add an individual to the population."""
        self.individuals.append(individual)

    def remove(self, individual: Individual) -> None:
        """Remove an individual from the population."""
        self.individuals.remove(individual)

    def ranked(self) -> list[Individual]:
        """Return individuals sorted by fitness (descending)."""
        return sorted(self.individuals, key=lambda ind: ind.compute_fitness(), reverse=True)

    def best(self) -> Individual | None:
        """Return the fittest individual, or None if empty."""
        ranked = self.ranked()
        return ranked[0] if ranked else None

    def worst(self) -> Individual | None:
        """Return the least fit individual, or None if empty."""
        ranked = self.ranked(reverse=True)
        return ranked[-1] if ranked else None

    def mean_fitness(self) -> float:
        """Return the mean composite fitness of the population."""
        if not self.individuals:
            return 0.0
        return sum(ind.compute_fitness() for ind in self.individuals) / len(self.individuals)

    def median_fitness(self) -> float:
        """Return the median composite fitness."""
        if not self.individuals:
            return 0.0
        sorted_fitness = sorted(ind.compute_fitness() for ind in self.individuals)
        mid = len(sorted_fitness) // 2
        if len(sorted_fitness) % 2 == 0:
            return (sorted_fitness[mid - 1] + sorted_fitness[mid]) / 2
        return sorted_fitness[mid]

    def best_fitness(self) -> float:
        """Return the best fitness in the population."""
        best = self.best()
        return best.compute_fitness() if best else 0.0

    def diversity(self) -> float:
        """Measure population diversity as average genome Hamming distance.

        Returns a value between 0 (all identical) and 1 (maximally different).
        """
        if len(self.individuals) < 2:
            return 0.0
        genomes = [ind.genome for ind in self.individuals]
        total_diff = 0
        pairs = 0
        for i in range(len(genomes)):
            for j in range(i + 1, len(genomes)):
                total_diff += self._genome_diff(genomes[i], genomes[j])
                pairs += 1
        return total_diff / max(1, pairs)

    @staticmethod
    def _genome_diff(ga: Genome, gb: Genome) -> float:
        """Compute normalized difference between two genomes."""
        a, b = ga.to_dict(), gb.to_dict()
        if not a:
            return 0.0
        diffs = sum(1 for k in a if abs(a[k] - b.get(k, a[k])) > 1e-10)
        return diffs / len(a)

    def truncate(self, fraction: float) -> list[Individual]:
        """Return the top fraction of the population by fitness."""
        return self.ranked()[: max(1, int(len(self.individuals) * fraction))]

    def select(self, count: int, strategy: str = "tournament", rng: random.Random | None = None) -> list[Individual]:
        """Select `count` individuals from the population."""
        from evolution.selection import select_many
        return select_many(self.individuals, count, strategy=strategy, rng=rng or random.Random())

    def replace(self, new_individuals: list[Individual]) -> None:
        """Replace the entire population."""
        self.individuals = new_individuals
        self.generation += 1

    def to_dict(self) -> dict:
        return {
            "generation": self.generation,
            "size": self.size,
            "individuals": [ind.to_dict() for ind in self.individuals],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Population":
        individuals = [Individual.from_dict(d) for d in data.get("individuals", [])]
        return cls(individuals, generation=data.get("generation", 0))
