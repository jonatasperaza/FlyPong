"""Hall of Fame management for the Fly Evolution Engine.

Tracks the best individuals across all generations. The Hall of Fame
serves two purposes:
1. Preserve champions so the population cannot "forget" how to beat
   a past champion by cycling past it (Rosin & Belew 1997).
2. Provide opponents that test the current population against
   historical best performers.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evolution.genome import Genome
from evolution.individual import Individual


class HallOfFame:
    """Maintains a bounded collection of the best individuals ever seen."""

    def __init__(self, max_size: int = 3):
        self.max_size = max_size
        self._entries: list[Individual] = []
        self._genome_keys: list[str] = []

    @property
    def entries(self) -> list[Individual]:
        return list(self._entries)

    @property
    def champion(self) -> Individual | None:
        """Return the all-time best individual, or None if empty."""
        return self._entries[0] if self._entries else None

    def add(self, individual: Individual) -> bool:
        """Add an individual if it's better than the worst entry (or if there's room).

        Returns True if the individual was added, False if rejected.
        """
        genome_key = json.dumps(individual.genome.to_dict(), sort_keys=True)

        if genome_key in self._genome_keys:
            return False

        if len(self._entries) < self.max_size:
            self._entries.append(individual)
            self._genome_keys.append(genome_key)
            self._entries.sort(key=lambda ind: ind.compute_fitness(), reverse=True)
            return True

        if individual.compute_fitness() > self._entries[-1].compute_fitness():
            self._entries[-1] = individual
            self._genome_keys[-1] = genome_key
            self._entries.sort(key=lambda ind: ind.compute_fitness(), reverse=True)
            return True

        return False

    def add_genome(self, genome: Genome, generation: int = 0, id: int | None = None) -> bool:
        """Add a Genome directly, wrapping it in an Individual."""
        return self.add(Individual.from_genome(genome, generation=generation, id=id))

    def get_opponents(self, count: int | None = None) -> list[Individual]:
        """Return up to `count` Hall-of-Fame entries as opponents."""
        n = count if count is not None else len(self._entries)
        return self._entries[:min(n, len(self._entries))]

    def clear(self) -> None:
        """Reset the Hall of Fame."""
        self._entries.clear()
        self._genome_keys.clear()

    def to_dict(self) -> dict:
        return {
            "max_size": self.max_size,
            "entries": [ind.to_dict() for ind in self._entries],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "HallOfFame":
        hof = cls(max_size=data.get("max_size", 3))
        hof._entries = [Individual.from_dict(d) for d in data.get("entries", [])]
        hof._genome_keys = [json.dumps(d["genome"], sort_keys=True) for d in data.get("entries", [])]
        return hof

    def save(self, path: Path) -> None:
        """Persist the Hall of Fame to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "HallOfFame":
        """Load a Hall of Fame from a JSON file."""
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data)
