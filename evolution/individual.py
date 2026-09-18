"""Individual = Genome + multi-dimensional fitness profile.

Each individual wraps a Genome and carries a fitness profile with
five dimensions as described in Etapa 1 of the roadmap:
performance, consistency, robustness, learning, generalization.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

from evolution.genome import Genome


@dataclass
class FitnessProfile:
    performance: float = 0.0
    consistency: float = 0.0
    robustness: float = 0.0
    learning: float = 0.0
    generalization: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "FitnessProfile":
        return cls(**{k: data.get(k, 0.0) for k in ["performance", "consistency", "robustness", "learning", "generalization"]})

    def composite(self, weights: dict[str, float] | None = None) -> float:
        if weights is None:
            weights = {
                "performance": 0.40,
                "consistency": 0.20,
                "robustness": 0.15,
                "learning": 0.15,
                "generalization": 0.10,
            }
        return sum(getattr(self, d) * weights[k] for k, d in [("performance", "performance"), ("consistency", "consistency"), ("robustness", "robustness"), ("learning", "learning"), ("generalization", "generalization")])


@dataclass
class MatchRecord:
    wins: int = 0
    losses: int = 0
    draws: int = 0
    points_scored: int = 0
    hits: int = 0
    max_rally: int = 0
    learning_gain: float = 0.0
    variance: float = 0.0


@dataclass
class Individual:
    genome: Genome
    fitness: FitnessProfile = field(default_factory=FitnessProfile)
    record: MatchRecord = field(default_factory=MatchRecord)
    generation: int = 0
    id: int | None = None

    def __post_init__(self):
        if self.id is None and hasattr(self.genome, "_default_id_counter"):
            self.id = self._next_id()

    @classmethod
    def from_genome(cls, genome: Genome, generation: int = 0, id: int | None = None) -> "Individual":
        return cls(genome=genome, generation=generation, id=id)

    def compute_fitness(self, weights: dict[str, float] | None = None) -> float:
        self.fitness.performance = self._compute_performance()
        self.fitness.consistency = self._compute_consistency()
        self.fitness.robustness = self._compute_robustness()
        self.fitness.learning = self._compute_learning()
        self.fitness.generalization = self._compute_generalization()
        return self.fitness.composite(weights)

    def _compute_performance(self) -> float:
        total = self.record.wins + self.record.losses
        if total == 0:
            return 0.0
        return self.record.wins / total

    def _compute_consistency(self) -> float:
        total_matches = self.record.wins + self.record.losses + self.record.draws
        if total_matches == 0:
            return 0.0
        mean_score = self.record.points_scored / total_matches
        return min(1.0, mean_score / 5.0) if total_matches > 0 else 0.0

    def _compute_robustness(self) -> float:
        if self.record.max_rally == 0:
            return 0.0
        return min(1.0, self.record.hits / max(1, self.record.max_rally * 2))

    def _compute_learning(self) -> float:
        if self.record.learning_gain is None:
            return 0.0
        return max(0.0, min(1.0, self.record.learning_gain / 2.0 + 0.5))

    def _compute_generalization(self) -> float:
        if self.record.hits == 0:
            return 0.0
        return min(1.0, self.record.hits / max(1, (self.record.wins + self.record.losses) * 3))

    def add_match_result(self, won: bool, points_scored: int = 0, hits: int = 0, max_rally: int = 0, learning_gain: float | None = None):
        if won:
            self.record.wins += 1
        else:
            self.record.losses += 1
        self.record.points_scored += points_scored
        self.record.hits += hits
        if max_rally > self.record.max_rally:
            self.record.max_rally = max_rally
        if learning_gain is not None:
            self.record.learning_gain += learning_gain
        self.record.variance = self._compute_variance()

    def _compute_variance(self) -> float:
        total = self.record.wins + self.record.losses
        if total == 0:
            return 0.0
        win_rate = self.record.wins / total
        return win_rate * (1 - win_rate)

    @property
    def win_rate(self) -> float:
        total = self.record.wins + self.record.losses
        return self.record.wins / total if total > 0 else 0.0

    @property
    def is_viable(self) -> bool:
        return (self.record.wins + self.record.losses) >= 1

    def to_dict(self) -> dict:
        return {
            "genome": self.genome.to_dict(),
            "fitness": self.fitness.to_dict(),
            "record": asdict(self.record),
            "generation": self.generation,
            "id": self.id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Individual":
        return cls(
            genome=Genome.from_dict(data["genome"]),
            fitness=FitnessProfile.from_dict(data.get("fitness", {})),
            record=MatchRecord(**data.get("record", {})),
            generation=data.get("generation", 0),
            id=data.get("id"),
        )
