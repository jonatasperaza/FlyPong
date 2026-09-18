"""Individual = Genome + learned synaptic weights + multi-dimensional fitness profile.

Each individual wraps a Genome and carries:
  - synaptic_weights: the learned synaptic weights after a lifetime of plasticity
  - fitness: multi-dimensional fitness profile (performance, consistency, robustness, learning, generalization)
  - record: match statistics

Supports three inheritance modes for Etapa 2:
  - GENOME_ONLY: standard evolution (only hyperparameters inherited)
  - LAMARCKIAN: genome + learned weights inherited
  - EVOLVE_LEARN: genome seeds a fresh brain that then learns (plasticity during life)
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional
import numpy as np

from evolution.genome import Genome


class InheritanceMode(Enum):
    """How traits are passed to offspring."""
    GENOME_ONLY = "genome_only"       # Standard: only Genome (hyperparameters)
    LAMARCKIAN = "lamarckian"         # Genome + learned synaptic weights
    EVOLVE_LEARN = "evolve_learn"     # Genome seeds brain, plasticity during life


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
    avg_rally: float = 0.0


@dataclass
class EvaluatedResult:
    """Result of evaluating an individual, including learned weights."""
    genome_dict: dict
    fitness_value: float
    fitness_profile: FitnessProfile
    match_record: MatchRecord
    synaptic_weights: Optional[np.ndarray] = None
    weight_mean: float = 0.0
    weight_std: float = 0.0


@dataclass
class Individual:
    genome: Genome
    fitness: FitnessProfile = field(default_factory=FitnessProfile)
    record: MatchRecord = field(default_factory=MatchRecord)
    generation: int = 0
    id: int | None = None
    synaptic_weights: Optional[np.ndarray] = None
    inheritance_mode: InheritanceMode = InheritanceMode.GENOME_ONLY
    weight_mean: float = 0.0
    weight_std: float = 0.0

    @classmethod
    def from_genome(cls, genome: Genome, generation: int = 0, id: int | None = None, mode: InheritanceMode = InheritanceMode.GENOME_ONLY) -> "Individual":
        return cls(genome=genome, generation=generation, id=id, inheritance_mode=mode)

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
        return min(1.0, mean_score / 5.0)

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

    def add_match_result(self, won: bool, points_scored: int = 0, hits: int = 0, max_rally: int = 0, learning_gain: float | None = None, avg_rally: float = 0.0):
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
        self.record.avg_rally = avg_rally
        self.record.variance = self._compute_variance()

    def _compute_variance(self) -> float:
        total = self.record.wins + self.record.losses
        if total == 0:
            return 0.0
        win_rate = self.record.wins / total
        return win_rate * (1 - win_rate)

    def to_dict(self) -> dict:
        result = {
            "genome": self.genome.to_dict(),
            "fitness": self.fitness.to_dict(),
            "record": asdict(self.record),
            "generation": self.generation,
            "id": self.id,
            "inheritance_mode": self.inheritance_mode.value,
            "weight_mean": self.weight_mean,
            "weight_std": self.weight_std,
        }
        if self.synaptic_weights is not None:
            result["synaptic_weights"] = self.synaptic_weights.tolist()
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "Individual":
        weights = None
        if "synaptic_weights" in data and data["synaptic_weights"] is not None:
            weights = np.array(data["synaptic_weights"], dtype=np.float64)
        return cls(
            genome=Genome.from_dict(data["genome"]),
            fitness=FitnessProfile.from_dict(data.get("fitness", {})),
            record=MatchRecord(**data.get("record", {})),
            generation=data.get("generation", 0),
            id=data.get("id"),
            synaptic_weights=weights,
            inheritance_mode=InheritanceMode(data.get("inheritance_mode", "genome_only")),
            weight_mean=data.get("weight_mean", 0.0),
            weight_std=data.get("weight_std", 0.0),
        )
