"""Multi-dimensional fitness scoring for the Fly Evolution Engine.

Unlike the original single-score fitness(), each individual now
carries a FitnessProfile with five dimensions as described in
Etapa 1 of the roadmap:

    performance    — raw win rate
    consistency    — steady performance across matches
    robustness     — resilience to adversity (rallies, recovery)
    learning       — improvement within a lifetime
    generalization — performance against unseen opponents/styles

The composite score is a weighted sum of these dimensions,
configurable per experiment.
"""
from __future__ import annotations

from evolution.individual import FitnessProfile

DEFAULT_WEIGHTS = {
    "performance": 0.40,
    "consistency": 0.20,
    "robustness": 0.15,
    "learning": 0.15,
    "generalization": 0.10,
}

# Legacy constants for backward compatibility with existing code
WIN_WEIGHT = 10.0
POINTS_WEIGHT = 1.0
HITS_WEIGHT = 0.5
MAX_RALLY_WEIGHT = 0.3
LEARNING_GAIN_WEIGHT = 2.0


def fitness(*, won: bool, points_scored: int, hits: int, max_rally: int, learning_gain: float | None) -> float:
    """Legacy single-score fitness function (backward compatible)."""
    return (
        WIN_WEIGHT * (1.0 if won else 0.0)
        + POINTS_WEIGHT * points_scored
        + HITS_WEIGHT * hits
        + MAX_RALLY_WEIGHT * max_rally
        + LEARNING_GAIN_WEIGHT * (learning_gain or 0.0)
    )


def fitness_from_side_stats(side_stats, won: bool) -> float:
    """Convenience wrapper around evolution.duel_runner.SideStats."""
    return fitness(
        won=won,
        points_scored=side_stats.score,
        hits=side_stats.n_bounces,
        max_rally=side_stats.max_rally,
        learning_gain=side_stats.learning_gain,
    )


def fitness_profile_from_stats(won: bool, points_scored: int, hits: int, max_rally: int, learning_gain: float | None) -> FitnessProfile:
    """Create a multi-dimensional FitnessProfile from raw match stats."""
    profile = FitnessProfile()
    total = 1 if won else 0

    profile.performance = 1.0 if won else 0.0
    profile.consistency = min(1.0, points_scored / max(1, points_scored + 1))
    profile.robustness = min(1.0, hits / max(1, max_rally * 2)) if max_rally > 0 else 0.0
    profile.learning = max(0.0, min(1.0, (learning_gain or 0.0) / 2.0 + 0.5)) if learning_gain is not None else 0.0
    profile.generalization = min(1.0, hits / max(1, total * 3))

    return profile
