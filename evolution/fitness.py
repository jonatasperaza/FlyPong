"""Fitness scoring for one fly's side-stats from one or more duels.

Weights follow the handoff proposal discussed with the user: winning
matters most, but a fly that never wins should still be pulled toward
"got closer" (points scored), "made contact" (hits/bounces), sustained a
rally, or improved within its own lifetime (learning_gain) -- otherwise
early generations, where nobody wins yet, would all score zero and
selection would have nothing to select on.
"""
from __future__ import annotations

WIN_WEIGHT = 10.0
POINTS_WEIGHT = 1.0
HITS_WEIGHT = 0.5
MAX_RALLY_WEIGHT = 0.3
LEARNING_GAIN_WEIGHT = 2.0


def fitness(*, won: bool, points_scored: int, hits: int, max_rally: int, learning_gain: float | None) -> float:
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
