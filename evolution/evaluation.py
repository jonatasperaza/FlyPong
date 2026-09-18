"""Evaluation module for the Fly Evolution Engine.

Wraps the existing duel_runner.play_duel logic into a clean
evaluation interface. Separates the "how do we measure fitness"
concern from the evolution loop.

Provides:
    Evaluator — configurable fitness evaluation of individuals
               against opponents or a fixed opponent
"""
from __future__ import annotations

import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Callable

from evolution.genome import Genome
from evolution.duel_runner import play_duel


@dataclass
class EvaluationResult:
    genome_dict: dict
    left_stats: dict
    right_stats: dict
    fitness_left: float
    fitness_right: float


class Evaluator:
    """Evaluates individuals using fly-vs-fly duels.

    Wraps evolution.duel_runner.play_duel and provides a clean
    interface for the evolution loop. Supports parallel evaluation
    via ProcessPoolExecutor.

    Example:
        evaluator = Evaluator(frames=3000, calibration_frames=100)
        result = evaluator.evaluate(individual, opponent)
        fitness = evaluator.fitness_from_result(result)
    """

    def __init__(self, frames: int = 3000, calibration_frames: int = 100, target_score: int | None = None, workers: int = 1):
        self.frames = frames
        self.calibration_frames = calibration_frames
        self.target_score = target_score
        self.workers = workers

    def evaluate_pair(self, genome_a: Genome, genome_b: Genome, seed: int) -> EvaluationResult:
        """Evaluate a pair of genomes in a duel. Returns structured results."""
        result = play_duel(
            genome_a, genome_b, seed=seed,
            frames=self.frames,
            calibration_frames=self.calibration_frames,
            target_score=self.target_score,
        )
        fitness_left = _fitness_from_stats(result.left)
        fitness_right = _fitness_from_stats(result.right)
        return EvaluationResult(
            genome_dict=genome_a.to_dict(),
            left_stats=_side_stats_dict(result.left),
            right_stats=_side_stats_dict(result.right),
            fitness_left=fitness_left,
            fitness_right=fitness_right,
        )

    def evaluate_individual(self, individual, opponent, seed: int) -> float:
        """Evaluate a single Individual against an opponent. Returns fitness score."""
        result = play_duel(
            individual.genome, opponent.genome, seed=seed,
            frames=self.frames,
            calibration_frames=self.calibration_frames,
            target_score=self.target_score,
        )
        return _fitness_from_stats(result.left) if result.left.score > result.right.score else _fitness_from_stats(result.right)

    def evaluate_batch(self, individuals: list, opponent, seeds: list[int]) -> list[float]:
        """Evaluate a batch of individuals against a common opponent."""
        if self.workers > 1:
            with ProcessPoolExecutor(max_workers=self.workers) as executor:
                payloads = [
                    (individual.genome.to_dict(), opponent.genome.to_dict(), seed, self.frames, self.calibration_frames, self.target_score)
                    for individual, seed in zip(individuals, seeds)
                ]
                results = list(executor.map(_duel_worker, payloads))
        else:
            results = [_duel_worker((ind.genome.to_dict(), opp.genome.to_dict(), seed, self.frames, self.calibration_frames, self.target_score))
                       for ind, seed in zip(individuals, seeds)]
        return results

    def fitness_from_result(self, result: EvaluationResult) -> float:
        """Extract a single fitness score from an evaluation result."""
        return max(result.fitness_left, result.fitness_right)


def _fitness_from_stats(stats) -> float:
    """Convert SideStats to a fitness score (backward compatible with old fitness function)."""
    from evolution.fitness import fitness
    won = stats.score > 0  # heuristic: score > 0 means it performed well
    return fitness(
        won=won,
        points_scored=stats.score,
        hits=stats.n_bounces,
        max_rally=stats.max_rally,
        learning_gain=stats.learning_gain,
    )


def _side_stats_dict(stats) -> dict:
    return {
        "score": stats.score,
        "n_bounces": stats.n_bounces,
        "max_rally": stats.max_rally,
        "bounce_rate_inicio": stats.bounce_rate_inicio,
        "bounce_rate_fim": stats.bounce_rate_fim,
        "learning_gain": stats.learning_gain,
    }


def _duel_worker(payload: tuple) -> dict:
    """Worker function for multiprocessing evaluation.

    Rebuilds everything from primitives (genome dicts, ints).
    """
    from evolution.duel_runner import play_duel
    from evolution.genome import Genome

    genome_a_dict, genome_b_dict, seed, frames, calibration_frames, target_score = payload
    ga = Genome.from_dict(genome_a_dict)
    gb = Genome.from_dict(genome_b_dict)
    result = play_duel(
        ga, gb, seed=seed, frames=frames,
        calibration_frames=calibration_frames,
        target_score=target_score,
    )
    return {
        "left": _side_stats_dict(result.left),
        "right": _side_stats_dict(result.right),
        "left_score": result.left.score,
        "right_score": result.right.score,
    }
