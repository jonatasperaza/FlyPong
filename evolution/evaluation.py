"""Evaluation module for the Fly Evolution Engine — Etapa 2.

Extends basic evaluation to support:
  1. Standard evaluation (genome only, backward compatible)
  2. Weight extraction after evaluation (for Lamarckian inheritance)
  3. Detailed evaluation returning fitness profile + learned weights

Provides:
    Evaluator — configurable fitness evaluation of individuals
    WeightedEvaluationResult — structured result with fitness profile and weights
    evaluate_with_weights — evaluation that also returns synaptic weights
    extract_synaptic_weights — helper to get weights from a brain state
    get_weight_stats — helper for weight statistics
"""
from __future__ import annotations

import random
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

from evolution.genome import Genome
from evolution.individual import Individual, FitnessProfile, MatchRecord, InheritanceMode
from evolution.duel_runner import play_duel, make_duel, step_duel_frame, _BrainState, _max_rally, CALIBRATION_FRAMES
from evolution.fitness import fitness as _fitness_fn


@dataclass
class EvaluationResult:
    """Backward-compatible evaluation result."""
    genome_dict: dict
    left_stats: dict
    right_stats: dict
    fitness_left: float
    fitness_right: float


@dataclass
class WeightedEvaluationResult:
    """Evaluation result including learned synaptic weights."""
    genome_dict: dict
    left_stats: dict
    right_stats: dict
    fitness_value: float
    fitness_profile: FitnessProfile
    match_record: MatchRecord
    synaptic_weights: np.ndarray | None = None
    weight_mean: float = 0.0
    weight_std: float = 0.0
    left_brain_state: _BrainState | None = None
    right_brain_state: _BrainState | None = None


class Evaluator:
    """Evaluates individuals using fly-vs-fly duels.

    Supports standard evaluation (genome only) and weight extraction
    for Lamarckian inheritance (Etapa 2).
    """

    def __init__(self, frames: int = 3000, calibration_frames: int = 100, target_score: int | None = None, workers: int = 1):
        self.frames = frames
        self.calibration_frames = calibration_frames
        self.target_score = target_score
        self.workers = workers

    def evaluate_pair(self, genome_a: Genome, genome_b: Genome, seed: int) -> EvaluationResult:
        """Evaluate a pair of genomes in a duel. Backward compatible."""
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

    def evaluate_individual(self, individual: Individual, opponent: Individual, seed: int) -> float:
        """Evaluate an Individual against an opponent. Returns fitness score."""
        result = play_duel(
            individual.genome, opponent.genome, seed=seed,
            frames=self.frames,
            calibration_frames=self.calibration_frames,
            target_score=self.target_score,
        )
        return _fitness_from_stats(result.left) if result.left.score > result.right.score else _fitness_from_stats(result.right)

    def evaluate_with_weights(self, individual: Individual, opponent: Individual, seed: int) -> WeightedEvaluationResult:
        """Evaluate an Individual and extract learned synaptic weights.

        Uses make_duel + step_duel_frame to access brain states after
        evaluation, then extracts plastic weights from the network.
        """
        left_brain, right_brain, game = make_duel(
            individual.genome, opponent.genome, seed,
            calibration_frames=max(1, self.calibration_frames),
        )

        for _ in range(self.frames):
            step_duel_frame(left_brain, right_brain, game)
            if self.target_score is not None and (game.score_left >= self.target_score or game.score_right >= self.target_score):
                break

        # Extract learned weights from left brain
        plastic_weights = extract_synaptic_weights(left_brain)
        weight_mean, weight_std = get_weight_stats(plastic_weights) if plastic_weights is not None else (0.0, 0.0)

        # Compute stats from brain states
        left_stats = _brain_stats_to_dict(left_brain, game, is_left=True)
        right_stats = _brain_stats_to_dict(right_brain, game, is_left=False)

        fitness_value = _fitness_from_stats(_StatsProxy(left_stats))

        # Build fitness profile from stats
        profile = _build_fitness_profile(left_stats)

        # Build match record
        record = MatchRecord(
            wins=left_stats["wins"],
            losses=left_stats["losses"],
            hits=left_stats["hits"],
            max_rally=left_stats["max_rally"],
            learning_gain=left_stats["learning_gain"],
        )

        return WeightedEvaluationResult(
            genome_dict=individual.genome.to_dict(),
            left_stats=left_stats,
            right_stats=right_stats,
            fitness_value=fitness_value,
            fitness_profile=profile,
            match_record=record,
            synaptic_weights=plastic_weights,
            weight_mean=weight_mean,
            weight_std=weight_std,
            left_brain_state=left_brain,
            right_brain_state=right_brain,
        )

    def evaluate_batch(self, individuals: list, opponent, seeds: list[int]) -> list[float]:
        """Evaluate a batch of individuals against a common opponent."""
        if self.workers > 1:
            with ProcessPoolExecutor(max_workers=self.workers) as executor:
                payloads = [
                    (ind.genome.to_dict(), opp.genome.to_dict(), seed, self.frames, self.calibration_frames, self.target_score)
                    for ind, seed in zip(individuals, seeds)
                ]
                results = list(executor.map(_duel_worker, payloads))
        else:
            results = [_duel_worker((ind.genome.to_dict(), opp.genome.to_dict(), seed, self.frames, self.calibration_frames, self.target_score))
                       for ind, seed in zip(individuals, seeds)]
        return results

    def fitness_from_result(self, result: EvaluationResult) -> float:
        return max(result.fitness_left, result.fitness_right)


def _brain_stats_to_dict(brain: _BrainState, game, is_left: bool) -> dict:
    """Convert a brain state and game to a stats dict."""
    score = game.score_left if is_left else game.score_right
    n_bounces = brain.point_outcomes.count("bounce")
    n_misses = brain.point_outcomes.count("miss")
    max_rally = _max_rally(brain.point_outcomes)

    inicio = brain.bounce_rate_curve[0] if brain.bounce_rate_curve else None
    fim = brain.bounce_rate_curve[-1] if brain.bounce_rate_curve else None
    learning_gain = (fim - inicio) if (inicio is not None and fim is not None) else None

    return {
        "score": score,
        "n_bounces": n_bounces,
        "n_misses": n_misses,
        "max_rally": max_rally,
        "bounce_rate_inicio": inicio,
        "bounce_rate_fim": fim,
        "learning_gain": learning_gain,
        "wins": 1 if score > 0 else 0,
        "losses": 0 if score > 0 else 1,
    }


def _build_fitness_profile(stats: dict) -> FitnessProfile:
    """Build a FitnessProfile from brain stats dict."""
    total = stats["wins"] + stats["losses"]
    performance = stats["wins"] / total if total > 0 else 0.0

    total_matches = stats["wins"] + stats["losses"]
    consistency = min(1.0, stats["score"] / max(1, total_matches * 5)) if total_matches > 0 else 0.0

    max_rally = stats["max_rally"]
    hits = stats["n_bounces"]
    robustness = min(1.0, hits / max(1, max_rally * 2)) if max_rally > 0 else 0.0

    lg = stats["learning_gain"]
    learning = max(0.0, min(1.0, lg / 2.0 + 0.5)) if lg is not None else 0.0

    generalization = min(1.0, hits / max(1, total * 3)) if total > 0 else 0.0

    return FitnessProfile(
        performance=performance,
        consistency=consistency,
        robustness=robustness,
        learning=learning,
        generalization=generalization,
    )


def extract_synaptic_weights(brain: _BrainState | None) -> np.ndarray | None:
    """Extract learned synaptic weights from a brain's network."""
    if brain is None or not hasattr(brain, 'net'):
        return None
    if not hasattr(brain.net, 'W') or not hasattr(brain.net, 'plastic_data_idx'):
        return None
    if len(brain.net.plastic_data_idx) == 0:
        return None
    weights = brain.net.W.data[brain.net.plastic_data_idx].copy()
    return weights


def get_weight_stats(weights: np.ndarray | None) -> tuple[float, float]:
    """Return (mean, std) of a weight array."""
    if weights is None or len(weights) == 0:
        return 0.0, 0.0
    return float(np.mean(weights)), float(np.std(weights))


class _StatsProxy:
    """Adapt a stats dict to the SideStats interface for backward compatibility."""
    def __init__(self, stats: dict):
        self._stats = stats

    @property
    def score(self) -> int:
        return self._stats["score"]

    @property
    def n_bounces(self) -> int:
        return self._stats["n_bounces"]

    @property
    def max_rally(self) -> int:
        return self._stats["max_rally"]

    @property
    def learning_gain(self) -> float | None:
        return self._stats["learning_gain"]


def _fitness_from_stats(stats) -> float:
    """Convert SideStats-like object to a fitness score."""
    won = stats.score > 0
    return _fitness_fn(
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
    """Worker function for multiprocessing evaluation."""
    from evolution.duel_runner import play_duel
    from evolution.genome import Genome

    genome_a_dict, genome_b_dict, seed, frames, calibration_frames, target_score = payload
    ga = Genome.from_dict(genome_a_dict)
    gb = Genome.from_dict(genome_b_dict)
    result = play_duel(ga, gb, seed=seed, frames=frames, calibration_frames=calibration_frames, target_score=target_score)
    return {
        "left": _side_stats_dict(result.left),
        "right": _side_stats_dict(result.right),
        "left_score": result.left.score,
        "right_score": result.right.score,
    }
