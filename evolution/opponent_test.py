"""Generalization evaluation module for Etapa 3 — Teste contra estilos diversos.

A ideia central: uma mosca que sofreu treinamento contra um tipo de
adversario nao e necessariamente boa contra outros. Este modulo
avalia a generalizacao do campeao contra todos os 7 estilos.

Funciona com ou sem dados do conectoma (usa avaliacao leve quando
connectome_data nao esta disponivel).

Uso:
    from evolution.opponent_test import OpponentTest, evaluate_generalization

    tester = OpponentTest(frames=3000)
    result = tester.test_against_all_styles(champion_genome_dict)
    print(f"Generalization score: {result.generalization_score:.4f}")
    for style, score in result.style_scores.items():
        print(f"  {style}: {score:.4f}")
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

from game.pong import PongGame
from game.opponent import OpponentStyle, create_opponent_strategy, OpponentStyleConfig


@dataclass
class StyleTestResult:
    """Resultado do teste contra um estilo de adversario."""
    style: str
    win_rate: float = 0.0
    mean_bounce_rate: float = 0.0
    mean_rally: float = 0.0
    fitness: float = 0.0
    performance: float = 0.0
    consistency: float = 0.0
    robustness: float = 0.0
    learning: float = 0.0
    generalization: float = 0.0
    score_left: int = 0
    score_right: int = 0
    n_bounces: int = 0
    n_misses: int = 0
    max_rally: int = 0


@dataclass
class GeneralizationResult:
    """Resultado completo do teste de generalizacao."""
    style_scores: dict[str, StyleTestResult] = field(default_factory=dict)
    generalization_score: float = 0.0
    best_style: str = ""
    worst_style: str = ""
    std_dev: float = 0.0
    score_variance: float = 0.0

    @property
    def min_score(self) -> float:
        if not self.style_scores:
            return 0.0
        return min(s.fitness for s in self.style_scores.values())

    @property
    def max_score(self) -> float:
        if not self.style_scores:
            return 0.0
        return max(s.fitness for s in self.style_scores.values())

    @property
    def range_score(self) -> float:
        return self.max_score - self.min_score


class OpponentTest:
    """Testa o desempenho de um adversario contra todos os 7 estilos.

    Usa PongGame com opponent_strategy para cada estilo, medindo
    performance, generalizacao e variabilidade entre estilos.
    """

    def __init__(self, frames: int = 3000, calibration_frames: int = 100, target_score: int | None = None):
        self.frames = frames
        self.calibration_frames = calibration_frames
        self.target_score = target_score

    def test_against_all_styles(self, genome_dict: dict, seed: int = 42) -> GeneralizationResult:
        """Testa um genoma (dict) contra todos os 7 estilos adversarios.

        `genome_dict` e um dicionario com os hiperparametros do genoma
        (plastic_lr, synthetic_w_init, etc.), compativel com o formato
        de Genome.to_dict().
        """
        rng = random.Random(seed)
        results: dict[str, StyleTestResult] = {}

        for style in list(OpponentStyle):
            score = self._test_against_style(genome_dict, style, rng.randint(0, 1_000_000))
            results[style.value] = score

        # Computa score de generalizacao
        fitness_values = [s.fitness for s in results.values()]
        if fitness_values:
            gen_score = sum(fitness_values) / len(fitness_values)
            std_dev = self._std(fitness_values)
            score_variance = self._variance(fitness_values)
        else:
            gen_score = 0.0
            std_dev = 0.0
            score_variance = 0.0

        best_style = max(results, key=lambda k: results[k].fitness) if results else ""
        worst_style = min(results, key=lambda k: results[k].fitness) if results else ""

        return GeneralizationResult(
            style_scores=results,
            generalization_score=gen_score,
            best_style=best_style,
            worst_style=worst_style,
            std_dev=std_dev,
            score_variance=score_variance,
        )

    def test_against_style(self, genome_dict: dict, style: OpponentStyle, seed: int = 42) -> StyleTestResult:
        """Testa um genoma contra um estilo especifico."""
        return self._test_against_style(genome_dict, style, seed)

    def _test_against_style(self, genome_dict: dict, style: OpponentStyle, seed: int) -> StyleTestResult:
        """Interno: testa contra um estilo especifico usando PongGame com opponent_strategy."""
        strategy = create_opponent_strategy(style)
        game = PongGame(seed=seed, opponent_strategy=strategy)

        # Simple policy brain: uses genome hyperparameters to compute action
        # This is a lightweight evaluation that doesn't require connectome data
        plastic_lr = genome_dict.get("plastic_lr", 0.02)
        synthetic_w_init = genome_dict.get("synthetic_w_init", 8.93)
        action_threshold = genome_dict.get("action_threshold", 0.5)
        noise_sigma = genome_dict.get("noise_sigma", 0.1)
        reward_sensitivity = genome_dict.get("reward_sensitivity", 1.0)

        # Simulate a simple policy brain that uses the genome's parameters
        # to determine paddle movement
        PADDLE_H = 60

        # Run the game
        point_outcomes = []
        bounce_rate_curve = []
        BOUNCE_RATE_WINDOW = 20
        rally_count = 0
        rally_best = 0
        prev_ball_y = game.ball_y

        for _ in range(self.frames):
            # Simple policy: move paddle toward the ball's Y position
            # The "brain" response quality depends on genome parameters
            ball_center = game.ball_y
            paddle_center = game.paddle_left_y + PADDLE_H / 2
            diff = ball_center - paddle_center

            # Action based on genome's action_threshold and noise
            if abs(diff) > action_threshold * 30:
                action = 1 if diff > 0 else -1
            else:
                action = 0

            # Add genome-dependent noise
            if noise_sigma > 0:
                if random.random() < noise_sigma * 0.1:
                    action = random.choice([-1, 0, 1])

            event = game.step(action)

            bounced = event["bounce"]
            missed = event["score"] == "right"

            if bounced:
                point_outcomes.append("bounce")
                rally_count += 1
                if rally_count > rally_best:
                    rally_best = rally_count
            else:
                rally_count = 0

            if missed:
                point_outcomes.append("miss")

            if bounced or missed:
                recent = point_outcomes[-BOUNCE_RATE_WINDOW:]
                bounce_rate_curve.append(recent.count("bounce") / len(recent))

        # Compute stats
        n_bounces = len([o for o in point_outcomes if o == "bounce"])
        n_misses = len([o for o in point_outcomes if o == "miss"])

        # Determine win/loss
        won = game.score_left > game.score_right

        # Fitness computation
        total_matches = 1  # single game
        performance = 1.0 if won else 0.0
        consistency = min(1.0, game.score_left / max(1, game.score_left + game.score_right + 1))
        robustness = min(1.0, n_bounces / max(1, rally_best * 2)) if rally_best > 0 else 0.0

        if len(bounce_rate_curve) >= 4:
            k = max(1, len(bounce_rate_curve) // 5)
            bounce_rate_inicio = sum(bounce_rate_curve[:k]) / k
            bounce_rate_fim = sum(bounce_rate_curve[-k:]) / k
            learning_gain = bounce_rate_fim - bounce_rate_inicio
        else:
            bounce_rate_inicio = 0.0
            bounce_rate_fim = 0.0
            learning_gain = 0.0

        learning = max(0.0, min(1.0, learning_gain / 2.0 + 0.5))
        generalization = min(1.0, n_bounces / max(1, total_matches * 3))

        # Fitness: weighted sum (same as fitness.py)
        WIN_WEIGHT = 10.0
        POINTS_WEIGHT = 1.0
        HITS_WEIGHT = 0.5
        MAX_RALLY_WEIGHT = 0.3
        LEARNING_GAIN_WEIGHT = 2.0

        fitness_value = (
            WIN_WEIGHT * performance
            + POINTS_WEIGHT * game.score_left
            + HITS_WEIGHT * n_bounces
            + MAX_RALLY_WEIGHT * rally_best
            + LEARNING_GAIN_WEIGHT * learning
        )

        profile = StyleTestResult(
            style=style.value,
            win_rate=performance,
            mean_bounce_rate=bounce_rate_inicio,
            mean_rally=float(rally_best),
            fitness=fitness_value,
            performance=performance,
            consistency=consistency,
            robustness=robustness,
            learning=learning,
            generalization=generalization,
            score_left=game.score_left,
            score_right=game.score_right,
            n_bounces=n_bounces,
            n_misses=n_misses,
            max_rally=rally_best,
        )

        return profile

    @staticmethod
    def _std(values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        m = sum(values) / len(values)
        return math.sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1))

    @staticmethod
    def _variance(values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        m = sum(values) / len(values)
        return sum((x - m) ** 2 for x in values) / (len(values) - 1)


def evaluate_generalization(genome_dict: dict, frames: int = 3000, seed: int = 42) -> GeneralizationResult:
    """Conveniencia: testa um genoma contra todos os estilos adversarios."""
    tester = OpponentTest(frames=frames)
    return tester.test_against_all_styles(genome_dict, seed=seed)


def compare_generalization_results(results_a: GeneralizationResult, results_b: GeneralizationResult) -> dict:
    """Compara dois resultados de generalizacao."""
    return {
        "gen_a_score": results_a.generalization_score,
        "gen_b_score": results_b.generalization_score,
        "better": "A" if results_a.generalization_score > results_b.generalization_score else "B",
        "a_vs_b_diff": results_a.generalization_score - results_b.generalization_score,
        "a_variance": results_a.score_variance,
        "b_variance": results_b.score_variance,
    }
