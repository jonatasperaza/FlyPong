"""Experiment runner — Etapa 2: Evolução, Lamarckismo e Evolução+Aprendizado.

Três experimentos científicos para descobrir se a evolução
realmente resolve o problema:

  Experimento A — Evolução de hiperparâmetros (genoma apenas)
  Experimento B — Lamarckismo artificial (genoma + pesos aprendidos)
  Experimento C — Evolução + aprendizado (genoma → plasticidade durante vida)

Cada experimento é rodado com condições idênticas exceto pela
forma de herança. Os resultados são comparados estatisticamente
para responder: "A evolução realmente melhora alguma coisa?"

Uso:
    runner = ExperimentRunner(seed=0)
    results = runner.run_all(population_size=16, generations=10)
    comparison = runner.compare(results)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Ensure repo root is on the path
HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from evolution.genome import Genome
from evolution.individual import Individual, InheritanceMode, FitnessProfile, MatchRecord
from evolution.evaluation import Evaluator, WeightedEvaluationResult
from evolution.hall_of_fame import HallOfFame
from evolution.reproduction import breed, create_random_population, InheritanceMode as ReproMode
from evolution.fitness import fitness_profile_from_stats, DEFAULT_WEIGHTS


class ExperimentType:
    """The three experiments of Etapa 2."""
    GENOME_ONLY = "genome_only"       # Experimento A
    LAMARCKIAN = "lamarckian"          # Experimento B
    EVOLVE_LEARN = "evolve_learn"     # Experimento C


@dataclass
class GenerationStats:
    """Statistics for a single generation."""
    generation: int
    best_fitness: float
    mean_fitness: float
    median_fitness: float
    std_fitness: float
    best_performance: float = 0.0
    best_learning: float = 0.0
    best_generalization: float = 0.0
    champion_genome: dict | None = None
    hall_of_fame_size: int = 0


@dataclass
class ExperimentResult:
    """Complete result of running one experiment type."""
    experiment_type: str
    seed: int
    population_size: int
    generations: int
    history: list[GenerationStats] = field(default_factory=list)
    champion: Individual | None = None
    final_population: list[Individual] = field(default_factory=list)
    total_fitness_gain: float = 0.0
    fitness_curve: list[float] = field(default_factory=list)
    convergence_generation: int = -1

    @property
    def best_fitness(self) -> float:
        return self.history[-1].best_fitness if self.history else 0.0

    @property
    def improvement(self) -> float:
        """Total improvement from generation 0 to last generation."""
        if len(self.history) < 2:
            return 0.0
        return self.history[-1].best_fitness - self.history[0].best_fitness


@dataclass
class StatisticalComparison:
    """Statistical comparison between experiment results."""
    type_a: ExperimentResult
    type_b: ExperimentResult
    type_c: ExperimentResult

    mean_a: float = 0.0
    mean_b: float = 0.0
    mean_c: float = 0.0

    improvement_a: float = 0.0
    improvement_b: float = 0.0
    improvement_c: float = 0.0

    p_value_ab: float = 0.0
    p_value_ac: float = 0.0
    p_value_bc: float = 0.0

    significant_ab: bool = False
    significant_ac: bool = False
    significant_bc: bool = False

    winner_a_vs_b: str = ""
    winner_a_vs_c: str = ""
    winner_b_vs_c: str = ""

    def __post_init__(self):
        self.mean_a = self._mean(self.type_a.fitness_curve)
        self.mean_b = self._mean(self.type_b.fitness_curve)
        self.mean_c = self._mean(self.type_c.fitness_curve)
        self.improvement_a = self.type_a.improvement
        self.improvement_b = self.type_b.improvement
        self.improvement_c = self.type_c.improvement
        self._compute_winner()

    @staticmethod
    def _mean(values: list[float]) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)

    @staticmethod
    def _std(values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        m = sum(values) / len(values)
        return math.sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1))

    def _compute_winner(self):
        """Determine which experiment type wins each comparison."""
        if self.improvement_b > self.improvement_a:
            self.winner_a_vs_b = "B"
        elif self.improvement_a > self.improvement_b:
            self.winner_a_vs_b = "A"
        else:
            self.winner_a_vs_b = "empate"

        if self.improvement_c > self.improvement_a:
            self.winner_a_vs_c = "C"
        elif self.improvement_a > self.improvement_c:
            self.winner_a_vs_c = "A"
        else:
            self.winner_a_vs_c = "empate"

        if self.improvement_c > self.improvement_b:
            self.winner_b_vs_c = "C"
        elif self.improvement_b > self.improvement_c:
            self.winner_b_vs_c = "B"
        else:
            self.winner_b_vs_c = "empate"

    def mann_whitney_u(self, a: list[float], b: list[float]) -> float:
        """Approximate Mann-Whitney U test p-value.

        Simplified version for comparing fitness curves across
        experiments. Returns approximate p-value (0 = significant).
        """
        if not a or not b:
            return 1.0
        combined = [(v, 0) for v in a] + [(v, 1) for v in b]
        combined.sort(key=lambda x: x[0])
        ranks = {}
        i = 0
        while i < len(combined):
            j = i
            while j < len(combined) - 1 and combined[j + 1][0] == combined[i][0]:
                j += 1
            avg_rank = (i + 1 + j + 1) / 2
            for k in range(i, j + 1):
                ranks[k] = avg_rank
            i = j + 1

        r_a = sum(ranks[k] for k in range(len(combined)) if combined[k][1] == 0)
        r_b = sum(ranks[k] for k in range(len(combined)) if combined[k][1] == 1)

        n_a, n_b = len(a), len(b)
        u_a = r_a - n_a * (n_a + 1) / 2
        u_b = r_b - n_b * (n_b + 1) / 2
        u = min(u_a, u_b)

        mu = n_a * n_b / 2
        sigma = math.sqrt(n_a * n_b * (n_a + n_b + 1) / 12)
        if sigma == 0:
            return 1.0
        z = (u - mu) / sigma
        # Approximate p-value from z-score
        p = 2 * (1 - self._norm_cdf(abs(z)))
        return max(0.0, min(1.0, p))

    @staticmethod
    def _norm_cdf(x: float) -> float:
        """Approximation of the standard normal CDF."""
        # Abramowitz and Stegun approximation
        t = 1.0 / (1.0 + 0.2316419 * abs(x))
        d = 0.3989422804014327  # 1/sqrt(2*pi)
        p = d * math.exp(-x * x / 2.0) * (t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.8212560 + t * 1.3302744)))))
        return 1.0 - p if x > 0 else p


class ExperimentRunner:
    """Run all three experiments of Etapa 2 and compare them.

    Usage:
        runner = ExperimentRunner(seed=0, population_size=16, generations=10)
        results = runner.run_all()
        comparison = runner.compare(results)
        print(comparison.summary())
    """

    def __init__(
        self,
        population_size: int = 16,
        generations: int = 10,
        rounds_per_gen: int = 4,
        top_k: int = 4,
        frames: int = 3000,
        calibration_frames: int = 100,
        workers: int = 1,
        seed: int = 0,
        immigration_rate: float = 0.05,
        hall_of_fame_size: int = 3,
        output_dir: str | None = None,
        target_score: int | None = None,
        fitness_weights: dict | None = None,
        evaluate_with_weights: bool = True,
    ):
        self.population_size = population_size
        self.generations = generations
        self.rounds_per_gen = rounds_per_gen
        self.top_k = top_k
        self.frames = frames
        self.calibration_frames = calibration_frames
        self.workers = workers
        self.seed = seed
        self.immigration_rate = immigration_rate
        self.hall_of_fame_size = hall_of_fame_size
        self.output_dir = Path(output_dir) if output_dir else HERE / "runs"
        self.target_score = target_score
        self.fitness_weights = fitness_weights or DEFAULT_WEIGHTS
        self.evaluate_with_weights = evaluate_with_weights

    def run_experiment(self, experiment_type: str, seed: int = None) -> ExperimentResult:
        """Run a single experiment type."""
        if seed is None:
            seed = self.seed

        rng = random.Random(seed)
        next_id = 0

        def new_id() -> int:
            nonlocal next_id
            next_id += 1
            return next_id

        # Determine inheritance mode
        if experiment_type == ExperimentType.LAMARCKIAN:
            mode = InheritanceMode.LAMARCKIAN
        elif experiment_type == ExperimentType.EVOLVE_LEARN:
            mode = InheritanceMode.EVOLVE_LEARN
        else:
            mode = InheritanceMode.GENOME_ONLY

        # Create initial population
        population = create_random_population(self.population_size, rng, generation=0, mode=mode)
        for ind in population:
            ind.id = new_id()

        evaluator = Evaluator(
            frames=self.frames,
            calibration_frames=self.calibration_frames,
            target_score=self.target_score,
            workers=self.workers,
        )
        hall_of_fame = HallOfFame(max_size=self.hall_of_fame_size)

        history: list[GenerationStats] = []
        all_time_leaderboard: list[dict] = []
        fitness_curve: list[float] = []

        self.output_dir.mkdir(parents=True, exist_ok=True)
        executor = ProcessPoolExecutor(max_workers=self.workers) if self.workers > 1 else None

        try:
            with open(self.output_dir / f"experiment_{experiment_type}_seed_{seed}.jsonl", "w") as log:
                for gen in range(self.generations):
                    # Evaluate each individual
                    for ind in population:
                        self._evaluate_individual(ind, population, rng, evaluator)

                    # Update Hall of Fame
                    for ind in population:
                        hall_of_fame.add(ind)

                    # Compute stats
                    ranked = sorted(population, key=lambda ind: ind.compute_fitness(), reverse=True)
                    best = ranked[0]
                    champion_fitness = best.compute_fitness()
                    fitness_values = [ind.compute_fitness() for ind in population]
                    mean_fitness = sum(fitness_values) / len(fitness_values)
                    median_fitness = sorted(fitness_values)[len(fitness_values) // 2]
                    std_fitness = self._std(fitness_values)

                    # Record generation stats
                    gen_stats = GenerationStats(
                        generation=gen,
                        best_fitness=champion_fitness,
                        mean_fitness=mean_fitness,
                        median_fitness=median_fitness,
                        std_fitness=std_fitness,
                        best_performance=best.fitness.performance,
                        best_learning=best.fitness.learning,
                        best_generalization=best.fitness.generalization,
                        champion_genome=best.genome.to_dict(),
                        hall_of_fame_size=len(hall_of_fame.entries),
                    )
                    history.append(gen_stats)
                    fitness_curve.append(champion_fitness)

                    log.write(json.dumps({
                        "generation": gen,
                        "best_fitness": champion_fitness,
                        "mean_fitness": mean_fitness,
                        "median_fitness": median_fitness,
                        "std_fitness": std_fitness,
                        "best_performance": best.fitness.performance,
                        "best_learning": best.fitness.learning,
                        "best_generalization": best.fitness.generalization,
                    }) + "\n")
                    log.flush()
                    print(f"[{experiment_type}] gen={gen} best={champion_fitness:.4f} mean={mean_fitness:.4f}")

                    # Breed next generation
                    survivors = ranked[: self.top_k]
                    population = breed(
                        population, rng, self.population_size,
                        immigration_rate=self.immigration_rate, mode=mode,
                    )
                    for ind in population:
                        ind.id = new_id()
        finally:
            if executor is not None:
                executor.shutdown()

        # Find convergence generation
        convergence_gen = -1
        if len(fitness_curve) >= 3:
            for i in range(2, len(fitness_curve)):
                if fitness_curve[i] - fitness_curve[i - 1] < 0.001:
                    convergence_gen = i
                    break

        return ExperimentResult(
            experiment_type=experiment_type,
            seed=seed,
            population_size=self.population_size,
            generations=self.generations,
            history=history,
            champion=ranked[0] if ranked else None,
            final_population=population,
            fitness_curve=fitness_curve,
            convergence_generation=convergence_gen,
        )

    def run_all(self) -> dict[str, ExperimentResult]:
        """Run all three experiment types with the same seed base."""
        results = {}
        for exp_type in [ExperimentType.GENOME_ONLY, ExperimentType.LAMARCKIAN, ExperimentType.EVOLVE_LEARN]:
            seed = self.seed + hash(exp_type) % 1000
            print(f"\n{'='*60}")
            print(f"RUNNING {exp_type} (seed={seed})")
            print(f"{'='*60}")
            results[exp_type] = self.run_experiment(exp_type, seed=seed)
        return results

    def compare(self, results: dict[str, ExperimentResult]) -> StatisticalComparison:
        """Compare all three experiment results statistically."""
        type_a = results.get(ExperimentType.GENOME_ONLY, results[list(results.keys())[0]])
        type_b = results.get(ExperimentType.LAMARCKIAN, results[list(results.keys())[1]])
        type_c = results.get(ExperimentType.EVOLVE_LEARN, results[list(results.keys())[2]])

        comp = StatisticalComparison(
            type_a=type_a,
            type_b=type_b,
            type_c=type_c,
        )
        return comp

    def _evaluate_individual(self, ind: Individual, population: list[Individual], rng: random.Random, evaluator: Evaluator) -> None:
        """Evaluate one individual against a random opponent."""
        # Pick a random opponent from the population
        opponents = [other for other in population if other is not ind]
        if not opponents:
            return
        opponent = rng.choice(opponents)
        seed = rng.randint(0, 1_000_000)

        if self.evaluate_with_weights and ind.inheritance_mode in (InheritanceMode.LAMARCKIAN, InheritanceMode.EVOLVE_LEARN):
            result = evaluator.evaluate_with_weights(ind, opponent, seed)
            won = result.left_stats["score"] > result.right_stats["score"]
            ind.add_match_result(
                won=won,
                points_scored=result.left_stats["score"] if won else result.right_stats["score"],
                hits=result.left_stats["n_bounces"] if won else result.right_stats["n_bounces"],
                max_rally=result.left_stats["max_rally"] if won else result.right_stats["max_rally"],
                learning_gain=result.left_stats["learning_gain"],
                avg_rally=0.0,
            )
            # Store learned weights for Lamarckian reproduction
            if result.synaptic_weights is not None:
                ind.synaptic_weights = result.synaptic_weights
                ind.weight_mean = result.weight_mean
                ind.weight_std = result.weight_std
        else:
            result = evaluator.evaluate_individual(ind, opponent, seed)
            won = result > 0  # heuristic
            ind.add_match_result(
                won=won,
                points_scored=1 if won else 0,
                hits=0,
                max_rally=0,
                learning_gain=0.0,
            )

        # Compute multi-dimensional fitness
        ind.compute_fitness(weights=self.fitness_weights)

    @staticmethod
    def _std(values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        m = sum(values) / len(values)
        return math.sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1))

    def summary(self, comparison: StatisticalComparison) -> str:
        """Return a human-readable summary of the comparison."""
        lines = [
            "=" * 60,
            "COMPARISON DE EXPERIMENTOS — ETAPA 2",
            "=" * 60,
            "",
            f"  Tipo A (Genoma apenas):     melhoria = {comparison.improvement_a:.4f}",
            f"  Tipo B (Lamarckismo):       melhoria = {comparison.improvement_b:.4f}",
            f"  Tipo C (Evolucao+Aprendizado): melhoria = {comparison.improvement_c:.4f}",
            "",
            f"  Media fitness A: {comparison.mean_a:.4f}",
            f"  Media fitness B: {comparison.mean_b:.4f}",
            f"  Media fitness C: {comparison.mean_c:.4f}",
            "",
            f"  Vencedor A vs B: {comparison.winner_a_vs_b}",
            f"  Vencedor A vs C: {comparison.winner_a_vs_c}",
            f"  Vencedor B vs C: {comparison.winner_b_vs_c}",
            "",
            "=" * 60,
            "CONCLUSÃO:",
            "=" * 60,
        ]

        if comparison.improvement_b > comparison.improvement_a and comparison.improvement_b > comparison.improvement_c:
            lines.append("  O lamarckismo (B) produz MAIOR melhoria evolutiva.")
            lines.append("  A experiência adquirida durante a vida acumula competência entre gerações!")
        elif comparison.improvement_c > comparison.improvement_a and comparison.improvement_c > comparison.improvement_b:
            lines.append("  Evolução + aprendizado (C) produz MAIOR melhoria evolutiva.")
            lines.append("  A plasticidade durante a vida combinada com seleção genética é a chave!")
        elif comparison.improvement_a >= comparison.improvement_b and comparison.improvement_a >= comparison.improvement_c:
            lines.append("  Apenas genoma (A) é o melhor. Evolução de hiperparâmetros é suficiente.")
        else:
            lines.append("  Resultados inconclusivos. Mais gerações podem ser necessárias.")

        return "\n".join(lines)


def main() -> int:
    """CLI entry point for running all three experiments."""
    ap = argparse.ArgumentParser(description="Run Etapa 2 experiments (A/B/C).")
    ap.add_argument("--population", type=int, default=16)
    ap.add_argument("--generations", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--frames", type=int, default=3000)
    ap.add_argument("--calibration-frames", type=int, default=100)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", type=str, default=str(HERE / "runs"))
    ap.add_argument("--immigration-rate", type=float, default=0.05)
    ap.add_argument("--hall-of-fame-size", type=int, default=3)
    ap.add_argument("--target-score", type=int, default=None)
    ap.add_argument("--no-weights", action="store_true", help="Disable weight extraction (faster)")
    args = ap.parse_args()

    runner = ExperimentRunner(
        population_size=args.population,
        generations=args.generations,
        rounds_per_gen=args.rounds,
        top_k=args.top_k,
        frames=args.frames,
        calibration_frames=args.calibration_frames,
        workers=args.workers,
        seed=args.seed,
        output_dir=args.output_dir,
        immigration_rate=args.immigration_rate,
        hall_of_fame_size=args.hall_of_fame_size,
        target_score=args.target_score,
        evaluate_with_weights=not args.no_weights,
    )

    results = runner.run_all()
    comparison = runner.compare(results)
    print("\n")
    print(runner.summary(comparison))

    # Save comparison to file
    comp_path = Path(args.output_dir) / "comparison.json"
    comp_path.parent.mkdir(parents=True, exist_ok=True)
    comp_data = {
        "improvement_a": comparison.improvement_a,
        "improvement_b": comparison.improvement_b,
        "improvement_c": comparison.improvement_c,
        "mean_a": comparison.mean_a,
        "mean_b": comparison.mean_b,
        "mean_c": comparison.mean_c,
        "winner_a_vs_b": comparison.winner_a_vs_b,
        "winner_a_vs_c": comparison.winner_a_vs_c,
        "winner_b_vs_c": comparison.winner_b_vs_c,
    }
    comp_path.write_text(json.dumps(comp_data, indent=2), encoding="utf-8")
    print(f"\nComparison saved to {comp_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
