"""Experiment runner — top-level orchestration for the Fly Evolution Engine.

This is the main entry point for running a full evolutionary experiment
as described in Etapa 1 of the roadmap. It ties together all the modular
components: Population, Evaluator, Selection, Reproduction, and HallOfFame.

Usage:
    experiment = Experiment(
        population_size=16,
        generations=10,
        rounds_per_gen=4,
        seed=0,
    )
    champion, history = experiment.run()

The experiment produces:
    - A champion Genome
    - A history of per-generation stats (best_fitness, mean_fitness, etc.)
    - Hall of Fame tracking
    - Optional output files
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Ensure repo root is on the path
HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from evolution.genome import Genome
from evolution.individual import Individual, FitnessProfile, MatchRecord
from evolution.population import Population
from evolution.evaluation import Evaluator
from evolution.hall_of_fame import HallOfFame
from evolution.selection import tournament_select
from evolution.reproduction import breed, create_random_population


class Experiment:
    """Run a full evolutionary experiment.

    Parameters map to the Etapa 1 roadmap requirements:
        population_size → number of individuals
        generations → number of evolutionary generations
        rounds_per_gen → Swiss tournament rounds per generation
        seed → reproducibility
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
        fitness_weights: dict[str, float] | None = None,
        selection_strategy: str = "tournament",
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
        self.fitness_weights = fitness_weights
        self.selection_strategy = selection_strategy

        self.rng = random.Random(seed)
        self.evaluator = Evaluator(
            frames=frames,
            calibration_frames=calibration_frames,
            target_score=target_score,
            workers=workers,
        )
        self.hall_of_fame = HallOfFame(max_size=hall_of_fame_size)
        self.history: list[dict] = []

    def run(self) -> tuple[Genome, list[dict]]:
        """Execute the full experiment. Returns (champion, history)."""
        rng = self.rng
        next_id = 0

        def new_id() -> int:
            nonlocal next_id
            next_id += 1
            return next_id

        population = create_random_population(self.population_size, rng)
        for ind in population:
            ind.generation = 0
            ind.id = new_id()

        champion = population[0]
        all_time_leaderboard: list[dict] = []

        self.output_dir.mkdir(parents=True, exist_ok=True)

        executor = ProcessPoolExecutor(max_workers=self.workers) if self.workers > 1 else None

        log_path = self.output_dir / "evolution.jsonl"
        with open(log_path, "w", encoding="utf-8") as log:
            for gen in range(self.generations):
                # --- Evaluate all individuals ---
                self._evaluate_population(population, gen, executor, rng)

                # --- Update Hall of Fame ---
                for ind in population:
                    if ind.compute_fitness() > 0:
                        self.hall_of_fame.add(ind)

                # --- Select champions and compute stats ---
                pop_ranked = sorted(population, key=lambda ind: ind.compute_fitness(), reverse=True)
                survivors = pop_ranked[: self.top_k]
                champion = pop_ranked[0]
                champion_fitness = champion.compute_fitness()
                avg_fitness = sum(ind.compute_fitness() for ind in pop_ranked) / len(pop_ranked)

                # --- Record history ---
                gen_record = {
                    "generation": gen,
                    "best_fitness": champion_fitness,
                    "avg_fitness": avg_fitness,
                    "champion_genome": champion.genome.to_dict(),
                    "champion_fitness_profile": champion.fitness.to_dict(),
                    "champion_record": {
                        "wins": champion.record.wins,
                        "losses": champion.record.losses,
                        "hits": champion.record.hits,
                        "max_rally": champion.record.max_rally,
                    },
                    "hall_of_fame_size": len(self.hall_of_fame.entries),
                }
                self.history.append(gen_record)

                log.write(json.dumps(gen_record) + "\n")
                log.flush()
                print(f"[experiment] gen={gen} best_fitness={champion_fitness:.2f} avg_fitness={avg_fitness:.2f}")

                # --- Save artifacts ---
                self._save_genome(champion, gen)
                self._save_leaderboard(all_time_leaderboard)

                # --- Hall of Fame opponents for next generation ---
                hof_opponents = self.hall_of_fame.get_opponents()
                for hof_ind in hof_opponents:
                    new_ind = Individual.from_genome(hof_ind.genome, generation=gen, id=new_id())
                    if new_ind not in population:
                        population.append(new_ind)

                # --- Breed next generation ---
                population = breed(
                    population, rng, self.population_size,
                    immigration_rate=self.immigration_rate,
                )
                for ind in population:
                    ind.generation = gen + 1
                    ind.id = new_id()

        if executor is not None:
            executor.shutdown()

        return champion, self.history

    def _evaluate_population(self, population: list[Individual], gen: int, executor, rng: random.Random) -> None:
        """Evaluate every individual in the population against a random opponent."""
        # Create random opponents for this generation
        opponents = [Individual.from_genome(Genome.random(self.rng)) for _ in population]

        if executor is not None:
            payloads = [
                (ind.genome.to_dict(), opp.genome.to_dict(), self.rng.randint(0, 1_000_000), self.frames, self.calibration_frames, self.target_score)
                for ind, opp in zip(population, opponents)
            ]
            results = list(executor.map(_experiment_worker, payloads))
        else:
            results = [
                _evaluate_pair(ind.genome, opp.genome, self.rng.randint(0, 1_000_000))
                for ind, opp in zip(population, opponents)
            ]

        for ind, result in zip(population, results):
            won = result["left_score"] > result["right_score"]
            ind.add_match_result(
                won=won,
                points_scored=result["left_score"] if won else result["right_score"],
                hits=result["left_stats"]["n_bounces"] if won else result["right_stats"]["n_bounces"],
                max_rally=result["left_stats"]["max_rally"] if won else result["right_stats"]["max_rally"],
                learning_gain=result["left_stats"]["learning_gain"],
            )
            ind.fitness = FitnessProfile(performance=ind._compute_performance())

    def _save_genome(self, champion: Individual, gen: int) -> None:
        """Save the current generation's champion genome."""
        champion_path = self.output_dir / "champions" / f"gen_{gen}.json"
        champion_path.parent.mkdir(parents=True, exist_ok=True)
        champion_path.write_text(json.dumps(champion.genome.to_dict(), indent=2), encoding="utf-8")

    def _save_leaderboard(self, leaderboard: list[dict]) -> None:
        """Save the all-time leaderboard."""
        top3_path = self.output_dir / "champions" / "top3.json"
        top3_path.parent.mkdir(parents=True, exist_ok=True)
        top3_path.write_text(json.dumps(leaderboard, indent=2), encoding="utf-8")


def _experiment_worker(payload: tuple) -> dict:
    """Worker for multiprocessing in Experiment._evaluate_population."""
    return _evaluate_pair(*payload)


def _evaluate_pair(genome_a_dict: dict, genome_b_dict: dict, seed: int, frames: int, calibration_frames: int, target_score: int | None) -> dict:
    """Evaluate a pair of genomes. Used as the worker function."""
    from evolution.duel_runner import play_duel
    from evolution.genome import Genome

    ga = Genome.from_dict(genome_a_dict)
    gb = Genome.from_dict(genome_b_dict)
    result = play_duel(ga, gb, seed=seed, frames=frames, calibration_frames=calibration_frames, target_score=target_score)
    return {
        "left_score": result.left.score,
        "right_score": result.right.score,
        "left_stats": {
            "n_bounces": result.left.n_bounces,
            "max_rally": result.left.max_rally,
            "learning_gain": result.left.learning_gain,
        },
        "right_stats": {
            "n_bounces": result.right.n_bounces,
            "max_rally": result.right.max_rally,
            "learning_gain": result.right.learning_gain,
        },
    }


def main() -> int:
    """CLI entry point for running experiments."""
    ap = argparse.ArgumentParser(description="Run a Fly Evolution Engine experiment (Etapa 1).")
    ap.add_argument("--population", type=int, default=16)
    ap.add_argument("--generations", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=4, help="Swiss rounds per generation.")
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--frames", type=int, default=3000)
    ap.add_argument("--calibration-frames", type=int, default=100)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", type=str, default=str(HERE / "runs"))
    ap.add_argument("--immigration-rate", type=float, default=0.05)
    ap.add_argument("--hall-of-fame-size", type=int, default=3)
    ap.add_argument("--target-score", type=int, default=None)
    ap.add_argument("--fitness-weights", type=str, default=None, help='JSON string of fitness weights, e.g. \'{"performance": 0.4, "consistency": 0.2}\'')
    ap.add_argument("--selection-strategy", type=str, default="tournament")
    args = ap.parse_args()

    fitness_weights = None
    if args.fitness_weights:
        fitness_weights = json.loads(args.fitness_weights)

    experiment = Experiment(
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
        fitness_weights=fitness_weights,
        selection_strategy=args.selection_strategy,
    )

    champion, history = experiment.run()

    print(f"\n[experiment] Champion saved to {experiment.output_dir / 'champions' / 'champion.json'}")
    print(f"\n[experiment] Generation history:")
    for record in history:
        print(
            f"  gen={record['generation']} best={record['best_fitness']:.2f} "
            f"avg={record['avg_fitness']:.2f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
