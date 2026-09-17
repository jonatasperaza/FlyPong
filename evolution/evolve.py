#!/usr/bin/env python3
"""Generational evolution over fly-vs-fly Swiss tournaments, with multiprocessing.

Only hyperparameters (Genome) are heritable; each duel builds fresh brains
(evolution.duel_runner.play_duel), so learned weights never carry across
generations -- see evolution/genome.py for why.

Windows multiprocessing note: this uses ProcessPoolExecutor with the
default "spawn" start method, which re-imports this module in each worker
process. The worker function (_duel_worker) is therefore a plain
module-level function taking only picklable primitives (genome dicts,
ints), never a live ConnectomeNetwork/Genome object -- and the CLI entry
point is guarded by `if __name__ == "__main__"`, both required for
multiprocessing to work at all on Windows.

Start small (per the handoff's own advice): --population 8 --rounds 3
--generations 5 --workers 1 first, to see the loop and the fitness curve
make sense, before scaling up population/generations/workers.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from evolution.genome import Genome
from evolution.tournament import Standing, pair_round
from evolution.fitness import fitness


class WatchWindows:
    """Optional live viewer: spawns N `run_duel_windowed.py` subprocesses so
    the user sees duels immediately (random genomes at the start) instead of
    waiting for the whole headless tournament to finish before watching
    anything. Re-spawned with the current all-time top-3 whenever the
    leaderboard changes.

    Each window is a real subprocess, not a thread, specifically so it gets
    its own SDL_VIDEODRIVER: evolve.py itself forces SDL_VIDEODRIVER=dummy
    at import time (the headless tournament must never try to open a
    window), so a watch window's environment has to explicitly clear that
    override or pygame would silently stay headless too.
    """

    def __init__(self, n_windows: int, frames_per_window_seed: int):
        self.n_windows = n_windows
        self._seed_counter = frames_per_window_seed
        self._procs: list[subprocess.Popen] = []
        self._tmpfiles: list[Path] = []
        self._last_genomes_key: str | None = None

    def _write_genome(self, genome: Genome) -> Path:
        fd, path = tempfile.mkstemp(suffix=".json", prefix="flypong_watch_")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(genome.to_dict(), f)
        p = Path(path)
        self._tmpfiles.append(p)
        return p

    def _spawn(self, pairs: list[tuple[Genome, Genome]]) -> None:
        self.stop()
        env = dict(os.environ)
        env.pop("SDL_VIDEODRIVER", None)  # let pygame pick a real display driver
        script = str(HERE / "evolution" / "run_duel_windowed.py")
        for left_genome, right_genome in pairs[: self.n_windows]:
            left_path = self._write_genome(left_genome)
            right_path = self._write_genome(right_genome)
            self._seed_counter += 1
            proc = subprocess.Popen(
                [
                    sys.executable, script,
                    "--left-genome", str(left_path),
                    "--right-genome", str(right_path),
                    "--seed", str(self._seed_counter),
                ],
                env=env,
            )
            self._procs.append(proc)

    def show_random(self, population: list[Genome], rng: random.Random) -> None:
        pool = population[:]
        rng.shuffle(pool)
        pairs = []
        for i in range(self.n_windows):
            a = pool[(2 * i) % len(pool)]
            b = pool[(2 * i + 1) % len(pool)]
            pairs.append((a, b))
        self._last_genomes_key = "random"
        self._spawn(pairs)

    def show_top(self, top_entries: list[dict]) -> None:
        if len(top_entries) < 2:
            return
        key = json.dumps([e["genome"] for e in top_entries])
        if key == self._last_genomes_key:
            return  # leaderboard didn't actually change; don't flicker the windows
        self._last_genomes_key = key
        genomes = [Genome.from_dict(e["genome"]) for e in top_entries]
        pairs = [(genomes[i % len(genomes)], genomes[(i + 1) % len(genomes)]) for i in range(self.n_windows)]
        self._spawn(pairs)

    def stop(self) -> None:
        for proc in self._procs:
            proc.terminate()
        self._procs.clear()
        for p in self._tmpfiles:
            p.unlink(missing_ok=True)
        self._tmpfiles.clear()


def _duel_worker(payload: tuple) -> dict:
    """Runs in a worker process: rebuild everything from primitives, return primitives."""
    from evolution.duel_runner import play_duel

    genome_a, genome_b, seed, frames, calibration_frames, target_score = payload
    ga = Genome.from_dict(genome_a)
    gb = Genome.from_dict(genome_b)
    result = play_duel(
        ga, gb, seed=seed, frames=frames, calibration_frames=calibration_frames,
        target_score=target_score,
    )
    return {
        "left": {
            "score": result.left.score,
            "n_bounces": result.left.n_bounces,
            "max_rally": result.left.max_rally,
            "learning_gain": result.left.learning_gain,
        },
        "right": {
            "score": result.right.score,
            "n_bounces": result.right.n_bounces,
            "max_rally": result.right.max_rally,
            "learning_gain": result.right.learning_gain,
        },
    }


def _breed(survivors: list[Genome], rng: random.Random, target_size: int, immigration_rate: float) -> list[Genome]:
    n_immigrants = max(1, round(target_size * immigration_rate))
    n_children = max(0, target_size - n_immigrants)
    children: list[Genome] = []
    while len(children) < n_children:
        a, b = (rng.sample(survivors, 2) if len(survivors) >= 2 else (survivors[0], survivors[0]))
        children.append(a.crossover(b, rng).mutate(rng))
    immigrants = [Genome.random(rng) for _ in range(n_immigrants)]
    return (children + immigrants)[:target_size]


def run_evolution(
    *,
    population_size: int,
    generations: int,
    rounds_per_gen: int,
    top_k: int,
    frames: int,
    calibration_frames: int,
    workers: int,
    seed: int,
    output_path: Path,
    champion_out: Path | None = None,
    top3_out: Path | None = None,
    hall_of_fame_size: int = 3,
    immigration_rate: float = 0.05,
    watch_windows: int = 0,
    target_score: int | None = None,
) -> tuple[Genome, list[dict]]:
    rng = random.Random(seed)
    next_id = 0

    def new_id() -> int:
        nonlocal next_id
        next_id += 1
        return next_id

    population: dict[int, Genome] = {new_id(): Genome.random(rng) for _ in range(population_size)}
    hall_of_fame: list[Genome] = []
    champion = next(iter(population.values()))
    all_time_leaderboard: list[dict] = []

    output_path.parent.mkdir(parents=True, exist_ok=True)
    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None

    watcher = WatchWindows(watch_windows, seed) if watch_windows > 0 else None
    if watcher is not None:
        # Show something immediately: 3 random pairings, before any duel has
        # even been scored, so the wait to "see the three windows" is however
        # long pygame takes to open, not however long generation 0 takes.
        watcher.show_random(list(population.values()), rng)

    try:
        with output_path.open("w", encoding="utf-8") as log:
            for gen in range(generations):
                # Hall-of-fame entries are extra opponents this generation, never bred:
                # they exist so the population cannot "forget" how to beat a past
                # champion by cycling past it (Rosin & Belew 1997).
                participants: dict[int, Genome] = dict(population)
                for hof_genome in hall_of_fame:
                    participants[new_id()] = hof_genome

                standings = {pid: Standing(entity_id=pid) for pid in participants}
                wins = {pid: 0 for pid in participants}
                matches = {pid: 0 for pid in participants}
                match_seed_base = rng.randrange(1_000_000)

                for round_idx in range(rounds_per_gen):
                    pairs = pair_round(list(standings.values()))
                    payloads = [
                        (
                            participants[a_id].to_dict(),
                            participants[b_id].to_dict(),
                            match_seed_base + round_idx * 100_003 + a_id * 97 + b_id,
                            frames,
                            calibration_frames,
                            target_score,
                        )
                        for a_id, b_id in pairs
                    ]

                    if executor is not None:
                        results = list(executor.map(_duel_worker, payloads))
                    else:
                        results = [_duel_worker(p) for p in payloads]

                    for (a_id, b_id), result in zip(pairs, results):
                        a_won = result["left"]["score"] > result["right"]["score"]
                        b_won = result["right"]["score"] > result["left"]["score"]
                        fa = fitness(
                            won=a_won,
                            points_scored=result["left"]["score"],
                            hits=result["left"]["n_bounces"],
                            max_rally=result["left"]["max_rally"],
                            learning_gain=result["left"]["learning_gain"],
                        )
                        fb = fitness(
                            won=b_won,
                            points_scored=result["right"]["score"],
                            hits=result["right"]["n_bounces"],
                            max_rally=result["right"]["max_rally"],
                            learning_gain=result["right"]["learning_gain"],
                        )
                        standings[a_id].score += fa
                        standings[b_id].score += fb
                        standings[a_id].opponents.add(b_id)
                        standings[b_id].opponents.add(a_id)
                        matches[a_id] += 1
                        matches[b_id] += 1
                        wins[a_id] += int(a_won)
                        wins[b_id] += int(b_won)

                # Rank only this generation's own population for selection/reproduction;
                # hall-of-fame ids exist to be beaten, not to be bred from again.
                pop_ranked = sorted(
                    (s for pid, s in standings.items() if pid in population),
                    key=lambda s: s.score,
                    reverse=True,
                )
                survivors = [population[s.entity_id] for s in pop_ranked[:top_k]]
                champion = population[pop_ranked[0].entity_id]
                champion_fitness = pop_ranked[0].score
                avg_fitness = sum(s.score for s in pop_ranked) / len(pop_ranked)

                # Feed the whole-tournament leaderboard: every individual this
                # generation is a candidate for "best of the entire run", not
                # just this generation's own champion (a strong genome from an
                # early generation can outscore every later champion).
                for s in pop_ranked:
                    win_rate = wins[s.entity_id] / matches[s.entity_id] if matches[s.entity_id] else 0.0
                    all_time_leaderboard.append({
                        "generation": gen,
                        "fitness": s.score,
                        "wins": wins[s.entity_id],
                        "matches": matches[s.entity_id],
                        "win_rate": win_rate,
                        "genome": population[s.entity_id].to_dict(),
                    })
                all_time_leaderboard.sort(key=lambda e: e["fitness"], reverse=True)
                del all_time_leaderboard[3:]

                if watcher is not None:
                    watcher.show_top(all_time_leaderboard)

                log.write(json.dumps({
                    "generation": gen,
                    "best_fitness": champion_fitness,
                    "avg_fitness": avg_fitness,
                    "champion_genome": champion.to_dict(),
                }) + "\n")
                log.flush()
                print(f"[evolve] gen={gen} best_fitness={champion_fitness:.2f} avg_fitness={avg_fitness:.2f}")

                # Written after every generation, not just at the end, so a
                # long run (hours) can be inspected or watched mid-flight
                # instead of only after it finishes or is killed.
                if champion_out is not None:
                    champion_out.parent.mkdir(parents=True, exist_ok=True)
                    champion_out.write_text(json.dumps(champion.to_dict(), indent=2), encoding="utf-8")
                if top3_out is not None:
                    top3_out.parent.mkdir(parents=True, exist_ok=True)
                    top3_out.write_text(json.dumps(all_time_leaderboard, indent=2), encoding="utf-8")

                if not hall_of_fame or champion.to_dict() != hall_of_fame[-1].to_dict():
                    hall_of_fame.append(champion)
                hall_of_fame = hall_of_fame[-hall_of_fame_size:]

                population = {new_id(): g for g in _breed(survivors, rng, population_size, immigration_rate)}
    finally:
        if executor is not None:
            executor.shutdown()

    return champion, all_time_leaderboard


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--population", type=int, default=8)
    ap.add_argument("--generations", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=3, help="Rodadas suicas por geracao.")
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument(
        "--frames", type=int, default=3000,
        help="Duelo de duracao fixa (padrao). Se --target-score for passado, "
             "isso vira o TETO de seguranca de frames em vez da duracao fixa.",
    )
    ap.add_argument(
        "--target-score", type=int, default=None, metavar="N",
        help="Se definido, o duelo termina assim que um lado marca N pontos "
             "(em vez de duracao fixa em frames). --frames continua valendo "
             "como teto de seguranca, pra genomas fracos nao travarem o duelo pra sempre.",
    )
    ap.add_argument("--calibration-frames", type=int, default=100)
    ap.add_argument("--workers", type=int, default=1, help="1 = sem paralelismo.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=Path, default=HERE / "runs" / "evolution.jsonl")
    ap.add_argument("--champion-out", type=Path, default=HERE / "champions" / "champion.json")
    ap.add_argument("--top3-out", type=Path, default=HERE / "champions" / "top3.json")
    ap.add_argument(
        "--watch", type=int, default=0, metavar="N",
        help="Abre N janelas de duelo ao vivo (3 pares aleatorios no inicio, "
             "depois troca pro top-3 sempre que o ranking muda). 0 = desligado (padrao).",
    )
    args = ap.parse_args()

    champion, top3 = run_evolution(
        population_size=args.population,
        generations=args.generations,
        rounds_per_gen=args.rounds,
        top_k=args.top_k,
        frames=args.frames,
        calibration_frames=args.calibration_frames,
        workers=args.workers,
        seed=args.seed,
        output_path=args.output,
        champion_out=args.champion_out,
        top3_out=args.top3_out,
        watch_windows=args.watch,
        target_score=args.target_score,
    )

    print(f"[evolve] campeao final salvo em {args.champion_out}")
    print(f"\n[evolve] TOP 3 de todo o torneio (por fitness, qualquer geracao):")
    for rank, entry in enumerate(top3, start=1):
        print(
            f"  #{rank} geracao={entry['generation']} fitness={entry['fitness']:.2f} "
            f"vitorias={entry['wins']}/{entry['matches']} (win_rate={entry['win_rate']:.2f})"
        )
    print(f"[evolve] top3 salvo em {args.top3_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
