#!/usr/bin/env python3
"""CLI to run one fly-vs-fly duel and print the result.

Example:
    python evolution/run_duel.py --seed 1 --frames 15000
    python evolution/run_duel.py --left-plastic-lr 0.02 --right-plastic-lr 0.0
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from evolution.genome import Genome
from evolution.duel_runner import play_duel


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=1, help="Seed do jogo (fisica da bola).")
    ap.add_argument("--genome-seed", type=int, default=0, help="Seed pros genomas aleatorios.")
    ap.add_argument("--frames", type=int, default=15000)
    ap.add_argument("--calibration-frames", type=int, default=200)
    ap.add_argument("--left-plastic-lr", type=float, default=None,
                     help="Sobrescreve plastic_lr do genoma esquerdo (ex.: 0.0 pra congelar).")
    ap.add_argument("--right-plastic-lr", type=float, default=None,
                     help="Sobrescreve plastic_lr do genoma direito.")
    args = ap.parse_args()

    rng = random.Random(args.genome_seed)
    left = Genome.random(rng)
    right = Genome.random(rng)
    if args.left_plastic_lr is not None:
        left = Genome(**{**left.to_dict(), "plastic_lr": args.left_plastic_lr})
    if args.right_plastic_lr is not None:
        right = Genome(**{**right.to_dict(), "plastic_lr": args.right_plastic_lr})

    print(f"[duel] genoma esquerdo: {left}")
    print(f"[duel] genoma direito:  {right}")
    print(f"[duel] rodando {args.frames} frames (seed={args.seed})...")

    result = play_duel(left, right, seed=args.seed, frames=args.frames,
                        calibration_frames=args.calibration_frames)

    print(f"[duel] placar: esquerda={result.left.score} direita={result.right.score}")
    print(f"[duel] esquerda: bounces={result.left.n_bounces} max_rally={result.left.max_rally} "
          f"bounce_rate_fim={result.left.bounce_rate_fim} learning_gain={result.left.learning_gain}")
    print(f"[duel] direita:  bounces={result.right.n_bounces} max_rally={result.right.max_rally} "
          f"bounce_rate_fim={result.right.bounce_rate_fim} learning_gain={result.right.learning_gain}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
