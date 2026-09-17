#!/usr/bin/env python3
"""Watch one fly-vs-fly duel in a pygame window.

Example:
    python evolution/run_duel_windowed.py --seed 1
    python evolution/run_duel_windowed.py --left-plastic-lr 0.02 --right-plastic-lr 0.0
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from game.pong import WIDTH, HEIGHT, PADDLE_H, PADDLE_W, BALL_SIZE
from evolution.genome import Genome
from evolution.duel_runner import make_duel, step_duel_frame


def _load_or_random(path: Path | None, rank: int, rng: random.Random) -> Genome:
    """path=None -> random genome. Otherwise load a JSON file that is either
    a single genome dict (champion.json) or a top-3-style list of
    {"genome": {...}, ...} entries (top3.json), in which case `rank` (1-based)
    picks which entry to use."""
    if path is None:
        return Genome.random(rng)
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        data = data[rank - 1]["genome"]
    return Genome.from_dict(data)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=1, help="Seed do jogo (fisica da bola).")
    ap.add_argument("--genome-seed", type=int, default=0, help="Usado so pro lado sem --*-genome.")
    ap.add_argument("--left-genome", type=Path, default=None,
                     help="JSON de genoma salvo (ex.: champions/champion.json) pro lado esquerdo.")
    ap.add_argument("--right-genome", type=Path, default=None,
                     help="JSON de genoma salvo pro lado direito.")
    ap.add_argument("--left-rank", type=int, default=1,
                     help="Se --left-genome for um top3.json, qual posicao (1-3) usar.")
    ap.add_argument("--right-rank", type=int, default=1,
                     help="Se --right-genome for um top3.json, qual posicao (1-3) usar.")
    ap.add_argument("--calibration-frames", type=int, default=200)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--left-plastic-lr", type=float, default=None)
    ap.add_argument("--right-plastic-lr", type=float, default=None)
    args = ap.parse_args()

    rng = random.Random(args.genome_seed)
    genome_left = _load_or_random(args.left_genome, args.left_rank, rng)
    genome_right = _load_or_random(args.right_genome, args.right_rank, rng)
    if args.left_plastic_lr is not None:
        genome_left = Genome(**{**genome_left.to_dict(), "plastic_lr": args.left_plastic_lr})
    if args.right_plastic_lr is not None:
        genome_right = Genome(**{**genome_right.to_dict(), "plastic_lr": args.right_plastic_lr})

    print(f"[duel] genoma esquerdo: {genome_left}")
    print(f"[duel] genoma direito:  {genome_right}")
    print("[duel] calibrando os dois cerebros (bola parada no centro)...")

    left, right, game = make_duel(genome_left, genome_right, seed=args.seed,
                                   calibration_frames=args.calibration_frames)

    import pygame
    pygame.init()
    pygame.display.set_caption("FlyPong duelo (fly vs fly)")
    screen = pygame.display.set_mode((WIDTH, HEIGHT + 30))
    font = pygame.font.SysFont(None, 22)
    clock = pygame.time.Clock()

    running = True
    while running:
        for evt in pygame.event.get():
            if evt.type == pygame.QUIT:
                running = False

        step_duel_frame(left, right, game)

        screen.fill((0, 0, 0))
        pygame.draw.rect(screen, (255, 255, 255), (0, game.paddle_left_y, PADDLE_W, PADDLE_H))
        pygame.draw.rect(
            screen, (255, 255, 255),
            (WIDTH - PADDLE_W, game.paddle_right_y, PADDLE_W, PADDLE_H),
        )
        pygame.draw.rect(screen, (255, 255, 255), (game.ball_x, game.ball_y, BALL_SIZE, BALL_SIZE))

        hud = (
            f"esquerda={game.score_left}  direita={game.score_right}   "
            f"lr_esq={genome_left.plastic_lr:.4f}  lr_dir={genome_right.plastic_lr:.4f}"
        )
        text = font.render(hud, True, (255, 255, 255))
        screen.blit(text, (8, HEIGHT + 6))

        pygame.display.flip()
        clock.tick(args.fps)

    pygame.quit()
    print(f"[duel] placar final: esquerda={game.score_left} direita={game.score_right}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
