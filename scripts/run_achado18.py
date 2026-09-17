#!/usr/bin/env python3
"""Run or resume Achado 18 with an explicit, append-only protocol.

The initial 23 results were created interactively. This script captures the
reconstructed recipe and refuses to overwrite an existing (mode, seed) result
unless --force is passed. Its default protocol intentionally uses the full
scientific path: four substeps, history enabled, no causal pruning.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import main
import sim.network as netmod


SEEDS = (42, 1, 2, 3, 4, 5)
MODES = ("control", "original", "freq_normalized", "tonic_baseline")


def load_existing(path: Path) -> dict[tuple[str, int], dict]:
    if not path.exists():
        return {}
    rows: dict[tuple[str, int], dict] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        key = (row["mode"], int(row["seed"]))
        if key in rows:
            raise ValueError(f"duplicate result for {key} on line {line_number}")
        rows[key] = row
    return rows


def run_condition(*, mode: str, seed: int, frames: int, weight: float, plastic_lr: float) -> dict:
    """Run exactly one paired condition from the Achado 18 protocol."""
    main.SYNTHETIC_W_INIT = weight
    netmod.PLASTIC_LR = 0.0 if mode == "control" else plastic_lr
    netmod.PLASTICITY_MODE = "original" if mode == "control" else mode

    runner = main.FlyPongRunner(
        main.DATA_DIR,
        signal_mode="synthetic_plastic",
        game_seed=seed,
        # Do not use --fast semantics for a scientific result.
        substeps=main.SUBSTEPS_PER_FRAME,
        prune_causal=False,
        record_history=True,
        calibration_frames=main.CALIBRATION_FRAMES,
    )
    weight_start = runner.synthetic_weight
    started = time.perf_counter()
    for _ in range(frames):
        runner.step_frame()
    elapsed = time.perf_counter() - started
    report = runner.validation_report()
    if "bounce_rate_fim" not in report:
        raise RuntimeError("run ended without enough attempts for a validation report")

    row = {
        "mode": mode,
        "seed": seed,
        "n_attempts": report["n_attempts"],
        "delta": report["delta"],
        "bounce_rate_inicio": report["bounce_rate_inicio"],
        "bounce_rate_fim": report["bounce_rate_fim"],
        "weight_start": weight_start,
        "weight_end": runner.synthetic_weight,
        "protocol": {
            "frames": frames,
            "synthetic_weight_init": weight,
            "plastic_lr": netmod.PLASTIC_LR,
            "plasticity_mode": netmod.PLASTICITY_MODE,
            "substeps": runner.substeps,
            "prune_causal": False,
            "record_history": True,
            "calibration_frames": main.CALIBRATION_FRAMES,
        },
        "elapsed_seconds": elapsed,
    }
    return row


def main_cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "runs" / "achado18_fase1.jsonl")
    parser.add_argument("--frames", type=int, default=15_000)
    parser.add_argument("--weight", type=float, default=40.0)
    parser.add_argument("--plastic-lr", type=float, default=0.02)
    parser.add_argument("--mode", choices=MODES, action="append")
    parser.add_argument("--seed", type=int, choices=SEEDS, action="append")
    parser.add_argument("--force", action="store_true", help="rerun and append an already present condition")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    modes = tuple(args.mode) if args.mode else MODES
    seeds = tuple(args.seed) if args.seed else SEEDS
    existing = load_existing(args.output)
    pending = [(mode, seed) for mode in modes for seed in seeds if args.force or (mode, seed) not in existing]

    print(f"[achado18] output={args.output}")
    print(f"[achado18] pending={pending}")
    if args.dry_run:
        return 0
    if not pending:
        print("[achado18] nothing to run")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    for mode, seed in pending:
        print(f"[achado18] starting mode={mode} seed={seed}", flush=True)
        row = run_condition(
            mode=mode,
            seed=seed,
            frames=args.frames,
            weight=args.weight,
            plastic_lr=args.plastic_lr,
        )
        with args.output.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[achado18] finished {json.dumps(row, ensure_ascii=False)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
