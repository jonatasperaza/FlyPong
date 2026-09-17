#!/usr/bin/env python3
"""Compare original and fast screening paths on the same dataset."""
from __future__ import annotations

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from main import FlyPongRunner


def bench(data_dir: str, frames: int, fast: bool) -> tuple[float, float, int]:
    kwargs = dict(
        substeps=1 if fast else 4,
        prune_causal=fast,
        record_history=False,
        calibration_frames=25 if fast else 200,
    )
    runner = FlyPongRunner(data_dir, **kwargs)
    t0 = time.perf_counter()
    for _ in range(frames):
        runner.step_frame()
    elapsed = time.perf_counter() - t0
    return elapsed, frames / elapsed, runner.net.n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(HERE, "connectome_data"))
    ap.add_argument("--frames", type=int, default=1000)
    args = ap.parse_args()

    base_t, base_fps, base_n = bench(args.data_dir, args.frames, False)
    fast_t, fast_fps, fast_n = bench(args.data_dir, args.frames, True)

    print(f"baseline: {base_t:.3f}s | {base_fps:.1f} frames/s | n={base_n}")
    print(f"fast:     {fast_t:.3f}s | {fast_fps:.1f} frames/s | n={fast_n}")
    print(f"speedup:  {base_t / fast_t:.2f}x")
    print(f"node reduction: {base_n} -> {fast_n} ({1 - fast_n / base_n:.1%})")


if __name__ == "__main__":
    main()
