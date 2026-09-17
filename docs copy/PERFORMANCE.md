# FlyPong performance patch

This patch adds a reversible performance mode without changing the default
scientific path.

## What changed

- `ConnectomeNetwork(..., prune_causal=True)` removes nodes that cannot lie on a
  causal path from externally stimulated sensory/reward sources to the
  descending motor readout. Photoreceptor, dopaminergic and descending pools
  are retained explicitly so the current model interface remains intact.
- Hot arrays in the network and LIF simulator are reused instead of allocating
  temporary vectors every step.
- `record_history=False` removes per-frame plastic-weight history in screening.
- A new `--fast` mode in `main.py` is intended for hypothesis screening:
  causal pruning, 1 substep/frame, reduced calibration, no per-frame history.

## Scientific meaning

`--fast` is **not** an interchangeable replacement for the original protocol.
The 1-substep setting changes temporal resolution. Use it to reject bad ideas
quickly, then rerun surviving configurations using the original settings.

Causal pruning itself is intended to preserve the observable path for the
current input/output interface, but it should be regression-tested against the
full graph before using it for published numbers.

## Suggested workflow

1. Fast screening: `python main.py --headless --fast --frames 3000 ...`
2. Candidate confirmation: original 4 substeps and full history.
3. Final validation: multiple seeds and the existing statistical protocol.

## Benchmark

Use:

```bash
python scripts/benchmark_fast.py --data-dir connectome_data --frames 1000
```

The benchmark reports baseline vs fast frame/s and the relative speedup. The
numbers are machine-dependent; no fixed 1000x claim is made without a local
benchmark.
