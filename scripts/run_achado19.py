#!/usr/bin/env python3
"""Achado 19: aprendizado por recompensa avaliado por saques independentes.

Protocolo definido antes de rodar (ver README, Achado 19):
  - seeds de treino `serve_eval.TRAIN_SEEDS`, seeds de avaliacao
    `serve_eval.EVAL_SEEDS` (pareadas por posicao); hiperparametros escolhidos
    so nas `serve_train.VALIDATION_SEEDS`;
  - politicas: trained, control_homeostasis, control_innate;
  - condicoes: inverted (teste decisivo) e normal;
  - criterio de sucesso: na condicao invertida, IC95 da rede treinada acima
    do IC95 de cada controle (sem sobreposicao) em pelo menos 5 de 6 seeds.

Anexa uma linha JSON por (condicao, politica, seed) em --out e recusa
sobrescrever resultados ja existentes.
"""

from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import main
import serve_eval as se
import serve_train as st

CONDITIONS = ("inverted", "normal")


def _job(args):
    data_dir, condition, policy, train_seed, eval_seed, trials, curve_every = args
    row = st.run_condition(
        data_dir, condition=condition, policy=policy, train_seed=train_seed,
        eval_seed=eval_seed, eval_trials=trials,
        curve_every=curve_every if policy == "trained" else 0, verbose=False,
    )
    with open(Path(data_dir) / "metadata.json", encoding="utf-8") as f:
        row["data_source"] = json.load(f).get("source", "desconhecido")
    return row


def load(path: Path) -> dict:
    rows = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                rows[(r["condition"], r["policy"], r["eval_seed"])] = r
    return rows


def summarize(rows: dict) -> list[str]:
    out = []
    for cond in CONDITIONS:
        wins = 0
        n = 0
        out.append(f"== {cond} ==")
        for tr_seed, ev_seed in zip(se.TRAIN_SEEDS, se.EVAL_SEEDS):
            got = {p: rows.get((cond, p, ev_seed)) for p in st.POLICIES}
            if any(v is None for v in got.values()):
                continue
            n += 1
            t = got["trained"]
            beat = all(t["ci95_low"] > got[c]["ci95_high"]
                       for c in ("control_homeostasis", "control_innate"))
            wins += beat
            out.append(
                f"seed {tr_seed:>2}/{ev_seed}: "
                + "  ".join(f"{p}={got[p]['hit_rate']:.3f} "
                            f"[{got[p]['ci95_low']:.3f},{got[p]['ci95_high']:.3f}]"
                            for p in st.POLICIES)
                + ("  OK" if beat else "  --")
            )
        out.append(f"{cond}: treinada acima dos dois controles (IC sem sobreposicao) "
                   f"em {wins}/{n} seeds")
    return out


def main_cli():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=main.DATA_DIR)
    ap.add_argument("--out", default=str(HERE / "docs" / "achado-19-resultados.jsonl"))
    ap.add_argument("--trials", type=int, default=se.DEFAULT_TRIALS)
    ap.add_argument("--curve-every", type=int, default=500)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    rows = load(out)
    if not args.summary_only:
        jobs = [
            (args.data_dir, cond, pol, tr, ev, args.trials, args.curve_every)
            for cond in CONDITIONS
            for pol in ("trained",) + st.POLICIES[1:]
            for tr, ev in zip(se.TRAIN_SEEDS, se.EVAL_SEEDS)
            if (cond, pol, ev) not in rows
        ]
        print(f"[achado19] {len(jobs)} execucoes pendentes", flush=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        with Pool(args.workers) as pool:
            for row in pool.imap_unordered(_job, jobs):
                with open(out, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows[(row["condition"], row["policy"], row["eval_seed"])] = row
                print(f"[achado19] {row['condition']}/{row['policy']} "
                      f"seed={row['eval_seed']} acerto={row['hit_rate']:.3f} "
                      f"({row['elapsed_seconds']:.0f}s)", flush=True)
    print("\n".join(summarize(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
