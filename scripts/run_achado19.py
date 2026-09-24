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
sobrescrever resultados ja existentes. Por padrao o arquivo de saida depende
da origem dos dados (`metadata.json` -> "source"):
docs/achado-19-resultados-<source>[-elevation].jsonl. O script recusa
misturar no mesmo arquivo resultados de origens ou eixos de retina
diferentes.
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
    data_dir, condition, policy, train_seed, eval_seed, trials, curve_every, axis, scope, serves = args
    row = st.run_condition(
        data_dir, condition=condition, policy=policy, train_seed=train_seed,
        eval_seed=eval_seed, eval_trials=trials,
        curve_every=curve_every if policy == "trained" else 0, verbose=False,
        retina_axis=axis, homeostasis_scope=scope, n_serves=serves,
    )
    row["data_source"] = data_source(data_dir)
    row["dataset"], row["dn_set"] = data_tags(data_dir)
    return row


def data_source(data_dir) -> str:
    return metadata(data_dir).get("source", "desconhecido")


def metadata(data_dir) -> dict:
    with open(Path(data_dir) / "metadata.json", encoding="utf-8") as f:
        return json.load(f)


def data_tags(data_dir) -> tuple[str, str]:
    """(dataset, dn_set). Metadados antigos, sem esses campos, sao o
    male-cns:v1.0 com o conjunto original de descendentes."""
    meta = metadata(data_dir)
    if meta.get("source") != "real":
        return ("synthetic", meta.get("dn_set", "original"))
    return (meta.get("dataset", "male-cns:v1.0"), meta.get("dn_set", "original"))


def origin(row: dict) -> tuple:
    """(origem dos dados, eixo da retina, escopo da homeostase, saques de
    treino) de uma linha de resultado."""
    cfg = row.get("config", {})
    dataset_default = "male-cns:v1.0" if row.get("data_source") == "real" else "synthetic"
    return (str(row.get("data_source")), row.get("dataset", dataset_default),
            row.get("dn_set", "original"), cfg.get("retina_axis", "pca"),
            cfg.get("homeostasis_scope", "plastic"),
            int(cfg.get("n_serves", st.DEFAULTS["n_serves"])))


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
    ap.add_argument("--out", default=None,
                    help="padrao: docs/achado-19-resultados-<source>[-elevation].jsonl")
    ap.add_argument("--retina-axis", choices=["pca", "elevation"], default="pca",
                    help="elevation: posicao na retina = elevacao dentro de cada olho "
                         "(exige retina_y; ver scripts/add_retina_coords.py)")
    ap.add_argument("--homeostasis-scope", choices=["plastic", "visual"], default="plastic")
    ap.add_argument("--serves", type=int, default=st.DEFAULTS["n_serves"])
    ap.add_argument("--trials", type=int, default=se.DEFAULT_TRIALS)
    ap.add_argument("--curve-every", type=int, default=500)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()

    if not (Path(args.data_dir) / "neurons.parquet").exists():
        print(f"ERRO: dados do conectoma nao encontrados em {args.data_dir}.")
        return 2
    source = data_source(args.data_dir)
    dataset, dn_set = data_tags(args.data_dir)
    suffix = ""
    if source == "real" and dataset != "male-cns:v1.0":
        suffix += "-" + dataset.replace(":", "-")
    if dn_set != "original":
        suffix += f"-dn-{dn_set}"
    if args.retina_axis != "pca":
        suffix += f"-{args.retina_axis}"
    if args.homeostasis_scope != "plastic":
        suffix += f"-{args.homeostasis_scope}"
    if args.serves != st.DEFAULTS["n_serves"]:
        suffix += f"-{args.serves}serves"
    out = Path(args.out) if args.out else HERE / "docs" / f"achado-19-resultados-{source}{suffix}.jsonl"
    rows = load(out)
    this = (source, dataset, dn_set, args.retina_axis, args.homeostasis_scope, args.serves)
    other = {origin(r) for r in rows.values()} - {this}
    if other:
        print(f"ERRO: {out} tem resultados de outra origem/eixo ({sorted(other)}); "
              f"esta rodada e {this}. Use outro --out.")
        return 2
    print(f"[achado19] dados: {source} {dataset} dn={dn_set} ({args.data_dir}), "
          f"retina: {args.retina_axis}, "
          f"homeostase: {args.homeostasis_scope}, saques: {args.serves}; "
          f"resultados: {out}", flush=True)
    if not args.summary_only:
        jobs = [
            (args.data_dir, cond, pol, tr, ev, args.trials, args.curve_every,
             args.retina_axis, args.homeostasis_scope, args.serves)
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
