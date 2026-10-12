#!/usr/bin/env python3
"""Teste de cruzamento de seeds (diagnostico apos o Achado 22).

No Achado 22, a seed de treino 1 chegou a 82-84% e as outras a 15-39%. Cada
seed fixa duas coisas: o estado em que a rede sai da fase de desenvolvimento
(dev_seed = seed + DEV_SEED_OFFSET) e a sequencia de treino (saques e ruido).
Este script cruza as duas, na condicao normal da configuracao A, avaliando
na seed de VALIDACAO 2001 (nunca nas seeds de teste):

    (dev 1, treino 1)  (dev 1, treino 3)  (dev 3, treino 1)  (dev 3, treino 3)

Se o sucesso acompanha a seed do desenvolvimento, o que importa e o estado
da rede antes do treino; se acompanha a seed do treino, importa a sequencia
de aprendizado. Usa checkpoints e retoma de onde parou.
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

import serve_train as st

ARMS = [(1, 1), (1, 3), (3, 1), (3, 3)]  # (seed do desenvolvimento, seed do treino)


def _job(a):
    data_dir, condition, dev, tr, eval_seed, trials, out, variant = a
    tag = "" if not variant else "_" + "_".join(f"{k}{v}" for k, v in sorted(variant.items()))
    ck = HERE / "runs" / "checkpoints" / Path(out).stem / f"{condition}_dev{dev}_tr{tr}{tag}.pkl"
    kw = dict(condition=condition, policy="trained", train_seed=tr,
              dev_seed=dev + st.DEV_SEED_OFFSET, eval_seed=eval_seed, eval_trials=trials,
              n_serves=1500, curve_every=500, curve_trials=100, verbose=False,
              retina_axis="elevation", homeostasis_scope="all_visual",
              checkpoint_path=str(ck))
    kw.update(variant)
    row = st.run_condition(data_dir, **kw)
    row["arm"] = {"dev": dev, "train": tr, **variant}
    row["_ck"] = str(ck)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(HERE / "connectome_v09_in21"))
    ap.add_argument("--condition", default="normal", choices=["normal", "inverted"])
    ap.add_argument("--eval-seed", type=int, default=2001)
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--out", default=str(HERE / "docs" / "cruzamento-seeds.jsonl"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--arms", default=None,
                    help='JSON: lista de [dev, treino, {opcoes de run_condition}]')
    args = ap.parse_args()
    out = Path(args.out)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["condition"], json.dumps(r["arm"], sort_keys=True)))
    arms = [(d, t, {}) for d, t in ARMS] if args.arms is None else json.loads(args.arms)
    jobs = []
    for d, t, v in arms:
        key = (args.condition, json.dumps({"dev": d, "train": t, **v}, sort_keys=True))
        if key not in done:
            jobs.append((args.data_dir, args.condition, d, t, args.eval_seed, args.trials, str(out), v))
    print(f"[cruzamento] {len(jobs)} pendentes", flush=True)
    with Pool(args.workers) as pool:
        for row in pool.imap_unordered(_job, jobs):
            ck = row.pop("_ck")
            with open(out, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            Path(ck).unlink(missing_ok=True)
            print(f"[cruzamento] {row['condition']} {row['arm']} "
                  f"acerto={row['hit_rate']:.3f} curva={[round(c['hit_rate'], 2) for c in row['curve']]}",
                  flush=True)
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    for r in sorted(rows, key=lambda r: (r["condition"], json.dumps(r["arm"], sort_keys=True))):
        print(f"{r['condition']:8s} {r['arm']}: "
              f"{r['hit_rate']:.3f} [{r['ci95_low']:.3f},{r['ci95_high']:.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
