#!/usr/bin/env python3
"""Controle de conectoma embaralhado (Achado 25).

Gera uma copia de um subgrafo com as arestas embaralhadas dentro de cada
bloco (papel pre-sinaptico -> papel pos-sinaptico), por exemplo
fotorreceptor -> interneuronio ou object -> descending:
  - em cada bloco, os neuronios POS-sinapticos das arestas sao permutados ao
    acaso; cada aresta mantem o neuronio pre-sinaptico e o peso;
  - assim cada neuronio mantem o numero de arestas de saida e de entrada e a
    forca de saida (soma de pesos) em cada bloco, e a arquitetura em camadas
    fica igual; a identidade dos parceiros (retinotopia, fiacao especifica)
    se perde;
  - pares repetidos criados pela permutacao sao somados (relatado);
  - neurons.parquet (tipos, papeis, posicao na retina, lado L/R) nao muda.

Uso:
    python scripts/make_shuffled_connectome.py --src connectome_v09_in21 \
        --out connectome_v09_in21_shuf0 --seed 0
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def shuffle_edges(neurons: pd.DataFrame, edges: pd.DataFrame, seed: int):
    role = neurons.set_index("bodyId")["role"]
    e = edges.copy()
    e["rpre"] = e["pre"].map(role).to_numpy()
    e["rpost"] = e["post"].map(role).to_numpy()
    rng = np.random.default_rng(seed)
    post = e["post"].to_numpy().copy()
    blocks = {}
    for (a, b), idx in e.groupby(["rpre", "rpost"]).indices.items():
        post[idx] = post[idx][rng.permutation(len(idx))]
        blocks[f"{a}->{b}"] = int(len(idx))
    e["post"] = post
    n_before = len(e)
    self_loops = int((e["pre"] == e["post"]).sum())
    out = e.groupby(["pre", "post"], as_index=False)["weight"].sum()
    stats = {"seed": seed, "edges_in": n_before, "edges_out": int(len(out)),
             "merged_duplicates": int(n_before - len(out)), "self_loops": self_loops,
             "blocks": blocks}
    return out, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    neurons = pd.read_parquet(src / "neurons.parquet")
    edges = pd.read_parquet(src / "edges.parquet")
    shuffled, stats = shuffle_edges(neurons, edges, args.seed)
    shutil.copy(src / "neurons.parquet", out / "neurons.parquet")
    shuffled.to_parquet(out / "edges.parquet", index=False)
    meta = json.loads((src / "metadata.json").read_text(encoding="utf-8"))
    meta["shuffle"] = {**stats, "src": str(src),
                       "method": "permutacao dos pos-sinapticos dentro de cada bloco de papeis"}
    meta["n_edges"] = int(len(shuffled))
    (out / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
                                       encoding="utf-8")
    print(json.dumps({k: v for k, v in stats.items() if k != "blocks"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
