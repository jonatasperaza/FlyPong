#!/usr/bin/env python3
"""Acrescenta retina_x/retina_y/retina_z a um neurons.parquet ja baixado.

Baixa so as sinapses dos fotorreceptores (nao o grafo inteiro) e grava o
centroide (x, y, z) de cada um em coordenadas do neuPrint. Essas colunas sao
exigidas por `retina_axis="elevation"` (sim/network.py). `retina_pos` e as
arestas nao mudam, entao os Achados anteriores continuam reproduziveis.

Uso (PowerShell):
    $env:NEUPRINT_TOKEN = "SEU_TOKEN"
    python scripts/add_retina_coords.py

Antes de sobrescrever, guarda uma copia em neurons.before_retina_coords.parquet.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import fetch_connectome as fc


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(HERE / "connectome_data"))
    ap.add_argument("--token", default=os.environ.get("NEUPRINT_TOKEN"))
    ap.add_argument("--dataset", default=fc.DATASET,
                    help=f"dataset do neuPrint (padrao: {fc.DATASET})")
    args = ap.parse_args()
    if not args.token:
        print("ERRO: defina NEUPRINT_TOKEN ou passe --token.", file=sys.stderr)
        return 2

    path = Path(args.data_dir) / "neurons.parquet"
    ndf = pd.read_parquet(path)
    photo = ndf[ndf["role"] == "photoreceptor"]
    print(f"[add_retina_coords] {len(photo)} fotorreceptores em {path}")

    client = fc.connect(args.token, args.dataset)
    if client is None:
        return 2
    centroid = fc.fetch_photoreceptor_centroids(photo, client)
    with_coords = fc.add_centroid_columns(photo, centroid)
    n_valid = int(with_coords["retina_y"].notna().sum())
    print(f"[add_retina_coords] centroide encontrado para {n_valid}/{len(photo)}")
    if n_valid == 0:
        print("ERRO: nenhuma sinapse encontrada; nada foi gravado.", file=sys.stderr)
        return 1

    for axis in ("x", "y", "z"):
        col = f"retina_{axis}"
        ndf[col] = np.nan
        ndf.loc[with_coords.index, col] = with_coords[col]

    # Diagnostico: o eixo de retina_pos (PCA) deve separar os olhos; o y
    # deve variar dentro de cada olho.
    p = ndf[ndf["role"] == "photoreceptor"]
    eye = p["instance"].astype(str).str.extract(r"_(L|R)$")[0].fillna("?")
    for e, g in p.groupby(eye):
        print(f"[add_retina_coords] olho {e}: n={len(g)} "
              f"x=[{g.retina_x.min():.0f},{g.retina_x.max():.0f}] "
              f"y=[{g.retina_y.min():.0f},{g.retina_y.max():.0f}] "
              f"z=[{g.retina_z.min():.0f},{g.retina_z.max():.0f}] "
              f"corr(retina_pos, x)={np.corrcoef(g.retina_pos, g.retina_x)[0, 1]:+.2f}")

    backup = path.with_name("neurons.before_retina_coords.parquet")
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"[add_retina_coords] copia de seguranca: {backup}")
    ndf.to_parquet(path, index=False)
    print(f"[add_retina_coords] gravado: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
