#!/usr/bin/env python3
"""Lista neuronios descendentes (DN*) candidatos para ampliar o readout motor.

Pergunta: quais descendentes recebem mais sinapses dos neuronios LC (projecao
colunar do lobulo)? Duas consultas ao neuPrint (so leitura):

  1. dos LC que ja estao no subgrafo (object: LC4/LPLC2; target: LC10a);
  2. de TODOS os tipos LC*/LPLC* -- para ver se outros LC levam a
     descendentes de direcao de voo que hoje nao alcancamos.

Para cada tipo de descendente, mostra peso total recebido, quantos neuronios
(e de que lado, _L/_R) e quais tipos LC mais contribuem. Salva tudo em
docs/dn_candidatos_*.csv. Nao altera connectome_data/.

Uso (PowerShell):
    $env:NEUPRINT_TOKEN = "SEU_TOKEN"
    python scripts/find_dn_candidates.py
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import fetch_connectome as fc

DN_REGEX = r"DN.*"
ALL_LC_REGEX = r"LC.*|LPLC.*"


def side(instance: str) -> str:
    if re.search(r"(_R$|\(R\))", str(instance)):
        return "R"
    if re.search(r"(_L$|\(L\))", str(instance)):
        return "L"
    return "?"


def query(client, src_regex: str, min_weight: int) -> pd.DataFrame:
    from neuprint import NeuronCriteria as NC, fetch_adjacencies

    neurons, conn = fetch_adjacencies(
        sources=NC(type=src_regex, regex=True, client=client),
        targets=NC(type=DN_REGEX, regex=True, client=client),
        min_total_weight=min_weight, client=client,
    )
    conn = conn.groupby(["bodyId_pre", "bodyId_post"], as_index=False)["weight"].sum()
    info = neurons.set_index("bodyId")[["type", "instance"]]
    conn["pre_type"] = conn["bodyId_pre"].map(info["type"])
    conn["post_type"] = conn["bodyId_post"].map(info["type"])
    conn["post_instance"] = conn["bodyId_post"].map(info["instance"])
    conn["post_side"] = conn["post_instance"].map(side)
    return conn


def summarize(conn: pd.DataFrame, top: int) -> pd.DataFrame:
    rows = []
    for dn_type, g in conn.groupby("post_type"):
        by_pre = g.groupby("pre_type")["weight"].sum().sort_values(ascending=False)
        cells = g.drop_duplicates("bodyId_post")
        rows.append({
            "dn_type": dn_type,
            "total_weight": int(g["weight"].sum()),
            "n_cells": len(cells),
            "sides": "".join(sorted(cells["post_side"])),
            "top_lc_inputs": ", ".join(f"{t}:{int(w)}" for t, w in by_pre.head(4).items()),
        })
    return pd.DataFrame(rows).sort_values("total_weight", ascending=False).head(top)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--token", default=os.environ.get("NEUPRINT_TOKEN"))
    ap.add_argument("--min-weight", type=int, default=5,
                    help="peso minimo por par pre->pos para contar a conexao")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()
    if not args.token:
        print("ERRO: defina NEUPRINT_TOKEN ou passe --token.", file=sys.stderr)
        return 2

    from neuprint import Client
    client = Client(fc.NEUPRINT_SERVER, dataset=fc.DATASET, token=args.token)
    current_lc = f"{fc.ROLE_TYPE_REGEX['object']}|{fc.ROLE_TYPE_REGEX['target']}"
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 80)

    out_dir = HERE / "docs"
    for label, regex in (("lc_do_subgrafo", current_lc), ("todos_lc", ALL_LC_REGEX)):
        print(f"\n=== descendentes que recebem de {label} ({regex}) ===", flush=True)
        conn = query(client, regex, args.min_weight)
        table = summarize(conn, args.top)
        print(table.to_string(index=False))
        path = out_dir / f"dn_candidatos_{label}.csv"
        table.to_csv(path, index=False)
        print(f"[find_dn_candidates] salvo: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
