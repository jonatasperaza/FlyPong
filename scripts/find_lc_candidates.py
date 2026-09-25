#!/usr/bin/env python3
"""Quais tipos LC alimentam os descendentes, e eles funcionariam no subgrafo?

Para cada tipo LC/LPLC que projeta para os descendentes (conjunto --dn-set de
fetch_connectome.DN_SETS), mostra:

  - peso total para o DNa10 e para o conjunto inteiro de descendentes;
  - numero de neuronios do tipo;
  - fracao do peso de ENTRADA do tipo que vem de tipos ja presentes no
    subgrafo (fotorreceptores, interneuronios, T4/T5, LC4/LPLC2, LC10a).
    Um tipo com fracao baixa ficaria quase sem entrada no modelo e nao
    carregaria informacao, mesmo se incluido;
  - os tipos de entrada mais fortes que NAO estao no subgrafo.

So leitura. Salva docs/lc_candidatos_<dataset>.csv.

Uso (PowerShell):
    $env:NEUPRINT_TOKEN = "SEU_TOKEN"
    python scripts/find_lc_candidates.py --dataset male-cns:v0.9
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

ALL_LC_REGEX = r"LC.*|LPLC.*"
SUBGRAPH_ROLES = ("photoreceptor", "interneuron", "motion", "object", "target")


def in_subgraph(type_name: str) -> bool:
    t = str(type_name)
    return any(re.fullmatch(fc.ROLE_TYPE_REGEX[r], t) for r in SUBGRAPH_ROLES)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--token", default=os.environ.get("NEUPRINT_TOKEN"))
    ap.add_argument("--dataset", default=fc.DATASET)
    ap.add_argument("--dn-set", choices=sorted(fc.DN_SETS), default="achado20")
    ap.add_argument("--min-weight", type=int, default=5)
    ap.add_argument("--top", type=int, default=15,
                    help="quantos tipos LC analisar (os que mais projetam para o DNa10 "
                         "e para o conjunto de descendentes)")
    args = ap.parse_args()
    if not args.token:
        print("ERRO: defina NEUPRINT_TOKEN ou passe --token.", file=sys.stderr)
        return 2
    client = fc.connect(args.token, args.dataset)
    if client is None:
        return 2
    from neuprint import NeuronCriteria as NC, fetch_simple_connections

    # 1) LC -> descendentes
    out = fetch_simple_connections(
        NC(type=ALL_LC_REGEX, regex=True, client=client),
        NC(type=fc.DN_SETS[args.dn_set], regex=True, client=client),
        min_weight=args.min_weight, client=client,
    )
    to_dn = out.groupby("type_pre")["weight"].sum()
    to_dna10 = out[out["type_post"].astype(str).str.startswith("DNa10")].groupby("type_pre")["weight"].sum()
    ranked = pd.concat([to_dna10.sort_values(ascending=False).head(args.top),
                        to_dn.sort_values(ascending=False).head(args.top)]).index.unique()

    rows = []
    for lc_type in ranked:
        # 2) entradas do tipo LC
        inp = fetch_simple_connections(
            None, NC(type=lc_type, client=client),
            min_weight=args.min_weight, client=client,
        )
        total = float(inp["weight"].sum()) if len(inp) else 0.0
        by_type = inp.groupby("type_pre")["weight"].sum().sort_values(ascending=False)
        inside = float(sum(w for t, w in by_type.items() if in_subgraph(t)))
        missing = [(t, int(w)) for t, w in by_type.items() if not in_subgraph(t)][:4]
        n_cells = int(inp["bodyId_post"].nunique()) if len(inp) else 0
        rows.append({
            "lc_type": lc_type,
            "n_cells": n_cells,
            "to_DNa10": int(to_dna10.get(lc_type, 0)),
            "to_dn_set": int(to_dn.get(lc_type, 0)),
            "input_total": int(total),
            "frac_input_from_subgraph": round(inside / total, 3) if total else 0.0,
            "top_missing_inputs": ", ".join(f"{t}:{w}" for t, w in missing),
        })
        print(f"[find_lc_candidates] {lc_type}: {rows[-1]['frac_input_from_subgraph']:.3f} "
              f"da entrada vem do subgrafo", flush=True)

    table = pd.DataFrame(rows).sort_values("to_DNa10", ascending=False)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 70)
    print(table.to_string(index=False))
    tag = re.sub(r"[^A-Za-z0-9.]+", "-", args.dataset)
    path = HERE / "docs" / f"lc_candidatos_{tag}.csv"
    table.to_csv(path, index=False)
    print(f"[find_lc_candidates] salvo: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
