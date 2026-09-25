#!/usr/bin/env python3
"""Completude de entrada dos LC: quais tipos do lobo optico faltam no subgrafo?

Achado 20: os LC do modelo recebem so uma fracao pequena da entrada real
(LC10a 12,9%, LC4 23,3%, LPLC2 49%), porque o subgrafo original so inclui
L1-5, Mi1/4/9, Tm1/2/4/9, T4/T5. Este script:

  1. soma, sobre os LC alvo, o peso de entrada vindo de tipos FORA do
     subgrafo e ranqueia esses tipos, mantendo so tipos do lobo optico
     (colunares/intrinsecos: Tm, TmY, T2/T3, Li, Y, Lpi, Mi, Dm, Pm, C2/C3...).
     Entradas do cerebro central (AOTU, TuTuA, PVLP, LT...) ficam de fora:
     sao realimentacao e exigiriam modelar outras regioes;
  2. escolhe os N tipos mais fortes (--n-types) e mede, com eles incluidos,
     (a) a nova completude de entrada de cada LC alvo e (b) a completude de
     entrada de cada tipo novo (se ele proprio seria alimentado pelo
     subgrafo, ou ficaria mudo);
  3. imprime a regex sugerida para fetch_connectome.

So leitura. Salva docs/input_completeness_<dataset>.csv.

Uso (PowerShell):
    $env:NEUPRINT_TOKEN = "SEU_TOKEN"
    python scripts/input_completeness.py --dataset male-cns:v0.9
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

TARGET_LC = ["LC4", "LPLC2", "LC10a", "LC10c-1", "LC10c-2", "LC10d"]
OPTIC_LOBE = r"(Tm|TmY|T2|T3|Li|Y|Lpi|Mi|Dm|Pm|C2|C3|Lawf|L[1-5]|T1).*"
SUBGRAPH_ROLES = ("photoreceptor", "interneuron", "motion", "object", "target")


def base_match(t: str) -> bool:
    return any(re.fullmatch(fc.ROLE_TYPE_REGEX[r], str(t)) for r in SUBGRAPH_ROLES)


def inputs_by_type(client, type_name: str, min_weight: int) -> pd.Series:
    from neuprint import NeuronCriteria as NC, fetch_simple_connections
    inp = fetch_simple_connections(None, NC(type=type_name, client=client),
                                   min_weight=min_weight, client=client)
    if not len(inp):
        return pd.Series(dtype=float)
    return inp.groupby("type_pre")["weight"].sum()


def completeness(by_type: pd.Series, inside) -> float:
    total = float(by_type.sum())
    if total == 0:
        return 0.0
    return float(sum(w for t, w in by_type.items() if inside(t))) / total


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--token", default=os.environ.get("NEUPRINT_TOKEN"))
    ap.add_argument("--dataset", default=fc.DATASET)
    ap.add_argument("--min-weight", type=int, default=5)
    ap.add_argument("--n-types", type=int, default=12,
                    help="quantos tipos do lobo optico faltantes incluir")
    args = ap.parse_args()
    if not args.token:
        print("ERRO: defina NEUPRINT_TOKEN ou passe --token.", file=sys.stderr)
        return 2
    client = fc.connect(args.token, args.dataset)
    if client is None:
        return 2

    lc_inputs = {}
    for t in TARGET_LC:
        lc_inputs[t] = inputs_by_type(client, t, args.min_weight)
        print(f"[input_completeness] entrada de {t}: {int(lc_inputs[t].sum())}", flush=True)

    missing = pd.concat(lc_inputs.values()).groupby(level=0).sum()
    missing = missing[[not base_match(t) and re.fullmatch(OPTIC_LOBE, str(t)) is not None
                       for t in missing.index]].sort_values(ascending=False)
    chosen = list(missing.head(args.n_types).index)
    print("\ntipos do lobo optico faltantes mais fortes (peso somado para os LC alvo):")
    print(missing.head(25).to_string())

    chosen_set = set(chosen)

    def inside(t):
        return base_match(t) or str(t) in chosen_set

    rows = []
    for t in TARGET_LC:
        rows.append({"type": t, "kind": "LC alvo",
                     "completude_antes": round(completeness(lc_inputs[t], base_match), 3),
                     "completude_depois": round(completeness(lc_inputs[t], inside), 3)})
    for t in chosen:
        by = inputs_by_type(client, t, args.min_weight)
        rows.append({"type": t, "kind": "novo",
                     "completude_antes": round(completeness(by, base_match), 3),
                     "completude_depois": round(completeness(by, inside), 3),
                     "peso_para_LC": int(missing[t])})
        print(f"[input_completeness] {t}: {rows[-1]['completude_depois']:.3f} da entrada "
              f"viria do subgrafo ampliado", flush=True)

    table = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\n" + table.to_string(index=False))
    regex = "|".join(re.escape(t) for t in chosen)
    print(f"\nregex sugerida (tipos novos): {regex}")
    tag = re.sub(r"[^A-Za-z0-9.]+", "-", args.dataset)
    path = HERE / "docs" / f"input_completeness_{tag}.csv"
    table.to_csv(path, index=False)
    print(f"[input_completeness] salvo: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
