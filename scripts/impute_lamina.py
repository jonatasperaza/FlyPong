#!/usr/bin/env python3
"""Completa as colunas da lamina sem R1-R6 traçados (imputacao documentada).

No MaleCNS v0.9 a lamina foi reconstruida so em parte: 66% das colunas do
olho esquerdo e 43% do direito nao tem nenhum R1-R6 ligado ao L1 da coluna
(o resto do lobo optico e simetrico). Na mosca, toda coluna recebe seis
R1-R6 (superposicao neural), uma das ligacoes mais estereotipadas do
sistema visual. Para cada coluna (posicao do L1) sem entrada de R1-R6,
este script cria um fotorreceptor virtual "R1-R6_virtual" (um neuronio
representando os seis) ligado a L1, L2, L3 e T1 da mesma coluna, com a
mediana das sinapses que as colunas traçadas do mesmo olho recebem.
Os R1-R6 reais sao mantidos. Saida em outro diretorio.

Uso:
    python scripts/impute_lamina.py --src connectome_v09_fv --out connectome_v09_fvi
    python scripts/add_visual_directions.py --data-dir connectome_v09_fvi
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

TARGETS = ("L1", "L2", "L3", "T1")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    n = pd.read_parquet(src / "neurons.parquet")
    e = pd.read_parquet(src / "edges.parquet")
    meta = json.loads((src / "metadata.json").read_text())
    t = n.set_index("bodyId")["type"].astype(str)
    n["side"] = n.instance.astype(str).str[-1]
    # alvos sem coluna anotada (L3 do olho esquerdo): coluna pelos alvos
    # deles, ponderada por sinapses (como os fotorreceptores)
    hexes = n.set_index("bodyId")[["hex1", "hex2"]]
    lack = n[n.type.isin(TARGETS) & n.hex1.isna()].bodyId
    if len(lack):
        x = e[e.pre.isin(lack)].join(hexes, on="post").dropna()
        g = x.assign(a=x.hex1 * x.weight, b=x.hex2 * x.weight).groupby("pre")[["a", "b", "weight"]].sum()
        inferred = pd.DataFrame({"hex1": g.a / g.weight, "hex2": g.b / g.weight})
        m = n.bodyId.isin(inferred.index)
        n.loc[m, "hex1"] = n.loc[m, "bodyId"].map(inferred.hex1)
        n.loc[m, "hex2"] = n.loc[m, "bodyId"].map(inferred.hex2)
        meta["hex_inferred_for_targets"] = int(m.sum())
    col = n.dropna(subset=["hex1", "hex2"]).copy()
    col["key"] = list(zip(col.side, col.hex1.round().astype(int), col.hex2.round().astype(int)))

    r = e[e.pre.map(t) == "R1-R6"]
    tot = r.groupby("post").weight.sum()
    new_n, new_e, stats = [], [], {}
    next_id = -1
    for s in ("L", "R"):
        l1 = col[(col.type == "L1") & (col.side == s)]
        has = l1.bodyId.map(tot).fillna(0) > 0
        med = {}
        for ty in TARGETS:
            tg = col[(col.type == ty) & (col.side == s)]
            got = tg.bodyId.map(tot).fillna(0)
            med[ty] = float(np.median(got[got > 0])) if (got > 0).any() else 0.0
        by_key = {ty: col[(col.type == ty) & (col.side == s)].groupby("key").bodyId.first() for ty in TARGETS}
        missing = l1[~has.to_numpy()]
        for _, row in missing.iterrows():
            bid = next_id
            next_id -= 1
            new_n.append({"bodyId": bid, "type": "R1-R6_virtual", "instance": f"R1-R6_virtual_{s}",
                          "status": "imputado", "role": "photoreceptor", "nt": "histamine",
                          "nt_confidence": np.nan, "flyvis_type": "R1",
                          "hex1": float(round(row.hex1)), "hex2": float(round(row.hex2))})
            for ty in TARGETS:
                tgt = by_key[ty].get(row.key)
                if tgt is not None and med[ty] > 0:
                    new_e.append({"pre": bid, "post": int(tgt), "weight": int(round(med[ty]))})
        stats[s] = {"colunas_L1": int(len(l1)), "sem_R1_R6": int((~has).sum()),
                    "mediana_sinapses_por_coluna": med}
    vn = pd.DataFrame(new_n)
    fit = meta["retina_y_fit"]
    for s in ("L", "R"):
        m = vn.instance.str.endswith(s)
        c = fit[s]["coef"]
        vn.loc[m, "retina_y"] = c[0] * vn.loc[m, "hex1"] + c[1] * vn.loc[m, "hex2"] + c[2]
    n2 = pd.concat([n.drop(columns=["side"]), vn], ignore_index=True)
    ry = n2.retina_y
    n2.loc[n2.role == "photoreceptor", "retina_pos"] = (ry - ry.min()) / (ry.max() - ry.min())
    e2 = pd.concat([e, pd.DataFrame(new_e)], ignore_index=True)
    out.mkdir(parents=True, exist_ok=True)
    n2.to_parquet(out / "neurons.parquet", index=False)
    e2.to_parquet(out / "edges.parquet", index=False)
    meta.update({"input_set": "flyvis_imputed", "n_neurons": int(len(n2)), "n_edges": int(len(e2)),
                 "lamina_imputation": {"stats": stats, "virtual_neurons": int(len(vn)),
                                       "virtual_edges": int(len(new_e)),
                                       "method": "um R1-R6 virtual por coluna sem R1-R6 traçado; mediana por olho"}})
    meta.pop("visual_directions", None)
    (out / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(meta["lamina_imputation"], indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
