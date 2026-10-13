#!/usr/bin/env python3
"""Constroi um subgrafo a partir das tabelas publicas do MaleCNS v0.9 (sem neuPrint).

Fontes (gs://flyem-male-cns/v0.9/connectome-data/flat-connectome/, baixadas em
runs/mcns_nt/): body-annotations, body-neurotransmitters e connectome-weights
(minconf 0,5). Conjunto "flyvis" (etapa 1 do plano pos-Achado 25):
  - photoreceptor: R1-R6, R7*, R8* (todos os subtipos);
  - interneuron: todos os tipos do lobo optico com parametros no flyvis
    (Lappalainen et al. 2024) presentes no MaleCNS, mais os interneuronios
    do subgrafo connectome_v09_in21;
  - motion: T4*, T5*;
  - object, target, descending: os mesmos neuronios de connectome_v09_in21;
  - dopaminergic: os de connectome_v09_in21 (PAM, PPL1) mais PAL03 (o unico
    tipo dopaminergico com sinapses nos LC10c/LC10d) e PPM1203 (LC17).
Posicao na retina: a coluna hexagonal (assignedOlHex1/2) dos alvos de cada
fotorreceptor, ponderada por sinapses; retina_y e um mapa linear da coluna,
ajustado por olho contra a elevacao usada ate o Achado 25 (R^2 0,992).
Colunas extras: flyvis_type, hex1, hex2, nt, nt_confidence.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import sim.flyvis_params as fp

SRC = HERE / "runs" / "mcns_nt"
BASE = HERE / "connectome_v09_in21"

# tipo do MaleCNS -> tipo do flyvis (os demais: mesmo nome)
FLYVIS_ALIAS = {"R1-R6": "R1", "Am1": "Am", "CT1": "CT1(M10)", "TmY9a": "TmY9", "TmY9b": "TmY9"}


def flyvis_type(t: str, fv_types: set) -> str | None:
    if t in FLYVIS_ALIAS:
        return FLYVIS_ALIAS[t]
    if t.startswith("R7"):
        return "R7"
    if t.startswith("R8"):
        return "R8"
    return t if t in fv_types else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(HERE / "connectome_v09_fv"))
    args = ap.parse_args()

    fv = fp.load()
    fv_types = set(fv.types)
    ann = pd.read_feather(SRC / "body-annotations-male-cns-v0.9-minconf-0.5.feather")
    ann["type"] = ann["type"].fillna("").astype(str)
    base = pd.read_parquet(BASE / "neurons.parquet")

    t = ann["type"]
    is_photo = t.str.match(r"^(R1-R6|R7.*|R8.*)$")
    is_motion = t.str.match(r"^T[45][a-d]$")
    ann["fv"] = [flyvis_type(x, fv_types) for x in t]
    base_inter = set(base.loc[base.role == "interneuron", "type"].astype(str))
    is_inter = (ann["fv"].notna() | t.isin(base_inter)) & ~is_photo & ~is_motion
    keep_roles = {r: set(base.loc[base.role == r, "bodyId"]) for r in ("object", "target", "descending")}
    dan_ids = set(base.loc[base.role == "dopaminergic", "bodyId"]) | set(
        ann.loc[t.isin(["PAL03", "PPM1203"]), "bodyId"])

    role = pd.Series(pd.NA, index=ann.index, dtype="object")
    role[is_photo] = "photoreceptor"
    role[is_inter] = "interneuron"
    role[is_motion] = "motion"
    for r, ids in keep_roles.items():
        role[ann.bodyId.isin(ids)] = r
    role[ann.bodyId.isin(dan_ids)] = "dopaminergic"
    ann["role"] = role
    sel = ann[ann.role.notna()].copy()

    # posicao dos fotorreceptores pela coluna dos alvos
    w = feather.read_table(SRC / "connectome-weights-male-cns-v0.9-minconf-0.5.feather", memory_map=True)
    photo_ids = pa.array(sel.loc[sel.role == "photoreceptor", "bodyId"].values)
    pe = w.filter(pc.is_in(w["body_pre"], value_set=photo_ids)).to_pandas()
    hexes = ann.set_index("bodyId")[["assignedOlHex1", "assignedOlHex2"]]
    pe = pe.join(hexes, on="body_post").dropna()
    pe["h1w"] = pe.assignedOlHex1 * pe.weight
    pe["h2w"] = pe.assignedOlHex2 * pe.weight
    g = pe.groupby("body_pre")[["h1w", "h2w", "weight"]].sum()
    sel = sel.set_index("bodyId")
    sel["hex1"] = sel["assignedOlHex1"]
    sel["hex2"] = sel["assignedOlHex2"]
    sel.loc[g.index, "hex1"] = g.h1w / g.weight
    sel.loc[g.index, "hex2"] = g.h2w / g.weight
    n_before = int((sel.role == "photoreceptor").sum())
    sel = sel[~((sel.role == "photoreceptor") & sel.hex1.isna())]

    # retina_y: mapa linear da coluna, ajustado por olho contra a elevacao antiga
    sel["side"] = sel["instance"].astype(str).str[-1]
    old = base.set_index("bodyId")
    sel["retina_y"] = np.nan
    fits = {}
    for side in ("L", "R"):
        m = (sel.role == "photoreceptor") & (sel.side == side)
        ref = sel[m].join(old[["retina_y"]], rsuffix="_old").dropna(subset=["retina_y_old"])
        X = np.c_[ref.hex1, ref.hex2, np.ones(len(ref))]
        c = np.linalg.lstsq(X, ref.retina_y_old.values, rcond=None)[0]
        pred = X @ c
        r2 = 1 - ((ref.retina_y_old - pred) ** 2).sum() / ((ref.retina_y_old - ref.retina_y_old.mean()) ** 2).sum()
        fits[side] = {"coef": c.round(4).tolist(), "r2": round(float(r2), 4), "n_ref": int(len(ref))}
        sel.loc[m, "retina_y"] = np.c_[sel.loc[m, "hex1"], sel.loc[m, "hex2"], np.ones(int(m.sum()))] @ c
    for col in ("retina_x", "retina_z"):
        sel[col] = np.nan
    ry = sel.retina_y
    sel["retina_pos"] = (ry - ry.min()) / (ry.max() - ry.min())

    nt = pd.read_feather(SRC / "body-neurotransmitters-male-cns-v0.9.feather",
                         columns=["body", "consensus_nt", "predicted_nt_confidence"]).set_index("body")
    sel = sel.join(nt.rename(columns={"consensus_nt": "nt", "predicted_nt_confidence": "nt_confidence"}))
    sel["nt"] = sel["nt"].fillna("unknown").astype(str)
    sel["status"] = sel["statusLabel"].astype(str)
    neurons = sel.reset_index().rename(columns={"fv": "flyvis_type"})[
        ["bodyId", "type", "instance", "status", "retina_pos", "retina_x", "retina_y", "retina_z",
         "role", "nt", "nt_confidence", "flyvis_type", "hex1", "hex2"]]
    neurons["instance"] = neurons["instance"].astype(str)

    ids = pa.array(neurons.bodyId.values)
    edges = w.filter(pc.and_(pc.is_in(w["body_pre"], value_set=ids),
                             pc.is_in(w["body_post"], value_set=ids))).to_pandas()
    edges = edges.rename(columns={"body_pre": "pre", "body_post": "post"})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    neurons.to_parquet(out / "neurons.parquet", index=False)
    edges.to_parquet(out / "edges.parquet", index=False)
    meta = {
        "source": "real", "dataset": "male-cns:v0.9", "dn_set": "original", "input_set": "flyvis",
        "built_from": "tabelas publicas gs://flyem-male-cns/v0.9 (minconf 0.5)",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "n_neurons": int(len(neurons)), "n_edges": int(len(edges)),
        "photoreceptors_without_column_dropped": n_before - int((neurons.role == "photoreceptor").sum()),
        "retina_y_fit": fits,
        "citation": "Berg et al., MaleCNS (Cell, 2026); flyvis: Lappalainen et al., Nature 2024.",
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(meta, indent=1, ensure_ascii=False))
    print(neurons.role.value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
