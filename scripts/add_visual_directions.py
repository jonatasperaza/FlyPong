#!/usr/bin/env python3
"""Direcao no campo visual (azimute, elevacao em graus) de cada fotorreceptor.

So usa o conectoma (subgrafo com colunas hex1/hex2, ver
build_subgraph_local.py):
  - eixo "para cima": direcao preferida dos T4c (sobe) e oposta dos T4d,
    medida pela geometria do dendrito (entradas de Mi4 na base menos Mi9 na
    ponta; para T5, Tm1 menos Tm9). Validado pela borda dorsal (R7d/R8d),
    que fica no mesmo canto;
  - eixo "para frente": direcao preferida dos T4b (de tras para frente) e
    oposta dos T4a (de frente para tras);
  - escala: angulo entre omatidios de 4,5 graus para a distancia mediana
    entre colunas vizinhas da grade de colunas (Mi1 de cada olho);
  - azimute: 0 na linha media; a borda frontal de cada olho fica 10 graus
    alem da linha media (sobreposicao binocular); olho direito com azimute
    positivo, esquerdo negativo. Elevacao: 0 na mediana do olho.
Grava as colunas az_deg e el_deg em neurons.parquet (NaN fora dos
fotorreceptores) e os eixos medidos em metadata.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parents[1]
INTEROMMATIDIAL_DEG = 4.5
BINOCULAR_OVERLAP_DEG = 10.0


def dendrite_vectors(neurons, edges, fam, tip, base):
    t = neurons.set_index("bodyId")["type"].astype(str)
    hexes = neurons.set_index("bodyId")[["hex1", "hex2"]]

    def centroid(pre_type):
        x = edges[(edges.pre.map(t) == pre_type) & edges.post.map(t).str.match(rf"^{fam}[a-d]$")]
        x = x.join(hexes, on="pre").dropna()
        g = x.assign(a=x.hex1 * x.weight, b=x.hex2 * x.weight).groupby("post")[["a", "b", "weight"]].sum()
        return pd.DataFrame({"h1": g.a / g.weight, "h2": g.b / g.weight})

    v = (centroid(base) - centroid(tip)).dropna()
    v["type"] = v.index.map(t).str[-1]
    return v.groupby("type")[["h1", "h2"]].mean()


def unit(x):
    x = np.asarray(x, dtype=float)
    return x / np.linalg.norm(x)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(HERE / "connectome_v09_fv"))
    args = ap.parse_args()
    d = Path(args.data_dir)
    neurons = pd.read_parquet(d / "neurons.parquet")
    edges = pd.read_parquet(d / "edges.parquet")
    v4 = dendrite_vectors(neurons, edges, "T4", "Mi9", "Mi4")
    v5 = dendrite_vectors(neurons, edges, "T5", "Tm9", "Tm1")
    up = unit(unit(v4.loc["c"]) - unit(v4.loc["d"]) + unit(v5.loc["c"]) - unit(v5.loc["d"]))
    front = unit(unit(v4.loc["b"]) - unit(v4.loc["a"]) + unit(v5.loc["b"]) - unit(v5.loc["a"]))
    angle = float(np.degrees(np.arccos(np.clip(up @ front, -1, 1))))

    ph = neurons.role == "photoreceptor"
    side = neurons.instance.astype(str).str[-1]
    H = neurons[["hex1", "hex2"]].to_numpy(dtype=float)
    p_up = H @ up
    p_front = H @ front
    neurons["az_deg"] = np.nan
    neurons["el_deg"] = np.nan
    dra = {}
    scale = {}
    for s in ("L", "R"):
        m = (ph & (side == s)).to_numpy()
        # escala pela grade de colunas (posicoes inteiras dos Mi1 deste olho)
        col = ((neurons.type.astype(str) == "Mi1") & (side == s)).to_numpy()
        lat = np.unique(H[col & ~np.isnan(H).any(1)], axis=0)
        uniq = np.c_[lat @ front, lat @ up]
        dist, _ = cKDTree(uniq).query(uniq, k=2)
        k = INTEROMMATIDIAL_DEG / float(np.median(dist[:, 1]))
        scale[s] = k
        el = (p_up[m] - np.median(p_up[m])) * k
        az = (np.percentile(p_front[m], 99.5) - p_front[m]) * k - BINOCULAR_OVERLAP_DEG
        neurons.loc[m, "el_deg"] = el
        neurons.loc[m, "az_deg"] = az if s == "R" else -az
        is_dra = m & neurons.type.astype(str).str.endswith("d").to_numpy()
        dra[s] = float(np.median(neurons.loc[is_dra, "el_deg"])) if is_dra.any() else None
    neurons.to_parquet(d / "neurons.parquet", index=False)
    meta = json.loads((d / "metadata.json").read_text())
    meta["visual_directions"] = {
        "up_axis_hex": up.round(4).tolist(), "front_axis_hex": front.round(4).tolist(),
        "angle_up_front_deg": round(angle, 1), "deg_per_hex_unit": {s: round(v, 3) for s, v in scale.items()},
        "dorsal_rim_median_elevation_deg": dra,
        "interommatidial_deg": INTEROMMATIDIAL_DEG, "binocular_overlap_deg": BINOCULAR_OVERLAP_DEG,
        "method": "eixos pela direcao preferida dos T4/T5 (dendritos), escala pelo angulo entre omatidios",
    }
    (d / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(meta["visual_directions"], indent=1))
    p = neurons[ph]
    print(p.groupby(p.instance.str[-1])[["az_deg", "el_deg"]].describe().round(1).T.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
