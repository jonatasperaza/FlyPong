"""Parametros publicados do flyvis (Lappalainen et al. 2024, Nature; MIT).

O flyvis e um modelo do lobo optico da mosca com neuronios graduados (sem
spikes), restrito pelo conectoma colunar FIB25/FIB19 e treinado para estimar
fluxo optico. Cada tipo celular tem uma constante de tempo e um potencial de
repouso; cada par de tipos (pre, pos) tem um sinal e uma forca por sinapse.
Dinamica (PPNeuronIGRSynapses):

    tau_i dv_i/dt = -v_i + b_i + sum_j sinal_ij * n_syn_ij * forca_ij * relu(v_j) + entrada_i

Este modulo reconstroi a ordem dos parametros como o flyvis faz (nos e
arestas na ordem do arquivo de especificacao, agrupados sem ordenar) e le os
checkpoints sem PyTorch. A ordem e verificada contra os sinais fixos do
checkpoint.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
FLYVIS_DIR = HERE / "runs" / "flyvis"
SPEC = FLYVIS_DIR / "fib25-fib19_v2.2.json"
MODELS = FLYVIS_DIR / "pretrained" / "results" / "flow" / "0000"


def _strided(extent, su, sv):
    out = []
    for u in range(-extent, extent + 1):
        for v in range(max(-extent, -extent - u), min(extent, extent - u) + 1):
            if u % su == 0 and v % sv == 0:
                out.append((u, v))
    return out


def _fill_hull(offsets, n_syn_fill=1):
    import matplotlib.path as mp
    import scipy.spatial as ss
    known = np.array([o[0] for o in offsets])
    known_set = set(map(tuple, known))
    hull = ss.ConvexHull((1 + 1e-6) * known, False, "QJ")
    verts = known[hull.vertices]
    grid = np.concatenate(np.dstack(np.mgrid[known[:, 0].min(): known[:, 0].max() + 1,
                                             known[:, 1].min(): known[:, 1].max() + 1]))
    inside = grid[mp.Path(verts).contains_points(grid)]
    return offsets + [[[u, v], n_syn_fill] for u, v in inside if (u, v) not in known_set]


def edge_order(spec, extent=15):
    """Ordem dos pares (pre, pos) e das chaves (pre, pos, du, dv) no flyvis."""
    pos = {}
    for n in spec["nodes"]:
        pattern, args = n["pattern"]
        if pattern == "stride":
            pts = _strided(extent, *args)
        elif pattern == "tile":
            pts = _strided(extent, args, args)
        else:
            pts = _strided(0, 1, 1)
        pos[n["name"]] = pts
    pos_set = {t: set(p) for t, p in pos.items()}
    pairs, keys = [], []
    seen_p, seen_k = set(), set()
    for e in spec["edges"]:
        offs = _fill_hull(e["offsets"]) if len(e["offsets"]) >= 3 else e["offsets"]
        for (du, dv), _n in offs:
            for (u, v) in pos[e["src"]]:
                if (u + du, v + dv) in pos_set[e["tar"]]:
                    p = (e["src"], e["tar"])
                    k = (e["src"], e["tar"], du, dv)
                    if p not in seen_p:
                        seen_p.add(p)
                        pairs.append(p)
                    if k not in seen_k:
                        seen_k.add(k)
                        keys.append(k)
    return pairs, keys


@dataclass
class FlyvisParams:
    types: list            # 65 tipos celulares
    tau: dict              # tipo -> constante de tempo (s)
    bias: dict             # tipo -> potencial de repouso
    sign: dict             # (pre, pos) -> +1/-1
    strength: dict         # (pre, pos) -> forca por sinapse
    model: str
    n_syn: dict = None     # (pre, pos) -> sinapses por neuronio pos (soma sobre colunas)


def load(model="000"):
    sys.path.insert(0, str(HERE / "scripts"))
    import read_torch_checkpoint as rtc
    spec = json.loads(SPEC.read_text())
    net = rtc.load(MODELS / model / "best_chkpt")["network"]
    types = [n["name"] for n in spec["nodes"]]
    pairs, keys = edge_order(spec)
    if len(types) != len(net["nodes_bias"]) or len(pairs) != len(net["edges_sign"]) \
            or len(keys) != len(net["edges_syn_count"]):
        raise ValueError(f"ordem nao confere: {len(types)}/{len(pairs)}/{len(keys)}")
    alpha = {(e["src"], e["tar"]): e["alpha"] for e in spec["edges"]}
    if not np.array_equal(np.array([alpha[p] for p in pairs], dtype=np.float32), net["edges_sign"]):
        raise ValueError("sinais do checkpoint nao conferem com a ordem reconstruida")
    n_syn = {}
    for (s_, t_, _du, _dv), c in zip(keys, net["edges_syn_count"]):
        n_syn[(s_, t_)] = n_syn.get((s_, t_), 0.0) + float(c)
    return FlyvisParams(
        n_syn=n_syn,
        types=types,
        tau={t: float(x) for t, x in zip(types, net["nodes_time_const"])},
        bias={t: float(x) for t, x in zip(types, net["nodes_bias"])},
        sign={p: float(s) for p, s in zip(pairs, net["edges_sign"])},
        strength={p: float(s) for p, s in zip(pairs, net["edges_syn_strength"])},
        model=model,
    )
