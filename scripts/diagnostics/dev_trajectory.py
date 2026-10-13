#!/usr/bin/env python3
"""Achado 24: trajetoria da atividade durante a fase de desenvolvimento.

Uso: python scripts/diagnostics/dev_trajectory.py <seed de treino> [saques] [synapse_sign] [taxa alvo visual ou -] [dados] [recozimento]

Repete a fase de desenvolvimento da configuracao A (connectome_v09_in21,
all_visual, mesmos parametros e seeds de run_condition) e registra, a cada
saque, a taxa media (spikes/passo) dos interneuronios, dos LC e dos
descendentes, e a fracao de frames em "explosao" (LC acima de 0,1
spike/passo). Grava uma linha por saque em
runs/dev_trajectory/<seed>_<synapse_sign>.tsv.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
import serve_eval as se
import serve_train as st
from sim.covariance_rule import CovarianceRPELearner, HomeostaticScaler

seed = int(sys.argv[1])
n_serves = int(sys.argv[2]) if len(sys.argv) > 2 else 300
sign = sys.argv[3] if len(sys.argv) > 3 else "none"
vis_target = float(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] != "-" else None
data = sys.argv[5] if len(sys.argv) > 5 else "connectome_v09_in21"
anneal = int(sys.argv[6]) if len(sys.argv) > 6 else 0
d = st.DEFAULTS
r = se.build_runner(str(HERE / data), sensory_mode="egocentric",
                    retina_axis="elevation", synapse_sign=sign)
net = r.net
learner = CovarianceRPELearner(net, eta=0.0, elig_decay=d["elig_decay"], w_cap_factor=d["w_cap_factor"],
                               homeostasis_rate=d["homeostasis_rate"], target_rate=d["target_rate"])
vis = np.concatenate([net.role_idx[k] for k in ("interneuron", "motion", "object", "target")])
scaler = HomeostaticScaler(net, vis, rate=d["homeostasis_rate"],
                           target_rate=d["target_rate"] if vis_target is None else vis_target)
lc = np.concatenate([net.object_idx, net.target_idx])
inter = net.role_idx["interneuron"]
desc = net.descending_idx
tag = ("" if vis_target is None else f"_vt{vis_target:g}") + ("" if data == "connectome_v09_in21" else f"_{data}") + (f"_anneal{anneal}" if anneal else "")
out = HERE / "runs" / "dev_trajectory" / f"{seed}_{sign}{tag}.tsv"
out.parent.mkdir(parents=True, exist_ok=True)
f = open(out, "w")
f.write("saque\tinterneuronio\tLC\tDN\tfrac_explosao\tframes\n")
acc = {"i": [], "l": [], "d": []}
state = {"k": 0}


def on_frame(counts):
    scaler.update(counts)
    acc["i"].append(counts[inter].mean() / r.substeps)
    acc["l"].append(counts[lc].mean() / r.substeps)
    acc["d"].append(counts[desc].mean() / r.substeps)


def flush():
    if acc["l"]:
        l = np.array(acc["l"])
        f.write(f"{state['k']}\t{np.mean(acc['i']):.4f}\t{l.mean():.4f}\t{np.mean(acc['d']):.4f}\t"
                f"{np.mean(l > 0.1):.3f}\t{len(l)}\n")
        f.flush()
    for v in acc.values():
        v.clear()


def set_rate(k):
    left = n_serves - k
    f = 1.0 if not anneal or left > anneal else left / anneal
    learner.homeostasis_rate = d["homeostasis_rate"] * f
    scaler.rate = d["homeostasis_rate"] * f


def at_serve_start(loop_state):
    # train() chama o checkpoint no inicio de cada saque k > 0: as contagens
    # acumuladas sao as do saque k - 1.
    state["k"] = loop_state["k"] - 1
    flush()


st.train(r, learner, n_serves=n_serves, train_seed=seed + st.DEV_SEED_OFFSET,
         noise_std=d["noise_std"], reward_mode="dense", on_frame=on_frame,
         checkpoint=at_serve_start, checkpoint_every=1, on_serve=set_rate)
state["k"] = n_serves - 1
flush()
f.close()
print(out)
