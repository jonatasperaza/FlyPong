#!/usr/bin/env python3
"""Achado 24: o que a regra de recompensa mudou nos pesos LC -> descendente.

Uso: python scripts/diagnostics/learned_weights.py <checkpoint final> <checkpoint pos-desenvolvimento> [saques]

O segundo checkpoint (fase "train", ex.: o de exploration_signal.py) da os
pesos do inicio do treino; o `w0` do checkpoint final e de antes do
desenvolvimento.

O checkpoint precisa estar na fase "final" (run_condition grava a rede
treinada nele). Mede, com a rede congelada e ruido 6:
  - seletividade de cada LC pre-sinaptico ao lado da bola (taxa com a bola
    abaixo do paddle menos taxa com a bola acima, em spikes/frame);
  - para cada descendente: mudanca total de peso, e correlacao entre a
    mudanca de peso de cada sinapse e a seletividade do LC de origem;
  - correlacao da contagem de cada descendente com o lado da bola, antes
    (pesos iniciais) e depois do treino.
"""
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
import serve_eval as se
import serve_train as st
import sim.network as netmod

# A regra de plasticidade original (sim/network.py) sobrescreve os pesos
# plasticos sempre que ha dopamina; train() e evaluate_network() a desligam.
netmod.PLASTIC_LR = 0.0
from game.pong import PADDLE_H, HEIGHT, WIDTH, PongGame
from game.sensory_map import ball_to_photoreceptor_stimulus as bs

ck = st._load_checkpoint(sys.argv[1])
dev = st._load_checkpoint(sys.argv[2])
n_serves = int(sys.argv[3]) if len(sys.argv) > 3 else 40
assert ck["phase"] == "final", ck["phase"]
assert dev["phase"] == "train" and dev["loop_state"] is None, dev["phase"]
r = ck["runner"]
w0 = dev["runner"].net.W.data[dev["runner"].net.plastic_data_idx].copy()
del dev
net = r.net
w1 = net.W.data[net.plastic_data_idx].copy()
desc = net.descending_idx
names = net.neurons_df["instance"].astype(str).to_numpy()


def play(weights):
    net.W.data[net.plastic_data_idx] = weights
    rng = random.Random(11)
    nrng = np.random.default_rng(12)
    g = PongGame(seed=13)
    ext = np.zeros(net.n)
    acc = np.zeros(net.n)
    X, Y = [], []
    for _ in range(n_serves):
        se.serve(g, rng)
        net.reset()
        for _ in range(se.MAX_FRAMES_PER_TRIAL):
            stim = bs(r.perceived_ball_y(g), g.ball_x, HEIGHT, WIDTH, net.photoreceptor_positions)
            acc.fill(0)
            for _ in range(r.substeps):
                ext.fill(0)
                ext[net.photoreceptor_idx] += stim
                ext[desc] += nrng.normal(0, 6.0, len(desc))
                np.add(acc, net.step(ext), out=acc)
            rel = g.ball_y - (g.paddle_left_y + PADDLE_H / 2)
            if abs(rel) > 15:
                X.append(acc.copy())
                Y.append(1.0 if rel > 0 else -1.0)
            # politica fixa (oraculo com 30% de acoes aleatorias): a mesma
            # sequencia de estados para os dois conjuntos de pesos.
            ev = g.step(se.oracle_policy(g) if rng.random() < 0.7 else rng.choice([-1, 0, 1]))
            if ev["bounce"] or ev["score"] == "right":
                break
    return np.array(X), np.array(Y)


X0, Y0 = play(w0)
X1, Y1 = play(w1)
net.W.data[net.plastic_data_idx] = w1
below, above = Y0 > 0, Y0 < 0
sel = X0[below].mean(0) - X0[above].mean(0)
pre, post = net.plastic_pre, net.plastic_post
dw = w1 - w0
print(f"{sys.argv[1]}: {len(Y0)} frames; mudanca media |dw|={np.abs(dw).mean():.3f} (peso medio {w0.mean():.3f})")
for dn in desc:
    m = post == dn
    c = np.corrcoef(dw[m], sel[pre[m]])[0, 1] if m.sum() > 2 and dw[m].std() > 0 else float("nan")
    c0 = np.corrcoef(X0[:, dn], Y0)[0, 1] if X0[:, dn].std() > 0 else float("nan")
    c1 = np.corrcoef(X1[:, dn], Y1)[0, 1] if X1[:, dn].std() > 0 else float("nan")
    print(f"  {names[dn]:12s} grupo={'sobe' if dn in net.motor_up_idx else 'desce'} n_sinapses={m.sum():3d} "
          f"soma w0={w0[m].sum():7.1f} soma dw={dw[m].sum():+7.1f} corr(dw, seletividade)={c:+.3f} "
          f"taxa={X0[:, dn].mean():.2f}->{X1[:, dn].mean():.2f} corr(contagem, bola abaixo)={c0:+.3f}->{c1:+.3f}")
lc = np.unique(pre)
print(f"  seletividade dos LC pre-sinapticos: media |sel|={np.abs(sel[lc]).mean():.3f} "
      f"taxa media={X0[:, lc].mean():.3f}/frame; fracao com |sel|>0,1: {(np.abs(sel[lc]) > 0.1).mean():.2f}")
print(f"  LC que preferem bola abaixo (sel>0,05): {(sel[lc] > 0.05).mean():.2f}; "
      f"acima (sel<-0,05): {(sel[lc] < -0.05).mean():.2f}")
