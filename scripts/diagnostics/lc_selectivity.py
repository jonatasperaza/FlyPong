#!/usr/bin/env python3
"""Achado 24: seletividade dos LC ao lado da bola num estado pos-desenvolvimento.

Uso: python scripts/diagnostics/lc_selectivity.py <checkpoint fase "train"> [saques]

Com a rede congelada, ruido 6 nos descendentes e politica fixa (oraculo com
30% de acoes aleatorias), mede para cada LC pre-sinaptico dos descendentes a
taxa com a bola abaixo do paddle menos a taxa com a bola acima
(spikes/frame), e a fracao de frames em explosao (LC > 0,1 spike/passo).
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
from game.pong import PADDLE_H, HEIGHT, WIDTH, PongGame
from game.sensory_map import ball_to_photoreceptor_stimulus as bs

netmod.PLASTIC_LR = 0.0
ck = st._load_checkpoint(sys.argv[1])
n_serves = int(sys.argv[2]) if len(sys.argv) > 2 else 40
r = ck["runner"]
net = r.net
desc = net.descending_idx
lc = np.unique(net.plastic_pre)
rng = random.Random(11)
nrng = np.random.default_rng(12)
g = PongGame(seed=13)
ext = np.zeros(net.n)
acc = np.zeros(net.n)
X, Y, E = [], [], []
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
        E.append(acc[lc].mean() / r.substeps > 0.1)
        rel = g.ball_y - (g.paddle_left_y + PADDLE_H / 2)
        if abs(rel) > 15:
            X.append(acc[lc].copy())
            Y.append(1.0 if rel > 0 else -1.0)
        ev = g.step(se.oracle_policy(g) if rng.random() < 0.7 else rng.choice([-1, 0, 1]))
        if ev["bounce"] or ev["score"] == "right":
            break
X, Y = np.array(X), np.array(Y)
sel = X[Y > 0].mean(0) - X[Y < 0].mean(0)
print(f"{Path(sys.argv[1]).stem}: explosao={np.mean(E):.2f} taxa LC={X.mean():.3f}/frame | "
      f"preferem abaixo={np.mean(sel > 0.05):.2f} acima={np.mean(sel < -0.05):.2f} "
      f"| sel media={sel.mean():+.3f}")
