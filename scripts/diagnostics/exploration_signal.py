#!/usr/bin/env python3
"""Achado 24: a exploracao gera sinal de aprendizado depois do desenvolvimento?

Uso: python scripts/diagnostics/exploration_signal.py <seed de treino> [saques] [ruido] [recozimento] [alvo visual]

Faz a fase de desenvolvimento da configuracao A (connectome_v09_in21,
all_visual, 300 saques, ruido independente) e depois joga `saques` saques
com a propria rede, pesos congelados, como no treino. Por frame mede:
  - d = (desce - sobe) - baseline, a variavel de decisao do readout;
  - x = ruido somado no frame, projetado na direcao do readout
    (media do ruido nos "desce" - media nos "sobe");
  - a acao e a recompensa densa.
Relata: fracao de frames com |d| < limiar (acao "parado"), correlacao(x, d)
(quanto o ruido move a decisao), correlacao(x, recompensa) (o gradiente que
a perturbacao de no consegue estimar) e o lado da bola contra d.
"""
import sys
import random
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
from game.motor_read import read_motor_action
from game.pong import PADDLE_H, HEIGHT, WIDTH
from game.sensory_map import ball_to_photoreceptor_stimulus as bs

seed = int(sys.argv[1])
n_serves = int(sys.argv[2]) if len(sys.argv) > 2 else 60
noise = float(sys.argv[3]) if len(sys.argv) > 3 else 6.0
anneal = int(sys.argv[4]) if len(sys.argv) > 4 else 0
vis_target = float(sys.argv[5]) if len(sys.argv) > 5 else None
ck = HERE / "runs" / "checkpoints" / "exploration_signal" / (f"dev{seed}" + (f"_anneal{anneal}" if anneal else "") + (f"_vt{vis_target:g}" if vis_target else "") + ".pkl")
dev = st._load_checkpoint(str(ck))
if dev is None or dev.get("phase") != "train":
    # So a fase de desenvolvimento (n_serves=0): run_condition salva o
    # checkpoint no inicio da fase de treino, com o runner ja desenvolvido.
    st.run_condition(str(HERE / "connectome_v09_in21"), condition="normal", policy="trained",
                     train_seed=seed, eval_seed=2001, eval_trials=1, n_serves=0,
                     verbose=False, retina_axis="elevation", homeostasis_scope="all_visual",
                     checkpoint_path=str(ck), dev_anneal_serves=anneal,
                     visual_target_rate=vis_target)
    dev = st._load_checkpoint(str(ck))
r = dev["runner"]
net = r.net
up, down = net.motor_up_idx, net.motor_down_idx
rng = random.Random(seed + 123)
nrng = np.random.default_rng(seed + 456)
from game.pong import PongGame
g = PongGame(seed=seed + 789)
ext = np.zeros(net.n)
acc = np.zeros(net.n)
nsum = np.zeros(net.n)
desc = net.descending_idx
lc = np.concatenate([net.object_idx, net.target_idx])
D, X, R, A, S, E = [], [], [], [], [], []
hits = 0
for k in range(n_serves):
    se.serve(g, rng)
    net.reset()
    for f in range(se.MAX_FRAMES_PER_TRIAL):
        stim = bs(r.perceived_ball_y(g), g.ball_x, HEIGHT, WIDTH, net.photoreceptor_positions)
        acc.fill(0)
        nsum.fill(0)
        for _ in range(r.substeps):
            ext.fill(0)
            ext[net.photoreceptor_idx] += stim
            xi = nrng.normal(0, noise, len(desc))
            ext[desc] += xi
            nsum[desc] += xi
            np.add(acc, net.step(ext), out=acc)
        d = acc[down].mean() - acc[up].mean() - r.motor_baseline_diff
        a = read_motor_action(acc, up, down, baseline_diff=r.motor_baseline_diff)
        before = st._distance(g)
        rel = g.ball_y - (g.paddle_left_y + PADDLE_H / 2)
        ev = g.step(a)
        done = ev["bounce"] or ev["score"] == "right"
        rew = 0.0
        if not done:
            p = before - st._distance(g)
            if abs(p) > st.DENSE_DEADZONE:
                rew = 1.0 if p > 0 else -1.0
        D.append(d); X.append(nsum[down].mean() - nsum[up].mean()); R.append(rew); A.append(a)
        S.append(np.sign(rel) if abs(rel) > 15 else 0.0)
        E.append(acc[lc].mean() / r.substeps > 0.1)
        if done:
            hits += bool(ev["bounce"])
            break
D, X, R, A, S = map(np.array, (D, X, R, A, S))
m = S != 0
c = lambda a, b: float(np.corrcoef(a, b)[0, 1])
print(f"seed {seed} ruido {noise:g} recozimento {anneal} alvo visual {vis_target}: explosao={np.mean(E):.2f} frames={len(D)} acerto={hits / n_serves:.2f} | "
      f"d media={D.mean():+.3f} dp={D.std():.3f} | parado={np.mean(A == 0):.2f} "
      f"sobe={np.mean(A == -1):.2f} desce={np.mean(A == 1):.2f} | "
      f"corr(ruido,d)={c(X, D):+.3f} corr(ruido,recompensa)={c(X, R):+.3f} | "
      f"corr(lado da bola abaixo, d)={c(S[m], D[m]):+.3f}", flush=True)
