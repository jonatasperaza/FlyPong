#!/usr/bin/env python3
"""Achado 23: estado da rede ao fim da fase de desenvolvimento (sem treino).

Uso: python scripts/diagnostics/dev_state.py <seed de treino>
Mede taxa, fracao de silenciosos e decodificacao linear (acima/abaixo do
paddle) em LC e descendentes, e a correlacao de cada descendente com o lado
da bola. Usa connectome_v09_in21, homeostase all_visual, 300 saques."""
import sys, random, numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
import serve_eval as se, serve_train as st, sim.network as netmod
from sim.covariance_rule import CovarianceRPELearner, HomeostaticScaler
from game.pong import PongGame, PADDLE_H, HEIGHT, WIDTH
from game.sensory_map import ball_to_photoreceptor_stimulus as bs
netmod.PLASTIC_LR = 0
seed = int(sys.argv[1]); data = str(HERE / 'connectome_v09_in21')
r = se.build_runner(data, sensory_mode="egocentric", retina_axis="elevation"); net = r.net
L = CovarianceRPELearner(net, eta=0.0, homeostasis_rate=0.01, target_rate=0.3)
vis = np.concatenate([net.role_idx[k] for k in ("interneuron", "motion", "object", "target")])
sc = HomeostaticScaler(net, vis, rate=0.01, target_rate=0.3)
st.train(r, L, n_serves=300, train_seed=seed + st.DEV_SEED_OFFSET, noise_std=6.0, reward_mode="dense", on_frame=sc.update)
rng = random.Random(77); nrng = np.random.default_rng(3); g = PongGame(seed=0); ext = np.zeros(net.n); acc = np.zeros(net.n); d = net.descending_idx
X = []; Y = []
for t in range(60):
    se.serve(g, rng); net.reset()
    for f in range(400):
        stim = bs(r.perceived_ball_y(g), g.ball_x, HEIGHT, WIDTH, net.photoreceptor_positions); acc.fill(0)
        for _ in range(4):
            ext.fill(0); ext[net.photoreceptor_idx] += stim; ext[d] += nrng.normal(0, 6.0, len(d)); np.add(acc, net.step(ext), out=acc)
        rel = g.ball_y - (g.paddle_left_y + PADDLE_H / 2)
        if abs(rel) > 15: X.append(acc.copy()); Y.append(1.0 if rel < 0 else -1.0)
        ev = g.step(se.oracle_policy(g) if rng.random() < 0.7 else rng.choice([-1, 0, 1]))
        if ev["bounce"] or ev["score"] == "right": break
X = np.array(X); Y = np.array(Y); n = len(Y); h = n // 2
def decode(idx):
    F = X[:, idx]; keep = F[:h].std(0) > 0; F = F[:, keep]
    if F.shape[1] == 0: return float('nan')
    mu, sd = F[:h].mean(0), F[:h].std(0) + 1e-9; Z = (F - mu) / sd
    U, S_, Vt = np.linalg.svd(Z[:h], full_matrices=False); k = min(50, Vt.shape[0]); P = Z @ Vt[:k].T
    A = np.c_[P, np.ones(n)]; w = np.linalg.solve(A[:h].T @ A[:h] + np.eye(k + 1), A[:h].T @ Y[:h])
    return ((A[h:] @ w > 0) == (Y[h:] > 0)).mean()
types = net.neurons_df["type"].astype(str).to_numpy(); rate = X.mean(0) / 4
pre, post = net.plastic_pre, net.plastic_post; w = net.W.data[net.plastic_data_idx]
out = [f"seed {seed}:"]
for name, idx in [("object", net.object_idx), ("LC10a", net.target_idx[[t.startswith("LC10a") for t in types[net.target_idx]]]),
                  ("LC10c", net.target_idx[[t.startswith("LC10c") for t in types[net.target_idx]]]), ("DN", d)]:
    out.append(f"  {name:6s} taxa media={rate[idx].mean():.3f} silenciosos={(rate[idx] < 0.005).mean():.2f} decod={decode(idx):.3f}")
for dn, nm in zip(d, net.neurons_df.instance.iloc[d]):
    m = post == dn
    out.append(f"  {nm:12s} peso plastico medio={w[m].mean():.3f} soma={w[m].sum():.1f} taxa={rate[dn]*4:.2f}/frame corr(contagem,acima)={np.corrcoef(X[:, dn], Y)[0,1]:+.3f}")
print("\n".join(out), flush=True)
