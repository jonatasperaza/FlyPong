"""A fiacao existente consegue expressar "acima vs abaixo" com os sinais do readout?

Classificador linear sobre as contagens por frame das entradas plasticas dos
descendentes: entradas do grupo "sobe" com sinal +, do grupo "desce" com
sinal -, pesos >= 0 (como sinapses). Comparado com o mesmo classificador sem
restricao. Nao mede a dinamica dos spikes, so se a informacao esta na
fiacao certa.

Uso: python scripts/diagnostics/wiring_info.py DATA_DIR {pca|elevation} {plastic|visual}
"""
import sys, random, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import serve_eval as se, serve_train as st, sim.network as netmod
from sim.covariance_rule import CovarianceRPELearner, HomeostaticScaler
from game.pong import PongGame, PADDLE_H, HEIGHT, WIDTH
from game.sensory_map import ball_to_photoreceptor_stimulus as bs
netmod.PLASTIC_LR = 0
data, axis, scope = sys.argv[1:4]
r = se.build_runner(data, sensory_mode="egocentric", retina_axis=axis); net = r.net
L = CovarianceRPELearner(net, eta=0.0, homeostasis_rate=0.01, target_rate=0.3)
on_frame = None
if scope == "visual":
    on_frame = HomeostaticScaler(net, np.concatenate([net.object_idx, net.target_idx]), rate=0.01, target_rate=0.3).update
st.train(r, L, n_serves=300, train_seed=42 + st.DEV_SEED_OFFSET, noise_std=6.0, reward_mode="dense", on_frame=on_frame)
rng = random.Random(77); g = PongGame(seed=0); ext = np.zeros(net.n); acc = np.zeros(net.n)
X = []; Y = []
for t in range(80):
    se.serve(g, rng); net.reset()
    for f in range(400):
        stim = bs(r.perceived_ball_y(g), g.ball_x, HEIGHT, WIDTH, net.photoreceptor_positions); acc.fill(0)
        for _ in range(r.substeps):
            ext.fill(0); ext[net.photoreceptor_idx] += stim; np.add(acc, net.step(ext), out=acc)
        rel = g.ball_y - (g.paddle_left_y + PADDLE_H / 2)
        if abs(rel) > 15: X.append(acc.copy()); Y.append(1.0 if rel < 0 else -1.0)  # +1 = deve subir
        ev = g.step(se.oracle_policy(g) if rng.random() < 0.7 else rng.choice([-1, 0, 1]))
        if ev["bounce"] or ev["score"] == "right": break
X = np.array(X); Y = np.array(Y); n = len(Y); h = n // 2
pre, post = net.plastic_pre, net.plastic_post
up, down = net.motor_up_idx, net.motor_down_idx

def fit(F, nonneg, iters=3000, lr=0.05, lam=1e-3):
    mu = F[:h].mean(0); sd = F[:h].std(0) + 1e-9
    Z = F / sd  # sem centrar: preserva o sinal das contagens (pesos >= 0 fazem sentido)
    w = np.zeros(Z.shape[1]); b = 0.0
    for _ in range(iters):  # regressao logistica por gradiente (projetado se nonneg)
        m = Y[:h] * (Z[:h] @ w + b)
        gcoef = -Y[:h] / (1 + np.exp(np.clip(m, -30, 30)))
        w -= lr * (Z[:h].T @ gcoef / h + lam * w); b -= lr * gcoef.mean()
        if nonneg: np.maximum(w, 0, out=w)
    return ((Z[h:] @ w + b > 0) == (Y[h:] > 0)).mean()

name = data.rstrip('/').split('/')[-1] if 'real' in data else 'synthetic'
print(f"[{name}] frames={n} maioria={max((Y>0).mean(), (Y<0).mean()):.3f}")
for cond in ("normal", "inverted"):
    U, D = (up, down) if cond == "normal" else (down, up)
    mU = np.isin(post, U); mD = np.isin(post, D)
    F = np.c_[X[:, pre[mU]], -X[:, pre[mD]]]   # sobe: +entradas do grupo sobe, -entradas do grupo desce
    print(f"  {cond:8s} fiacao com sinais do readout (pesos>=0): {fit(F, True):.3f}   mesmas entradas sem restricao: {fit(np.c_[X[:, pre[mU]], X[:, pre[mD]]], False):.3f}", flush=True)
