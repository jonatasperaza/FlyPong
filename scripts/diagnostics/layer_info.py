"""Decodificabilidade linear de "bola acima/abaixo" em cada camada.

Uso: python scripts/diagnostics/layer_info.py DATA_DIR {pca|elevation} N_SAQUES
"""
import sys,random,numpy as np; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import serve_eval as se, sim.network as netmod
from game.pong import PongGame, PADDLE_H, HEIGHT, WIDTH
from game.sensory_map import ball_to_photoreceptor_stimulus as bs
netmod.PLASTIC_LR=0
data, axis, nserves = sys.argv[1], sys.argv[2], int(sys.argv[3])
r=se.build_runner(data, sensory_mode="egocentric", retina_axis=axis); net=r.net
rng=random.Random(3); g=PongGame(seed=0); ext=np.zeros(net.n); acc=np.zeros(net.n)
layers=["photoreceptor","interneuron","motion","object","target","descending"]
F={l:[] for l in layers}; Y=[]
for t in range(nserves):
    se.serve(g,rng); net.reset()
    for f in range(400):
        stim=bs(r.perceived_ball_y(g),g.ball_x,HEIGHT,WIDTH,net.photoreceptor_positions); acc.fill(0)
        for _ in range(4):
            ext.fill(0); ext[net.photoreceptor_idx]+=stim; acc+=net.step(ext)
        rel=g.ball_y-(g.paddle_left_y+PADDLE_H/2)
        if abs(rel)>15:
            for l in layers: F[l].append(acc[net.role_idx[l]].copy())
            Y.append(1 if rel>0 else 0)
        ev=g.step(0)
        if ev["bounce"] or ev["score"]=="right": break
Y=np.array(Y); n=len(Y); h=n//2
def acc_of(X):
    X=np.array(X,float); keep=X[:h].std(0)>0; X=X[:,keep]
    if X.shape[1]==0: return float('nan'), 0
    mu,sd=X[:h].mean(0),X[:h].std(0)+1e-9; X=(X-mu)/sd
    # reduz dimensao com PCA do treino (max 50) para nao sobreajustar
    U,S,Vt=np.linalg.svd(X[:h],full_matrices=False); k=min(50,Vt.shape[0]); Z=X@Vt[:k].T
    A=np.c_[Z,np.ones(n)]; w=np.linalg.solve(A[:h].T@A[:h]+1.0*np.eye(k+1), A[:h].T@(2*Y[:h]-1))
    return ((A[h:]@w>0)==Y[h:]).mean(), keep.sum()
print(f"[{axis}] {data.split('/')[-2]} frames={n} maioria={max(Y.mean(),1-Y.mean()):.3f}")
for l in layers:
    a,k=acc_of(F[l]); print(f"  {l:14s} ativos={k:5d} acuracia={a:.3f}", flush=True)
