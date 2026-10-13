#!/usr/bin/env python3
"""Achado 25: taxa por camada numa rede congelada (checkpoint), politica oraculo, ruido 6.

Uso: python scripts/diagnostics/layer_rates.py <checkpoint>"""
import sys, random, numpy as np
sys.path.insert(0, '/home/user/FlyPong')
import serve_eval as se, serve_train as st, sim.network as netmod
from game.pong import PongGame, HEIGHT, WIDTH
from game.sensory_map import ball_to_photoreceptor_stimulus as bs
netmod.PLASTIC_LR = 0.0
ck = st._load_checkpoint(sys.argv[1]); r = ck["runner"]; net = r.net
rng = random.Random(5); nrng = np.random.default_rng(6); g = PongGame(seed=7); ext = np.zeros(net.n)
tot = {k: 0.0 for k in net.role_idx}; act = {k: np.zeros(len(v)) for k, v in net.role_idx.items()}; nf = 0
for t in range(10):
    se.serve(g, rng); net.reset()
    for f in range(400):
        stim = bs(r.perceived_ball_y(g), g.ball_x, HEIGHT, WIDTH, net.photoreceptor_positions)
        for _ in range(r.substeps):
            ext.fill(0); ext[net.photoreceptor_idx] += stim; ext[net.descending_idx] += nrng.normal(0, 6, len(net.descending_idx))
            s = net.step(ext)
            for k, v in net.role_idx.items(): act[k] += s[v]
        nf += 1
        ev = g.step(se.oracle_policy(g))
        if ev["bounce"] or ev["score"] == "right": break
print(sys.argv[1].split('/')[-1], " | ".join(f"{k}: taxa={act[k].mean()/nf:.3f}/frame ativos={np.mean(act[k]>0):.2f}" for k in net.role_idx if k != 'dopaminergic'))
