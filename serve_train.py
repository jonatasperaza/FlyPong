"""Treino por saques com a regra de covariancia + erro de predicao.

Cada tentativa de treino e um saque independente (mesmo protocolo de
serve_eval.py, mas com RNG de treino). A recompensa vem so do ambiente:

  - `dense`: a cada frame, +1 se o paddle ficou mais perto da altura da bola,
    -1 se ficou mais longe, 0 dentro de uma zona morta;
  - `sparse`: so no fim da tentativa, +1 acerto / -1 erro (use com
    `--elig-decay` > 0 para o credito chegar as acoes anteriores);
  - `both`: soma das duas.

A recompensa nunca e injetada como corrente na rede nem diz qual e a direcao
certa: entra so como escalar global na regra (sim/covariance_rule.py). A
exploracao vem de ruido gaussiano de corrente nos neuronios descendentes; o
mesmo ruido fica ligado na avaliacao (ruido intrinseco), igual para a rede
treinada e para os controles. A avaliacao e sempre com pesos congelados.

Sequencia do treino: (1) fase de desenvolvimento, so escalonamento
homeostatico e sem recompensa, que tira os descendentes da saturacao; (2)
treino com a regra de recompensa, homeostase desligada. Os controles sao a
rede inata e a rede so com a fase (1).

Uso:
    python serve_train.py --condition inverted --train-seed 42 --eval-seed 1001
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np

import main
import serve_eval as se
import sim.network as netmod
from sim.covariance_rule import CovarianceRPELearner, HomeostaticScaler
from game.pong import PongGame, HEIGHT, WIDTH, PADDLE_H
from game.sensory_map import ball_to_photoreceptor_stimulus
from game.motor_read import read_motor_action

# Seeds so para escolher hiperparametros; nunca usadas no resultado final
# (que usa se.EVAL_SEEDS).
VALIDATION_SEEDS = (2001, 2002, 2003)
DENSE_DEADZONE = 0.5


def _distance(game):
    return abs(game.ball_y - (game.paddle_left_y + PADDLE_H / 2))


def train(runner, learner, *, n_serves, train_seed, noise_std, reward_mode,
          elig_mode="covariance", curve_every=0, curve_seed=None, curve_trials=100,
          curve_noise=0.0, log=None, on_frame=None):
    """Treina `learner` em `n_serves` saques. Devolve a curva de avaliacao
    (lista de dicts) se `curve_every` > 0."""
    net = runner.net
    rng = random.Random(train_seed)
    noise_rng = np.random.default_rng(train_seed + 7919)
    game = PongGame(seed=train_seed)
    external = np.zeros(net.n, dtype=np.float64)
    accum = np.zeros(net.n, dtype=np.float64)
    noise_sum = np.zeros(net.n, dtype=np.float64)
    desc = net.descending_idx
    curve = []
    saved_lr = netmod.PLASTIC_LR
    netmod.PLASTIC_LR = 0.0  # a regra original fica desligada
    hits = 0
    try:
        for k in range(n_serves):
            if curve_every and k % curve_every == 0:
                curve.append({"serves": k, **se.evaluate_network(runner, curve_trials, curve_seed, curve_noise)})
                if log:
                    log(f"  serves={k:5d} acerto={curve[-1]['hit_rate']:.3f}")
            se.serve(game, rng)
            net.reset()
            learner.reset_traces()
            for _ in range(se.MAX_FRAMES_PER_TRIAL):
                stim = ball_to_photoreceptor_stimulus(
                    runner.perceived_ball_y(game), game.ball_x, HEIGHT, WIDTH,
                    net.photoreceptor_positions,
                )
                accum.fill(0.0)
                noise_sum.fill(0.0)
                for _ in range(runner.substeps):
                    external.fill(0.0)
                    external[net.photoreceptor_idx] += stim
                    if noise_std > 0:
                        xi = noise_rng.normal(0.0, noise_std, len(desc))
                        external[desc] += xi
                        noise_sum[desc] += xi
                    np.add(accum, net.step(external), out=accum)
                action = read_motor_action(
                    accum, net.motor_up_idx, net.motor_down_idx,
                    baseline_diff=runner.motor_baseline_diff,
                )
                d_before = _distance(game)
                event = game.step(action)
                done = event["bounce"] or event["score"] == "right"

                reward = 0.0
                if reward_mode in ("dense", "both") and not done:
                    progress = d_before - _distance(game)
                    if abs(progress) > DENSE_DEADZONE:
                        reward += 1.0 if progress > 0 else -1.0
                if reward_mode in ("sparse", "both") and done:
                    reward += 1.0 if event["bounce"] else -1.0
                learner.update(
                    accum, reward,
                    perturbation=noise_sum if elig_mode == "perturbation" else None,
                    apply=done or reward_mode != "sparse",
                )
                if on_frame is not None:
                    on_frame(accum)
                if done:
                    hits += bool(event["bounce"])
                    break
        if curve_every:
            curve.append({"serves": n_serves, **se.evaluate_network(runner, curve_trials, curve_seed, curve_noise)})
            if log:
                log(f"  serves={n_serves:5d} acerto={curve[-1]['hit_rate']:.3f}")
    finally:
        netmod.PLASTIC_LR = saved_lr
        net.reset()
    return {"curve": curve, "train_hit_rate": hits / max(1, n_serves)}


# Hiperparametros escolhidos so nas VALIDATION_SEEDS (varredura registrada no
# README, Achado 19), antes de qualquer rodada nas EVAL_SEEDS.
DEFAULTS = dict(
    n_serves=3000, eta=0.001, noise_std=6.0, reward_mode="dense",
    elig_mode="perturbation", elig_decay=0.0, w_cap_factor=4.0,
    dev_serves=300, homeostasis_rate=0.01, target_rate=0.3,
    homeostasis_scope="plastic",
)
DEV_SEED_OFFSET = 500_000

POLICIES = ("trained", "control_homeostasis", "control_innate")
HOMEOSTASIS_SCOPES = ("plastic", "visual", "all_visual")


def run_condition(data_dir, *, condition, policy, train_seed, eval_seed,
                  eval_trials=se.DEFAULT_TRIALS, n_serves=DEFAULTS["n_serves"],
                  eta=DEFAULTS["eta"], noise_std=DEFAULTS["noise_std"],
                  reward_mode=DEFAULTS["reward_mode"], elig_mode=DEFAULTS["elig_mode"],
                  elig_decay=DEFAULTS["elig_decay"], w_cap_factor=DEFAULTS["w_cap_factor"],
                  dev_serves=DEFAULTS["dev_serves"],
                  homeostasis_rate=DEFAULTS["homeostasis_rate"],
                  target_rate=DEFAULTS["target_rate"],
                  homeostasis_scope=DEFAULTS["homeostasis_scope"],
                  curve_every=0, curve_trials=100, verbose=True,
                  retina_axis="pca"):
    """Roda uma condicao do experimento.

    policy:
      - "trained": fase de desenvolvimento (so homeostase, sem recompensa)
        seguida de `n_serves` saques de treino com a regra de recompensa
        (homeostase desligada);
      - "control_homeostasis": a mesma fase de desenvolvimento, sem treino
        com recompensa (isola o efeito da homeostase);
      - "control_innate": a rede como carregada, sem nenhuma mudanca.
    Todas sao avaliadas igual: pesos congelados, mesmas seeds de saque e o
    mesmo ruido intrinseco `noise_std`.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown policy: {policy}")
    runner = se.build_runner(data_dir, sensory_mode="egocentric",
                             retina_axis=retina_axis)
    if condition == "inverted":
        se.invert_readout(runner)
    learner = CovarianceRPELearner(
        runner.net, eta=0.0, elig_decay=elig_decay, w_cap_factor=w_cap_factor,
        homeostasis_rate=homeostasis_rate, target_rate=target_rate,
    )
    w0 = runner.net.W.data[runner.net.plastic_data_idx].copy()
    t0 = time.perf_counter()
    tr = {"curve": [], "train_hit_rate": None}
    if homeostasis_scope not in HOMEOSTASIS_SCOPES:
        raise ValueError(f"unknown homeostasis_scope: {homeostasis_scope}")
    on_frame = None
    if homeostasis_scope != "plastic":
        net = runner.net
        if homeostasis_scope == "visual":
            # Tambem escala todas as entradas dos neuronios LC (object/target).
            scaled = np.concatenate([net.object_idx, net.target_idx])
        else:
            # "all_visual": todas as entradas de todo neuronio visual
            # (interneuronios, T4/T5, LC). Com o ganho sinaptico unico do
            # modelo, quase todo o lobo optico dispara na taxa maxima; em
            # saturacao a informacao de posicao se perde antes dos LC
            # (Achado 21).
            scaled = np.concatenate([net.role_idx.get(k, np.empty(0, dtype=np.int64))
                                     for k in ("interneuron", "motion", "object", "target")])
        on_frame = HomeostaticScaler(net, scaled, rate=homeostasis_rate,
                                     target_rate=target_rate).update
    if policy != "control_innate" and dev_serves > 0:
        train(runner, learner, n_serves=dev_serves, on_frame=on_frame,
              train_seed=train_seed + DEV_SEED_OFFSET, noise_std=noise_std,
              # eta = 0 aqui, entao a recompensa nao importa; "dense" so
              # garante que a homeostase atualiza a cada frame.
              reward_mode="dense")
    learner.homeostasis_rate = 0.0
    if policy == "trained":
        learner.eta = eta
        tr = train(runner, learner, n_serves=n_serves, train_seed=train_seed,
                   noise_std=noise_std, reward_mode=reward_mode, elig_mode=elig_mode,
                   curve_every=curve_every,
                   curve_seed=eval_seed if curve_every else None,
                   curve_trials=curve_trials, curve_noise=noise_std,
                   log=print if verbose else None)
    final = se.evaluate_network(runner, eval_trials, eval_seed, noise_std)
    w1 = runner.net.W.data[runner.net.plastic_data_idx]
    return {
        "policy": policy,
        "condition": condition,
        "train_seed": train_seed,
        **final,
        "train_hit_rate": tr["train_hit_rate"],
        "curve": tr["curve"],
        "config": {"n_serves": n_serves, "eta": eta, "noise_std": noise_std,
                   "reward_mode": reward_mode, "elig_mode": elig_mode,
                   "elig_decay": elig_decay, "w_cap_factor": w_cap_factor,
                   "dev_serves": dev_serves, "homeostasis_rate": homeostasis_rate,
                   "target_rate": target_rate, "homeostasis_scope": homeostasis_scope,
                   "sensory_mode": "egocentric",
                   "retina_axis": retina_axis},
        "weights": {"mean_before": float(w0.mean()), "mean_after": float(w1.mean()),
                    "frac_zero_after": float((w1 <= 1e-9).mean()),
                    "mean_abs_change": float(np.abs(w1 - w0).mean())},
        "elapsed_seconds": time.perf_counter() - t0,
    }


def main_cli():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=main.DATA_DIR)
    ap.add_argument("--condition", choices=["normal", "inverted"], default="inverted")
    ap.add_argument("--policy", choices=POLICIES, default="trained")
    ap.add_argument("--train-seed", type=int, default=se.TRAIN_SEEDS[0])
    ap.add_argument("--eval-seed", type=int, default=se.EVAL_SEEDS[0])
    ap.add_argument("--eval-trials", type=int, default=se.DEFAULT_TRIALS)
    ap.add_argument("--serves", type=int, default=DEFAULTS["n_serves"])
    ap.add_argument("--eta", type=float, default=DEFAULTS["eta"])
    ap.add_argument("--noise", type=float, default=DEFAULTS["noise_std"])
    ap.add_argument("--reward", choices=["dense", "sparse", "both"], default=DEFAULTS["reward_mode"])
    ap.add_argument("--elig-mode", choices=["perturbation", "covariance"], default=DEFAULTS["elig_mode"])
    ap.add_argument("--elig-decay", type=float, default=DEFAULTS["elig_decay"])
    ap.add_argument("--w-cap-factor", type=float, default=DEFAULTS["w_cap_factor"])
    ap.add_argument("--dev-serves", type=int, default=DEFAULTS["dev_serves"])
    ap.add_argument("--homeostasis-rate", type=float, default=DEFAULTS["homeostasis_rate"])
    ap.add_argument("--target-rate", type=float, default=DEFAULTS["target_rate"])
    ap.add_argument("--homeostasis-scope", choices=list(HOMEOSTASIS_SCOPES),
                    default=DEFAULTS["homeostasis_scope"],
                    help="visual: a fase de desenvolvimento tambem escala as entradas "
                         "dos neuronios LC (object/target); all_visual: de todo neuronio "
                         "visual")
    ap.add_argument("--retina-axis", choices=["pca", "elevation"], default="pca")
    ap.add_argument("--curve-every", type=int, default=0)
    ap.add_argument("--curve-trials", type=int, default=100)
    ap.add_argument("--out", default=None, help="acrescenta o resultado (JSONL)")
    args = ap.parse_args()

    row = run_condition(
        args.data_dir, condition=args.condition, policy=args.policy,
        train_seed=args.train_seed, eval_seed=args.eval_seed,
        eval_trials=args.eval_trials, n_serves=args.serves, eta=args.eta,
        noise_std=args.noise, reward_mode=args.reward, elig_mode=args.elig_mode,
        elig_decay=args.elig_decay, w_cap_factor=args.w_cap_factor,
        dev_serves=args.dev_serves, homeostasis_rate=args.homeostasis_rate,
        target_rate=args.target_rate, homeostasis_scope=args.homeostasis_scope,
        curve_every=args.curve_every,
        curve_trials=args.curve_trials, retina_axis=args.retina_axis,
    )
    print(f"{row['policy']}/{row['condition']} treino={row['train_seed']} "
          f"aval={row['eval_seed']} acerto={row['hit_rate']:.3f} "
          f"IC95=[{row['ci95_low']:.3f}, {row['ci95_high']:.3f}] "
          f"pesos {row['weights']['mean_before']:.2f}->{row['weights']['mean_after']:.2f} "
          f"({row['elapsed_seconds']:.0f}s)")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
