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
import pickle
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
          curve_noise=0.0, log=None, on_frame=None,
          resume=None, checkpoint=None, checkpoint_every=50,
          noise_mode="independent"):
    """Treina `learner` em `n_serves` saques. Devolve a curva de avaliacao
    (lista de dicts) se `curve_every` > 0.

    `checkpoint(state)`, se dado, e chamado no inicio de cada saque multiplo
    de `checkpoint_every` (exceto o primeiro) com o estado do laco: indice do
    saque, geradores aleatorios, jogo, curva parcial e acertos. Passar esse
    dict em `resume` continua o laco exatamente do mesmo ponto: o resultado e
    identico ao de uma execucao sem interrupcao (o estado da rede e do
    aprendiz e salvo junto, por quem chama).

    `noise_mode`: "independent" (padrao) sorteia o ruido de exploracao de
    cada descendente independentemente. "antisymmetric" sorteia um valor por
    par direita/esquerda do mesmo tipo e injeta +xi no lado direito e -xi no
    esquerdo: a exploracao fica toda na diferenca direita - esquerda, que e o
    que o readout le (Achado 23).
    """
    if noise_mode not in NOISE_MODES:
        raise ValueError(f"unknown noise_mode: {noise_mode}")
    net = runner.net
    if resume is None:
        rng = random.Random(train_seed)
        noise_rng = np.random.default_rng(train_seed + 7919)
        game = PongGame(seed=train_seed)
        curve = []
        hits = 0
        k0 = 0
    else:
        rng, noise_rng, game = resume["rng"], resume["noise_rng"], resume["game"]
        curve, hits, k0 = resume["curve"], resume["hits"], resume["k"]
    external = np.zeros(net.n, dtype=np.float64)
    accum = np.zeros(net.n, dtype=np.float64)
    noise_sum = np.zeros(net.n, dtype=np.float64)
    desc = net.descending_idx
    if noise_mode == "antisymmetric":
        pair_r, pair_l = antisymmetric_pairs(net)
    saved_lr = netmod.PLASTIC_LR
    netmod.PLASTIC_LR = 0.0  # a regra original fica desligada
    try:
        for k in range(k0, n_serves):
            if checkpoint is not None and k > k0 and k % checkpoint_every == 0:
                checkpoint({"k": k, "rng": rng, "noise_rng": noise_rng, "game": game,
                            "curve": curve, "hits": hits})
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
                    if noise_std > 0 and noise_mode == "antisymmetric":
                        xi = noise_rng.normal(0.0, noise_std, len(pair_r))
                        external[pair_r] += xi
                        external[pair_l] -= xi
                        noise_sum[pair_r] += xi
                        noise_sum[pair_l] -= xi
                    elif noise_std > 0:
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


NOISE_MODES = ("independent", "antisymmetric")


def antisymmetric_pairs(net):
    """Pares (direita, esquerda) de descendentes do mesmo tipo, nos grupos do
    readout (motor_up_idx = lado R, motor_down_idx = lado L). Exige que cada
    tipo tenha o mesmo numero de neuronios nos dois lados."""
    types = net.neurons_df["type"].astype(str).to_numpy()
    right, left = [], []
    for t in sorted({types[i] for i in net.motor_up_idx}):
        r = [i for i in net.motor_up_idx if types[i] == t]
        l = [i for i in net.motor_down_idx if types[i] == t]
        if len(r) != len(l):
            raise ValueError(f"tipo {t}: {len(r)} descendente(s) R e {len(l)} L; sem pareamento")
        right += r
        left += l
    if len(right) != len(net.motor_up_idx) or len(left) != len(net.motor_down_idx):
        raise ValueError("descendentes sem par do mesmo tipo no outro lado")
    return np.array(right, dtype=np.int64), np.array(left, dtype=np.int64)


def calibrate_baseline(runner, noise_std, *, frames=400, seed=0):
    """Zero do readout: media de (desce - sobe) com a bola parada no centro e o
    ruido intrinseco `noise_std` nos descendentes. Nao usa recompensa nem
    direcao; e a mesma ideia de FlyPongRunner._calibrate_motor_baseline, mas
    com a rede no estado atual e com ruido."""
    net = runner.net
    rng = np.random.default_rng(seed + 31337)
    stim = ball_to_photoreceptor_stimulus(HEIGHT / 2, WIDTH / 2, HEIGHT, WIDTH,
                                          net.photoreceptor_positions)
    external = np.zeros(net.n, dtype=np.float64)
    accum = np.zeros(net.n, dtype=np.float64)
    desc = net.descending_idx
    diffs = []
    saved_lr = netmod.PLASTIC_LR
    netmod.PLASTIC_LR = 0.0
    try:
        net.reset()
        for _ in range(frames):
            accum.fill(0.0)
            for _ in range(runner.substeps):
                external.fill(0.0)
                external[net.photoreceptor_idx] += stim
                if noise_std > 0:
                    external[desc] += rng.normal(0.0, noise_std, len(desc))
                np.add(accum, net.step(external), out=accum)
            diffs.append(accum[net.motor_down_idx].mean() - accum[net.motor_up_idx].mean())
    finally:
        netmod.PLASTIC_LR = saved_lr
        net.reset()
    return float(np.mean(diffs))


def _save_checkpoint(path, obj):
    """Grava `obj` de forma atomica (arquivo temporario + rename)."""
    path = os.fspath(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def _load_checkpoint(path):
    if path is None or not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


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
                  retina_axis="pca", checkpoint_path=None, checkpoint_every=50,
                  dev_seed=None, recalibrate_after_dev=False,
                  visual_target_rate=None, noise_mode="independent",
                  synapse_sign="none"):
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

    `visual_target_rate`: taxa alvo (spikes/frame) da homeostase das camadas
    visuais (HomeostaticScaler). Padrao: a mesma `target_rate` dos
    descendentes. Diagnostico pos-Achado 22: a rede que aprende (seed 1) saiu
    do desenvolvimento com os LC esparsos (~0,04 spike/frame, 40-60%
    silenciosos), as que nao aprendem com LC densos perto de 0,3.

    `recalibrate_after_dev`: ao fim da fase de desenvolvimento, recalcula o
    zero do readout (`motor_baseline_diff`) com a rede ja escalada e o mesmo
    ruido intrinseco da avaliacao (bola parada no centro, sem recompensa).
    A calibracao original e feita antes da homeostase, com a rede saturada.

    `dev_seed`: seed da fase de desenvolvimento. Padrao: train_seed +
    DEV_SEED_OFFSET (como em todos os Achados). Passar outro valor separa o
    estado em que a rede sai do desenvolvimento da sequencia de treino.

    `noise_mode`: estrutura do ruido de exploracao so na fase de TREINO (ver
    `train`). O desenvolvimento e a avaliacao usam sempre ruido independente,
    entao o estado pos-desenvolvimento nao muda.

    `synapse_sign`: "none" (todas as sinapses excitatorias, como ate o Achado
    23) ou "nt" (GABA e glutamato inibitorios; ver sim/network.py).
    """
    if dev_seed is None:
        dev_seed = train_seed + DEV_SEED_OFFSET
    if policy not in POLICIES:
        raise ValueError(f"unknown policy: {policy}")
    if homeostasis_scope not in HOMEOSTASIS_SCOPES:
        raise ValueError(f"unknown homeostasis_scope: {homeostasis_scope}")
    t0 = time.perf_counter()
    ck = _load_checkpoint(checkpoint_path)
    if ck is None:
        runner = se.build_runner(data_dir, sensory_mode="egocentric",
                                 retina_axis=retina_axis, synapse_sign=synapse_sign)
        if condition == "inverted":
            se.invert_readout(runner)
        learner = CovarianceRPELearner(
            runner.net, eta=0.0, elig_decay=elig_decay, w_cap_factor=w_cap_factor,
            homeostasis_rate=homeostasis_rate, target_rate=target_rate,
        )
        w0 = runner.net.W.data[runner.net.plastic_data_idx].copy()
        scaler = None
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
            scaler = HomeostaticScaler(
                net, scaled, rate=homeostasis_rate,
                target_rate=target_rate if visual_target_rate is None else visual_target_rate)
        phase, loop_state, tr, elapsed_before = "dev", None, {"curve": [], "train_hit_rate": None}, 0.0
    else:
        runner, learner, scaler, w0 = ck["runner"], ck["learner"], ck["scaler"], ck["w0"]
        phase, loop_state, tr, elapsed_before = ck["phase"], ck["loop_state"], ck["tr"], ck["elapsed"]
        if verbose:
            print(f"  retomando do checkpoint: fase={phase} saque={loop_state['k'] if loop_state else 0}")

    def save(phase_now, state):
        if checkpoint_path is None:
            return
        _save_checkpoint(checkpoint_path, {
            "runner": runner, "learner": learner, "scaler": scaler, "w0": w0,
            "phase": phase_now, "loop_state": state, "tr": tr,
            "elapsed": elapsed_before + time.perf_counter() - t0,
        })

    on_frame = scaler.update if scaler is not None else None
    if phase == "dev":
        if policy != "control_innate" and dev_serves > 0:
            train(runner, learner, n_serves=dev_serves, on_frame=on_frame,
                  train_seed=dev_seed, noise_std=noise_std,
                  # eta = 0 aqui, entao a recompensa nao importa; "dense" so
                  # garante que a homeostase atualiza a cada frame.
                  reward_mode="dense",
                  resume=loop_state, checkpoint_every=checkpoint_every,
                  checkpoint=lambda st: save("dev", st))
        learner.homeostasis_rate = 0.0
        if recalibrate_after_dev and policy != "control_innate":
            runner.motor_baseline_diff = calibrate_baseline(runner, noise_std, seed=dev_seed)
        if policy == "trained":
            learner.eta = eta
        phase, loop_state = "train", None
        save("train", None)
    if phase == "train":
        if policy == "trained":
            tr = train(runner, learner, n_serves=n_serves, train_seed=train_seed,
                       noise_std=noise_std, reward_mode=reward_mode, elig_mode=elig_mode,
                       curve_every=curve_every,
                       curve_seed=eval_seed if curve_every else None,
                       curve_trials=curve_trials, curve_noise=noise_std,
                       log=print if verbose else None, noise_mode=noise_mode,
                       resume=loop_state, checkpoint_every=checkpoint_every,
                       checkpoint=lambda st: save("train", st))
        phase = "final"
    final = se.evaluate_network(runner, eval_trials, eval_seed, noise_std)
    w1 = runner.net.W.data[runner.net.plastic_data_idx]
    return {
        "policy": policy,
        "condition": condition,
        "train_seed": train_seed,
        "dev_seed": dev_seed,
        "motor_baseline_diff": runner.motor_baseline_diff,
        **final,
        "train_hit_rate": tr["train_hit_rate"],
        "curve": tr["curve"],
        "config": {"n_serves": n_serves, "eta": eta, "noise_std": noise_std,
                   "reward_mode": reward_mode, "elig_mode": elig_mode,
                   "elig_decay": elig_decay, "w_cap_factor": w_cap_factor,
                   "dev_serves": dev_serves, "homeostasis_rate": homeostasis_rate,
                   "target_rate": target_rate, "homeostasis_scope": homeostasis_scope,
                   "recalibrate_after_dev": recalibrate_after_dev,
                   "visual_target_rate": visual_target_rate,
                   "noise_mode": noise_mode,
                   "synapse_sign": synapse_sign,
                   "sensory_mode": "egocentric",
                   "retina_axis": retina_axis},
        "weights": {"mean_before": float(w0.mean()), "mean_after": float(w1.mean()),
                    "frac_zero_after": float((w1 <= 1e-9).mean()),
                    "mean_abs_change": float(np.abs(w1 - w0).mean())},
        "elapsed_seconds": elapsed_before + time.perf_counter() - t0,
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
