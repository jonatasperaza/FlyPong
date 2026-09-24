"""Avaliacao por saques independentes (protocolo B do briefing).

Por que existe: a taxa de rebatida numa partida continua (main.py,
`validation_report`) permite orbitas periodicas travadas -- a mesma
trajetoria se repete e a metrica mede sorte de trajetoria, nao
comportamento (seed 2 nos Achados 13, 17 e 18). Aqui cada tentativa e
independente:

  - a rede volta ao estado de repouso (`net.reset()`);
  - a bola sai do centro em direcao ao paddle do conectoma, com angulo
    sorteado por um RNG proprio da avaliacao;
  - o paddle comeca numa altura sorteada pelo mesmo RNG;
  - a tentativa termina no acerto (rebatida) ou no erro (bola passa).

A avaliacao roda com pesos congelados (PLASTIC_LR = 0 durante a medicao,
restaurado no fim) e sem nenhum pulso de reforco: a corrente externa e so o
estimulo dos fotorreceptores. Reporta taxa de acerto com intervalo de
confianca de Wilson 95%.

Condicao "inverted" (teste C): troca os grupos "sobe" e "desce" do readout e
inverte o sinal de `motor_baseline_diff`, entao o circuito inato passa a
jogar para o lado errado.

Limitacao medida (grafo sintetico, 6 seeds x 300 saques): o estimulo de
`ball_to_photoreceptor_stimulus` so codifica a bola, nunca o paddle. Com o
paddle em altura sorteada, nenhuma politica cega ao paddle passa de
~PADDLE_H / (HEIGHT - PADDLE_H) ~ 23% de acerto -- o mesmo nivel do paddle
parado. A rede inata normal fica nesse nivel (24,2% vs 24,7% parado); a
invertida fica abaixo (15,8%). Enquanto a entrada sensorial nao informar a
posicao do paddle, nenhuma regra de aprendizado pode superar o controle
neste protocolo.

Uso:
    python serve_eval.py --condition both --trials 300
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from datetime import datetime, timezone

import numpy as np

import main
import sim.network as netmod
from game.pong import PongGame, WIDTH, HEIGHT, PADDLE_H, PADDLE_W, BALL_SPEED
from game.sensory_map import ball_to_photoreceptor_stimulus
from game.motor_read import read_motor_action

# Seeds definidas ANTES de rodar qualquer experimento. Treino usa as seeds da
# fisica do Pong ja usadas nos Achados anteriores; avaliacao usa um conjunto
# disjunto, que nunca deve ser usado para treinar nem para escolher
# hiperparametros.
TRAIN_SEEDS = (42, 1, 2, 3, 4, 5)
EVAL_SEEDS = (1001, 1002, 1003, 1004, 1005, 1006)

SERVE_ANGLE_MAX = 0.4  # mesmo intervalo de PongGame.reset_ball
DEFAULT_TRIALS = 300
# A bola leva ~(WIDTH / 2) / BALL_SPEED ~ 69 frames ate o paddle; o limite so
# protege contra um laco infinito se a fisica mudar.
MAX_FRAMES_PER_TRIAL = 400


def wilson_interval(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalo de confianca de Wilson para uma proporcao binomial."""
    if n <= 0:
        return (0.0, 1.0)
    p = hits / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def serve(game: PongGame, rng: random.Random, angle_max: float = SERVE_ANGLE_MAX):
    """Posiciona um saque independente: bola no centro indo para a esquerda,
    angulo e altura do paddle sorteados por `rng` (nunca pelo RNG do jogo)."""
    angle = rng.uniform(-angle_max, angle_max)
    game.ball_x = WIDTH / 2
    game.ball_y = HEIGHT / 2
    game.ball_vx = -BALL_SPEED
    game.ball_vy = BALL_SPEED * angle
    game.paddle_left_y = rng.uniform(0.0, HEIGHT - PADDLE_H)
    game.paddle_right_y = HEIGHT / 2 - PADDLE_H / 2


def play_trial(game: PongGame, act, max_frames: int = MAX_FRAMES_PER_TRIAL) -> bool:
    """Joga um saque ate acerto (True) ou erro (False). `act(game)` devolve
    -1/0/1 com a mesma convencao de PongGame.step."""
    for _ in range(max_frames):
        event = game.step(act(game))
        if event["bounce"]:
            return True
        if event["score"] == "right":
            return False
    return False


def run_serves(act, n_trials: int, eval_seed: int, *, on_trial_start=None,
               angle_max: float = SERVE_ANGLE_MAX) -> dict:
    """Roda `n_trials` saques independentes com a politica `act`."""
    rng = random.Random(eval_seed)
    # O RNG do jogo so e usado pela IA adversaria e pelo reset_ball apos um
    # ponto; nenhum dos dois afeta o resultado de uma tentativa.
    game = PongGame(seed=eval_seed)
    outcomes = []
    for _ in range(n_trials):
        serve(game, rng, angle_max)
        if on_trial_start is not None:
            on_trial_start()
        outcomes.append(play_trial(game, act))
    hits = int(sum(outcomes))
    lo, hi = wilson_interval(hits, n_trials)
    return {
        "eval_seed": eval_seed,
        "n_trials": n_trials,
        "hits": hits,
        "hit_rate": hits / n_trials if n_trials else 0.0,
        "ci95_low": lo,
        "ci95_high": hi,
    }


def invert_readout(runner: "main.FlyPongRunner") -> None:
    """Troca os grupos "sobe"/"desce" do readout e inverte o baseline.

    So o readout e invertido; a rede e o canal sintetico (dna10_up/down_idx,
    fixados no __init__ do runner) ficam como estao. Chamar duas vezes
    restaura o estado original.
    """
    net = runner.net
    net.motor_up_idx, net.motor_down_idx = net.motor_down_idx, net.motor_up_idx
    runner.motor_baseline_diff = -runner.motor_baseline_diff


def network_policy(runner: "main.FlyPongRunner", noise_std: float = 0.0,
                   noise_seed: int = 0):
    """Politica da rede sem reforco: mesma via sensorio-motora de
    FlyPongRunner.step_frame (estimulo -> substeps -> readout), mas a corrente
    externa e so o estimulo visual (mais ruido intrinseco opcional nos
    descendentes, com RNG proprio)."""
    net = runner.net
    external = np.zeros(net.n, dtype=np.float64)
    accum = np.zeros(net.n, dtype=np.float64)
    desc = net.descending_idx
    noise_rng = np.random.default_rng(noise_seed)

    def act(game: PongGame) -> int:
        stim = ball_to_photoreceptor_stimulus(
            runner.perceived_ball_y(game), game.ball_x, HEIGHT, WIDTH,
            net.photoreceptor_positions,
        )
        accum.fill(0.0)
        for _ in range(runner.substeps):
            external.fill(0.0)
            external[net.photoreceptor_idx] += stim
            if noise_std > 0:
                external[desc] += noise_rng.normal(0.0, noise_std, len(desc))
            np.add(accum, net.step(external), out=accum)
        return read_motor_action(
            accum, net.motor_up_idx, net.motor_down_idx,
            baseline_diff=runner.motor_baseline_diff,
        )

    return act


def evaluate_network(runner: "main.FlyPongRunner", n_trials: int, eval_seed: int,
                     noise_std: float = 0.0) -> dict:
    """Avalia a rede com pesos congelados. Restaura PLASTIC_LR ao sair.

    `noise_std`: ruido intrinseco nos descendentes (mesmo valor para rede
    treinada e controles); a semente do ruido deriva de `eval_seed`."""
    saved_lr = netmod.PLASTIC_LR
    w_before = runner.net.W.data.copy()
    netmod.PLASTIC_LR = 0.0
    try:
        result = run_serves(
            network_policy(runner, noise_std, noise_seed=eval_seed + 104729),
            n_trials, eval_seed,
            on_trial_start=runner.net.reset,
        )
    finally:
        netmod.PLASTIC_LR = saved_lr
        runner.net.reset()
    if not np.array_equal(w_before, runner.net.W.data):
        raise RuntimeError("pesos mudaram durante a avaliacao congelada")
    result["eval_noise_std"] = noise_std
    return result


def still_policy(game: PongGame) -> int:
    """Referencia de acaso: paddle parado."""
    return 0


def oracle_policy(game: PongGame) -> int:
    """Referencia de teto: segue a bola com posicao exata e sem atraso."""
    diff = game.ball_y - (game.paddle_left_y + PADDLE_H / 2)
    return 0 if abs(diff) < 1.0 else (1 if diff > 0 else -1)


def build_runner(data_dir: str, *, substeps=None, calibration_frames=None,
                 sensory_mode="allocentric", retina_axis="pca"):
    return main.FlyPongRunner(
        data_dir,
        signal_mode="real",
        sensory_mode=sensory_mode,
        retina_axis=retina_axis,
        substeps=substeps,
        record_history=False,
        calibration_frames=calibration_frames,
    )


def main_cli():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=main.DATA_DIR)
    ap.add_argument("--condition", choices=["normal", "inverted", "both"], default="both")
    ap.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    ap.add_argument("--eval-seeds", type=int, nargs="+", default=list(EVAL_SEEDS))
    ap.add_argument("--substeps", type=int, default=None)
    ap.add_argument("--calibration-frames", type=int, default=None)
    ap.add_argument("--sensory-mode", choices=["allocentric", "egocentric"],
                    default="allocentric")
    ap.add_argument("--retina-axis", choices=["pca", "elevation"], default="pca")
    ap.add_argument("--no-baselines", action="store_true",
                    help="nao roda as referencias paddle-parado e oraculo")
    ap.add_argument("--out", default=None, help="arquivo JSONL de saida (padrao: runs/)")
    args = ap.parse_args()

    if not os.path.exists(os.path.join(args.data_dir, "neurons.parquet")):
        print(f"ERRO: dados do conectoma nao encontrados em {args.data_dir}.")
        return 2
    with open(os.path.join(args.data_dir, "metadata.json"), encoding="utf-8") as f:
        source = json.load(f).get("source", "desconhecido")

    rows = []
    if not args.no_baselines:
        for name, policy in (("still", still_policy), ("oracle", oracle_policy)):
            for seed in args.eval_seeds:
                rows.append({"policy": name, **run_serves(policy, args.trials, seed)})

    conditions = ["normal", "inverted"] if args.condition == "both" else [args.condition]
    runner = build_runner(args.data_dir, substeps=args.substeps,
                          calibration_frames=args.calibration_frames,
                          sensory_mode=args.sensory_mode,
                          retina_axis=args.retina_axis)
    inverted = False
    for cond in conditions:
        if (cond == "inverted") != inverted:
            invert_readout(runner)
            inverted = not inverted
        for seed in args.eval_seeds:
            t0 = time.perf_counter()
            res = evaluate_network(runner, args.trials, seed)
            rows.append({
                "policy": "network_innate",
                "condition": cond,
                "sensory_mode": args.sensory_mode,
                "retina_axis": args.retina_axis,
                **res,
                "motor_baseline_diff": runner.motor_baseline_diff,
                "elapsed_seconds": time.perf_counter() - t0,
            })

    for row in rows:
        row["data_source"] = source
        label = row["policy"] + (f"/{row['condition']}" if "condition" in row else "")
        print(f"{label:24s} seed={row['eval_seed']:5d}  "
              f"acerto={row['hit_rate']:.3f}  "
              f"IC95=[{row['ci95_low']:.3f}, {row['ci95_high']:.3f}]  "
              f"({row['hits']}/{row['n_trials']})")

    out = args.out
    if out is None:
        os.makedirs(main.RUNS_DIR, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        out = os.path.join(main.RUNS_DIR, f"serve_eval_{stamp}.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[serve_eval] resultados salvos em: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
