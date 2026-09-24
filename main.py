"""FlyPong with an opt-in screening mode for fast experiment iteration."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from datetime import datetime, timezone

import numpy as np

import sim.network as netmod
from sim.network import ConnectomeNetwork
from game.pong import PongGame, WIDTH, HEIGHT, PADDLE_H
from game.sensory_map import ball_to_photoreceptor_stimulus, egocentric_ball_y
from game.motor_read import read_motor_action

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "connectome_data")
RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")

SUBSTEPS_PER_FRAME = 4
REWARD_PULSE_STEPS = 6
REWARD_CURRENT = 8.0
PROXIMITY_REWARD_CURRENT = 3.0
PROXIMITY_PROGRESS_DEADZONE = 0.5
BOUNCE_RATE_WINDOW = 20
CALIBRATION_FRAMES = 200
DENSE_REWARD_DEADZONE = PADDLE_H * 0.25
SYNTHETIC_W_INIT = 8.93
SYNTHETIC_DELTA_MAX = 3.0


class FlyPongRunner:
    def __init__(
        self,
        data_dir,
        reward_mode="sparse",
        mix_ratio=0.0,
        signal_mode="real",
        synthetic_gain=10.0,
        *,
        substeps=None,
        prune_causal=False,
        record_history=True,
        calibration_frames=None,
        game_seed=42,
        proximity_reward_current=PROXIMITY_REWARD_CURRENT,
        proximity_progress_deadzone=PROXIMITY_PROGRESS_DEADZONE,
        sensory_mode="allocentric",
        retina_axis="pca",
    ):
        if sensory_mode not in {"allocentric", "egocentric"}:
            raise ValueError(f"unknown sensory mode: {sensory_mode}")
        self.sensory_mode = sensory_mode
        if reward_mode not in {"sparse", "dense", "mix", "proximity"}:
            raise ValueError(f"unknown reward mode: {reward_mode}")
        if not 0.0 <= mix_ratio <= 1.0:
            raise ValueError("mix_ratio must be between 0 and 1")
        self.substeps = (
            SUBSTEPS_PER_FRAME if substeps is None else max(1, int(substeps))
        )
        self.record_history = bool(record_history)
        self.net = ConnectomeNetwork(
            data_dir,
            prune_causal=prune_causal,
            record_history=self.record_history,
            retina_axis=retina_axis,
        )
        self.game = PongGame(seed=game_seed)
        self.game_seed = game_seed
        self.motor_spike_accum = np.zeros(self.net.n, dtype=np.float64)
        self.external = np.zeros(self.net.n, dtype=np.float64)
        self._synthetic_current = np.zeros(self.net.n, dtype=np.float64)
        self.reward_pulse_remaining = 0
        self.reward_valence = 0.0
        self.reward_current = REWARD_CURRENT
        self.reward_mode = reward_mode
        self.mix_ratio = mix_ratio
        self.proximity_reward_current = float(proximity_reward_current)
        self.proximity_progress_deadzone = float(proximity_progress_deadzone)
        self._previous_approach_distance = None
        self._last_action = 0
        self._reward_rng = random.Random(0xF17B0A7)
        self.signal_mode = signal_mode
        self.synthetic_gain = synthetic_gain
        self.synthetic_weight = SYNTHETIC_W_INIT
        self.synthetic_delta = 0.0
        self.synthetic_weight_log = [] if self.record_history else None
        self.dna10_up_idx = np.intersect1d(self.net.motor_up_idx, self.net.dna10_idx)
        self.dna10_down_idx = np.intersect1d(
            self.net.motor_down_idx, self.net.dna10_idx
        )
        self.point_outcomes = []
        self.bounce_rate_curve = []
        self.plastic_weight_log = [] if self.record_history else None
        self.dense_reward_log = [] if self.record_history else None
        self.proximity_reward_log = [] if self.record_history else None
        calib = (
            CALIBRATION_FRAMES
            if calibration_frames is None
            else max(1, int(calibration_frames))
        )
        self.motor_baseline_diff = self._calibrate_motor_baseline(calib)

    def _effective_dense_probability(self):
        if self.reward_mode == "sparse":
            return 0.0
        if self.reward_mode == "dense":
            return 1.0
        if self.reward_mode == "mix":
            return self.mix_ratio
        return 0.0

    def _direction_signal(self):
        paddle_center = self.game.paddle_left_y + PADDLE_H / 2
        diff = self.game.ball_y - paddle_center
        if abs(diff) < DENSE_REWARD_DEADZONE:
            return 0.0
        return 1.0 if diff > 0 else -1.0

    def _synthetic_current_vector(self, direction_signal):
        if (
            self.signal_mode == "real"
            or direction_signal == 0.0
            or not len(self.net.dna10_idx)
        ):
            return None
        self._synthetic_current.fill(0.0)
        w = (
            self.synthetic_gain
            if self.signal_mode == "synthetic_strong"
            else self.synthetic_weight
        )
        if direction_signal > 0:
            self._synthetic_current[self.dna10_down_idx] += w
            self._synthetic_current[self.dna10_up_idx] -= w
        else:
            self._synthetic_current[self.dna10_up_idx] += w
            self._synthetic_current[self.dna10_down_idx] -= w
        return self._synthetic_current

    def _dense_reward_signal(self, action):
        paddle_center = self.game.paddle_left_y + PADDLE_H / 2
        diff = self.game.ball_y - paddle_center
        if abs(diff) < DENSE_REWARD_DEADZONE or action == 0:
            return 0.0
        correct_action = 1 if diff > 0 else -1
        return 1.0 if action == correct_action else -1.0

    def _proximity_reward_signal(self, was_approaching):
        """Return progress reward while the ball moves toward the left paddle.

        This is a potential-style shaping signal: it compares the current
        vertical ball-paddle distance to the previous frame. It deliberately
        does not run while the ball travels away, so the agent is not rewarded
        for chasing an irrelevant trajectory after a successful bounce.
        """
        if not was_approaching:
            self._previous_approach_distance = None
            return 0.0

        paddle_center = self.game.paddle_left_y + PADDLE_H / 2
        distance = abs(self.game.ball_y - paddle_center)
        previous = self._previous_approach_distance
        self._previous_approach_distance = distance
        if previous is None:
            return 0.0

        progress = previous - distance
        if abs(progress) <= self.proximity_progress_deadzone:
            return 0.0
        return 1.0 if progress > 0 else -1.0

    def _calibrate_motor_baseline(self, frames):
        diffs = []
        stim = ball_to_photoreceptor_stimulus(
            HEIGHT / 2, WIDTH / 2, HEIGHT, WIDTH, self.net.photoreceptor_positions
        )
        for _ in range(frames):
            self.motor_spike_accum.fill(0.0)
            for _ in range(self.substeps):
                self.external.fill(0.0)
                self.external[self.net.photoreceptor_idx] += stim
                self.motor_spike_accum += self.net.step(self.external)
            up = (
                self.motor_spike_accum[self.net.motor_up_idx].mean()
                if len(self.net.motor_up_idx)
                else 0.0
            )
            down = (
                self.motor_spike_accum[self.net.motor_down_idx].mean()
                if len(self.net.motor_down_idx)
                else 0.0
            )
            diffs.append(down - up)
        self.net.reset()
        return float(np.mean(diffs))

    def _reward_current_vector(self):
        self.external.fill(0.0)
        if self.reward_pulse_remaining > 0:
            idx = (
                self.net.dopamine_positive_idx
                if self.reward_valence > 0
                else self.net.dopamine_negative_idx
            )
            self.external[idx] = self.reward_current
            self.reward_pulse_remaining -= 1
        return self.external

    def _schedule_reward(self, valence, *, current=REWARD_CURRENT):
        """Schedule one reward pulse; terminal rewards keep full strength."""
        self.reward_valence = float(valence)
        self.reward_current = float(current)
        self.reward_pulse_remaining = REWARD_PULSE_STEPS

    def perceived_ball_y(self, game=None):
        """Altura da bola como a retina a ve no modo sensorial atual."""
        game = self.game if game is None else game
        if self.sensory_mode == "egocentric":
            return egocentric_ball_y(
                game.ball_y, game.paddle_left_y + PADDLE_H / 2, HEIGHT
            )
        return game.ball_y

    def step_frame(self):
        stim = ball_to_photoreceptor_stimulus(
            self.perceived_ball_y(),
            self.game.ball_x,
            HEIGHT,
            WIDTH,
            self.net.photoreceptor_positions,
        )
        direction_signal = self._direction_signal()
        self._synthetic_current_vector(direction_signal)

        self.motor_spike_accum.fill(0.0)
        last_spikes = None
        for _ in range(self.substeps):
            external = self._reward_current_vector()
            external[self.net.photoreceptor_idx] += stim
            if self.signal_mode != "real" and direction_signal != 0.0:
                external += self._synthetic_current
            last_spikes = self.net.step(external)
            self.motor_spike_accum += last_spikes

        if (
            self.signal_mode == "synthetic_plastic"
            and direction_signal != 0.0
            and netmod.PLASTIC_LR != 0.0
        ):
            post_idx = (
                self.dna10_down_idx if direction_signal > 0 else self.dna10_up_idx
            )
            post_spiked = (
                float(self.motor_spike_accum[post_idx].sum() > 0)
                if len(post_idx)
                else 0.0
            )
            dw = netmod.PLASTIC_LR * self.net.dopamine_level * post_spiked
            self.synthetic_delta = float(
                np.clip(
                    self.synthetic_delta + dw, -SYNTHETIC_DELTA_MAX, SYNTHETIC_DELTA_MAX
                )
            )
            self.synthetic_weight = SYNTHETIC_W_INIT + self.synthetic_delta
        if self.record_history:
            self.synthetic_weight_log.append(self.synthetic_weight)

        action = read_motor_action(
            self.motor_spike_accum,
            self.net.motor_up_idx,
            self.net.motor_down_idx,
            baseline_diff=self.motor_baseline_diff,
        )
        self._last_action = action

        if self._reward_rng.random() < self._effective_dense_probability():
            dense_r = self._dense_reward_signal(action)
            if dense_r:
                self._schedule_reward(dense_r)
                if self.record_history:
                    self.dense_reward_log.append(dense_r)

        was_approaching = self.game.ball_vx < 0.0
        event = self.game.step(action)
        if event["bounce"]:
            self.point_outcomes.append("bounce")
            self._schedule_reward(1.0)
        elif event["score"] == "right":
            self.point_outcomes.append("miss")
            self._schedule_reward(-1.0)
        elif self.reward_mode == "proximity":
            proximity_r = self._proximity_reward_signal(was_approaching)
            if proximity_r:
                self._schedule_reward(
                    proximity_r, current=self.proximity_reward_current
                )
                if self.record_history:
                    self.proximity_reward_log.append(proximity_r)

        if event["bounce"] or event["score"] == "right":
            recent = self.point_outcomes[-BOUNCE_RATE_WINDOW:]
            self.bounce_rate_curve.append(recent.count("bounce") / len(recent))

        if self.record_history:
            self.net.record_plastic_weight_snapshot()
            self.plastic_weight_log.append(self.net.plastic_weight_mean())
        return action, last_spikes

    def validation_report(self):
        n = len(self.bounce_rate_curve)
        if n < 4:
            return {
                "n_attempts": n,
                "verdict": "dados insuficientes (poucas tentativas de rebatida)",
            }
        k = max(1, n // 5)
        before = float(np.mean(self.bounce_rate_curve[:k]))
        after = float(np.mean(self.bounce_rate_curve[-k:]))
        delta = after - before
        return {
            "n_attempts": n,
            "bounce_rate_inicio": before,
            "bounce_rate_fim": after,
            "delta": delta,
            "verdict": (
                "aprendizado detectado (taxa de rebatida aumentou)"
                if delta > 0.05
                else "sem evidencia de aprendizado"
                if abs(delta) <= 0.05
                else "taxa de rebatida piorou"
            ),
        }


def run_headless(frames, data_dir, **kwargs):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    runner = FlyPongRunner(data_dir, **kwargs)
    t0 = time.perf_counter()
    for _ in range(frames):
        runner.step_frame()
    elapsed = time.perf_counter() - t0

    os.makedirs(RUNS_DIR, exist_ok=True)
    report = runner.validation_report()
    # Microseconds keep consecutive batch runs from overwriting one another.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    csv_path = os.path.join(RUNS_DIR, f"validation_{stamp}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["point_index", "bounce_rate_rolling"])
        for i, rate in enumerate(runner.bounce_rate_curve):
            w.writerow([i, rate])

    metadata_path = os.path.join(RUNS_DIR, f"validation_{stamp}.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp_utc": stamp,
                "data_dir": os.path.abspath(data_dir),
                "frames": frames,
                "elapsed_seconds": elapsed,
                "frames_per_second": frames / elapsed if elapsed else None,
                "config": kwargs,
                "effective_model_config": {
                    "synapse_gain": netmod.SYNAPSE_GAIN,
                    "plastic_lr": netmod.PLASTIC_LR,
                    "plasticity_mode": netmod.PLASTICITY_MODE,
                    "plastic_delta_max": netmod.PLASTIC_DELTA_MAX,
                },
                "network": {
                    "n_neurons": runner.net.n,
                    "n_edges": int(runner.net.W.nnz),
                    "n_plastic_synapses": int(len(runner.net.plastic_data_idx)),
                },
                "report": report,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"[headless] {frames} frames em {elapsed:.3f}s ({frames / elapsed:.0f} frames/s)"
    )
    print(f"[headless] tentativas de rebatida: {report['n_attempts']}")
    if "bounce_rate_inicio" in report:
        print(f"[headless] taxa inicio: {report['bounce_rate_inicio']:.2f}")
        print(f"[headless] taxa fim:    {report['bounce_rate_fim']:.2f}")
        print(f"[headless] delta:       {report['delta']:+.2f}")
    print(f"[headless] veredito: {report['verdict']}")
    print(
        f"[headless] modo: substeps={runner.substeps}, n={runner.net.n}, history={runner.record_history}"
    )
    print(f"[headless] log salvo em: {csv_path}")
    print(f"[headless] metadados salvos em: {metadata_path}")
    return report


def run_windowed(data_dir, fps=60, **kwargs):
    import pygame
    from dashboard import Dashboard, PANEL_W

    pygame.init()
    screen = pygame.display.set_mode((WIDTH + PANEL_W, HEIGHT))
    clock = pygame.time.Clock()
    runner = FlyPongRunner(data_dir, **kwargs)
    n_raster = min(40, runner.net.n)
    raster_idx = np.linspace(0, runner.net.n - 1, n_raster).astype(int)
    dashboard = Dashboard(HEIGHT, n_raster)
    running = True
    while running:
        for evt in pygame.event.get():
            if evt.type == pygame.QUIT:
                running = False
        action, spikes = runner.step_frame()
        dashboard.push_frame(spikes[raster_idx], runner.net.plastic_weight_mean())
        screen.fill((0, 0, 0))
        pygame.draw.rect(
            screen, (255, 255, 255), (0, runner.game.paddle_left_y, 10, 60)
        )
        pygame.draw.rect(
            screen, (255, 255, 255), (WIDTH - 10, runner.game.paddle_right_y, 10, 60)
        )
        pygame.draw.rect(
            screen, (255, 255, 255), (runner.game.ball_x, runner.game.ball_y, 8, 8)
        )
        dashboard.draw(
            screen,
            WIDTH,
            runner.game.score_left,
            runner.game.score_right,
            action,
            runner.net.dopamine_level,
        )
        pygame.display.flip()
        clock.tick(fps)
    pygame.quit()
    print(runner.validation_report())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--frames", type=int, default=6000)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42, help="Seed da física do Pong.")
    ap.add_argument(
        "--reward-mode",
        default="sparse",
        choices=["sparse", "dense", "mix", "proximity"],
        help="Agenda de reforço dopaminérgico.",
    )
    ap.add_argument(
        "--mix-ratio",
        type=float,
        default=0.0,
        help="Probabilidade por frame do reforço denso no modo mix.",
    )
    ap.add_argument(
        "--proximity-reward-current",
        type=float,
        default=PROXIMITY_REWARD_CURRENT,
        help="Corrente do pulso fraco de progresso no modo proximity.",
    )
    ap.add_argument(
        "--proximity-deadzone",
        type=float,
        default=PROXIMITY_PROGRESS_DEADZONE,
        help="Variação mínima de distância (pixels) para recompensar progresso.",
    )
    ap.add_argument(
        "--fast",
        action="store_true",
        help="screening mode: causal prune + 1 substep + no history",
    )
    ap.add_argument(
        "--prune-causal",
        action="store_true",
        help="prune graph nodes outside sensory/reward -> motor paths",
    )
    ap.add_argument("--substeps", type=int, default=None)
    ap.add_argument("--no-history", action="store_true")
    ap.add_argument("--calibration-frames", type=int, default=None)
    ap.add_argument(
        "--signal-mode",
        default="real",
        choices=["real", "synthetic_strong", "synthetic_plastic"],
    )
    ap.add_argument(
        "--sensory-mode",
        default="allocentric",
        choices=["allocentric", "egocentric"],
        help="egocentric: retina centrada no paddle (a mosca ve a bola "
        "relativa ao proprio corpo).",
    )
    ap.add_argument("--synthetic-gain", type=float, default=10.0)
    ap.add_argument("--synthetic-w-init", type=float, default=None)
    ap.add_argument("--plastic-lr", type=float, default=None)
    ap.add_argument(
        "--plasticity-mode",
        default=None,
        choices=["original", "freq_normalized", "tonic_baseline"],
    )
    args = ap.parse_args()

    if not os.path.exists(os.path.join(args.data_dir, "neurons.parquet")):
        print(f"ERRO: dados do conectoma nao encontrados em {args.data_dir}.")
        return 2

    if args.synthetic_w_init is not None:
        globals()["SYNTHETIC_W_INIT"] = args.synthetic_w_init
    if args.plastic_lr is not None:
        netmod.PLASTIC_LR = args.plastic_lr
    if args.plasticity_mode is not None:
        netmod.PLASTICITY_MODE = args.plasticity_mode
    if not 0.0 <= args.mix_ratio <= 1.0:
        ap.error("--mix-ratio deve estar entre 0 e 1.")

    fast = args.fast
    kwargs = dict(
        reward_mode=args.reward_mode,
        mix_ratio=args.mix_ratio,
        proximity_reward_current=args.proximity_reward_current,
        proximity_progress_deadzone=args.proximity_deadzone,
        signal_mode=args.signal_mode,
        sensory_mode=args.sensory_mode,
        synthetic_gain=args.synthetic_gain,
        game_seed=args.seed,
        substeps=1 if fast else args.substeps,
        prune_causal=bool(args.prune_causal or fast),
        record_history=not bool(args.no_history or fast),
        calibration_frames=25
        if fast and args.calibration_frames is None
        else args.calibration_frames,
    )
    if args.headless:
        return (
            0 if run_headless(args.frames, args.data_dir, **kwargs) is not None else 1
        )
    run_windowed(args.data_dir, fps=args.fps, **kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
