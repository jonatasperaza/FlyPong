"""FlyPong: Pong controlado por um subgrafo do conectoma MaleCNS v1.0.

Uso:
    python main.py                       # janela normal (precisa de display)
    python main.py --headless --frames 6000   # roda sem janela, so gera relatorio de validacao

IMPORTANTE (honestidade cientifica): este e um modelo computacional
aproximado. Nao ha alegacao de que "a mosca esta jogando" ou de qualquer
forma de consciencia/senciencia. Ver README.md para os resultados de
validacao (incluindo se o "aprendizado" de fato ocorreu, com numeros).
"""
import argparse
import csv
import os
import time
from datetime import datetime, timezone

import numpy as np

from sim.network import ConnectomeNetwork
from game.pong import PongGame, WIDTH, HEIGHT
from game.sensory_map import ball_to_photoreceptor_stimulus
from game.motor_read import read_motor_action

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "connectome_data")
RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")

SUBSTEPS_PER_FRAME = 4
MOTOR_WINDOW_FRAMES = 3  # janelas de leitura motora, em frames de jogo
REWARD_PULSE_STEPS = 6
REWARD_CURRENT = 8.0
BOUNCE_RATE_WINDOW = 20  # numero de pontos usados na media movel de validacao
CALIBRATION_FRAMES = 200  # frames com a bola parada no centro, so pra medir o baseline sobe/desce


class FlyPongRunner:
    def __init__(self, data_dir):
        self.net = ConnectomeNetwork(data_dir)
        self.game = PongGame(seed=42)
        self.motor_spike_accum = np.zeros(self.net.n)
        self.reward_pulse_remaining = 0
        self.reward_valence = 0.0

        self.point_outcomes = []  # 'bounce' ou 'miss', um por ponto perdido/ganho da nossa perspectiva
        self.bounce_rate_curve = []
        self.plastic_weight_log = []
        self.motor_baseline_diff = self._calibrate_motor_baseline()

    def _calibrate_motor_baseline(self):
        """Mede o viés estrutural sobe/desce com a bola parada no centro (sem
        informacao real sobre posicao). Ver docstring de read_motor_action."""
        diffs = []
        for _ in range(CALIBRATION_FRAMES):
            stim = ball_to_photoreceptor_stimulus(
                HEIGHT / 2, WIDTH / 2, HEIGHT, WIDTH, self.net.photoreceptor_positions,
            )
            accum = np.zeros(self.net.n)
            for _ in range(SUBSTEPS_PER_FRAME):
                external = np.zeros(self.net.n)
                external[self.net.photoreceptor_idx] += stim
                accum += self.net.step(external)
            up_rate = accum[self.net.motor_up_idx].mean() if len(self.net.motor_up_idx) else 0.0
            down_rate = accum[self.net.motor_down_idx].mean() if len(self.net.motor_down_idx) else 0.0
            diffs.append(down_rate - up_rate)
        self.net.reset()
        return float(np.mean(diffs))

    def _reward_current_vector(self):
        current = np.zeros(self.net.n)
        if self.reward_pulse_remaining > 0:
            idx = self.net.dopamine_positive_idx if self.reward_valence > 0 else self.net.dopamine_negative_idx
            current[idx] = REWARD_CURRENT
            self.reward_pulse_remaining -= 1
        return current

    def step_frame(self):
        stim = ball_to_photoreceptor_stimulus(
            self.game.ball_y, self.game.ball_x, HEIGHT, WIDTH,
            self.net.photoreceptor_positions,
        )

        self.motor_spike_accum[:] = 0.0
        last_spikes_subset = None
        for _ in range(SUBSTEPS_PER_FRAME):
            external = self._reward_current_vector()
            external[self.net.photoreceptor_idx] += stim
            spikes = self.net.step(external)
            self.motor_spike_accum += spikes
            last_spikes_subset = spikes

        action = read_motor_action(
            self.motor_spike_accum, self.net.motor_up_idx, self.net.motor_down_idx,
            baseline_diff=self.motor_baseline_diff,
        )

        event = self.game.step(action)

        # Metrica de validacao = taxa de REBATIDA (o paddle nosso conseguiu ou
        # nao devolver a bola), nao taxa de pontos ganhos: o oponente da
        # direita e uma IA quase perfeita (rastreia a bola com precisao a
        # cada frame), entao "ganhar pontos" contra ela e praticamente
        # impossivel por construcao do jogo, independente da qualidade do
        # controle neural. O evento relevante e o contato paddle-bola.
        if event["bounce"]:
            self.point_outcomes.append("bounce")
            self.reward_valence = 1.0
            self.reward_pulse_remaining = REWARD_PULSE_STEPS
        elif event["score"] == "right":
            self.point_outcomes.append("miss")
            self.reward_valence = -1.0
            self.reward_pulse_remaining = REWARD_PULSE_STEPS

        if event["bounce"] or event["score"] == "right":
            recent = self.point_outcomes[-BOUNCE_RATE_WINDOW:]
            rate = recent.count("bounce") / len(recent)
            self.bounce_rate_curve.append(rate)

        self.net.record_plastic_weight_snapshot()
        self.plastic_weight_log.append(self.net.plastic_weight_history[-1])

        return action, last_spikes_subset

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
                "aprendizado detectado (taxa de rebatida aumentou)" if delta > 0.05
                else "sem evidencia de aprendizado" if abs(delta) <= 0.05
                else "taxa de rebatida piorou"
            ),
        }


def run_headless(frames, data_dir):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    runner = FlyPongRunner(data_dir)
    t0 = time.time()
    for i in range(frames):
        runner.step_frame()
    elapsed = time.time() - t0

    os.makedirs(RUNS_DIR, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_path = os.path.join(RUNS_DIR, f"validation_{stamp}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["point_index", "bounce_rate_rolling", "plastic_weight_mean"])
        step_per_point = max(1, frames // max(1, len(runner.bounce_rate_curve)))
        for i, rate in enumerate(runner.bounce_rate_curve):
            w.writerow([i, rate, ""])

    report = runner.validation_report()
    print(f"[headless] {frames} frames em {elapsed:.1f}s ({frames/elapsed:.0f} frames/s)")
    print(f"[headless] tentativas de rebatida: {report['n_attempts']}")
    if "bounce_rate_inicio" in report:
        print(f"[headless] taxa de rebatida (inicio): {report['bounce_rate_inicio']:.2f}")
        print(f"[headless] taxa de rebatida (fim):    {report['bounce_rate_fim']:.2f}")
        print(f"[headless] delta: {report['delta']:+.2f}")
    print(f"[headless] veredito: {report['verdict']}")
    print(f"[headless] log salvo em: {csv_path}")
    return report


def run_windowed(data_dir, fps=60):
    import pygame
    from dashboard import Dashboard, PANEL_W

    pygame.init()
    screen = pygame.display.set_mode((WIDTH + PANEL_W, HEIGHT))
    pygame.display.set_caption("FlyPong")
    clock = pygame.time.Clock()

    runner = FlyPongRunner(data_dir)
    n_raster = min(40, runner.net.n)
    raster_idx = np.linspace(0, runner.net.n - 1, n_raster).astype(int)
    dashboard = Dashboard(HEIGHT, n_raster)

    running = True
    while running:
        for evt in pygame.event.get():
            if evt.type == pygame.QUIT:
                running = False

        action, spikes = runner.step_frame()
        dashboard.push_frame(spikes[raster_idx], runner.net.plastic_weight_history[-1])

        screen.fill((0, 0, 0))
        pygame.draw.rect(screen, (255, 255, 255), (0, runner.game.paddle_left_y, 10, 60))
        pygame.draw.rect(screen, (255, 255, 255), (WIDTH - 10, runner.game.paddle_right_y, 10, 60))
        pygame.draw.rect(screen, (255, 255, 255), (runner.game.ball_x, runner.game.ball_y, 8, 8))
        dashboard.draw(screen, WIDTH, runner.game.score_left, runner.game.score_right,
                       action, runner.net.dopamine_level)

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
    args = ap.parse_args()

    if not os.path.exists(os.path.join(args.data_dir, "neurons.parquet")):
        print(f"ERRO: dados do conectoma nao encontrados em {args.data_dir}. "
              f"Rode primeiro: python fetch_connectome.py --synthetic (ou com --token).")
        return

    if args.headless:
        run_headless(args.frames, args.data_dir)
    else:
        run_windowed(args.data_dir, fps=args.fps)


if __name__ == "__main__":
    main()
