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
import random
import time
from datetime import datetime, timezone

import numpy as np

import sim.network as netmod
from sim.network import ConnectomeNetwork
from game.pong import PongGame, WIDTH, HEIGHT, PADDLE_H
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
DENSE_REWARD_DEADZONE = PADDLE_H * 0.25  # px de tolerancia perto do centro do paddle (Achado 11)
# Achado 13: peso inicial do canal sintetico plastico calibrado pra ficar na
# mesma ordem de grandeza do peso medio das sinapses plasticas REAIS
# (object/target->descending), ~8.93 nos logs dos Achados 10-12 -- nao e um
# valor artificialmente gigante.
SYNTHETIC_W_INIT = 8.93
SYNTHETIC_DELTA_MAX = 3.0  # mesmo teto usado nas sinapses plasticas reais (PLASTIC_DELTA_MAX)


class FlyPongRunner:
    def __init__(self, data_dir, reward_mode="sparse", mix_ratio=0.0, signal_mode="real",
                 synthetic_gain=0.0):
        """signal_mode (Achado 13 -- testa diretamente a hipotese do Achado 8
        de que o canal LC10a->DNa10 e fraco demais pra sustentar aprendizado):
        - "real" (padrao): so o circuito real (LC10a->DNa10 via sinapses do
          conectoma).
        - "synthetic_strong": injeta corrente adicional FIXA (nao plastica),
          de magnitude `synthetic_gain`, diretamente nos neuronios DNa10,
          proporcional a sign(ball_y - centro_do_paddle). Teto de capacidade:
          se nem isso melhora a taxa de rebatida, o gargalo nao e (so) o
          canal sensorial.
        - "synthetic_plastic": mesma injecao, mas com peso inicial
          `SYNTHETIC_W_INIT` sujeito a regra de plasticidade de tres fatores
          igual as sinapses reais (mesmo PLASTIC_LR, mesmo teto de delta).

        reward_mode:
        - "sparse" (padrao, Achados 4-10): dopamina so dispara em eventos de
          contato/perda de bola.
        - "dense" (Achado 11): a cada frame, dopamina dispara com base em se
          a acao motora tomada bateu com a direcao correta
          (sinal de ball_y - centro_do_paddle), independente de contato.
        - "mix": com probabilidade `mix_ratio` por frame, aplica reforco
          denso naquele frame (alem do reforco esparso, que SEMPRE dispara
          nos eventos de contato/perda em qualquer modo -- e o resultado que
          define a metrica de validacao, nao muda com o modo de reforco).
          mix_ratio=0.0 reproduz exatamente o modo "sparse"; mix_ratio=1.0
          reproduz o modo "dense" (mais o reforco esparso ocasional, que so
          se sobrepoe nos poucos frames em que ha contato/perda).
        """
        self.net = ConnectomeNetwork(data_dir)
        self.game = PongGame(seed=42)
        self.motor_spike_accum = np.zeros(self.net.n)
        self.reward_pulse_remaining = 0
        self.reward_valence = 0.0
        self.reward_mode = reward_mode
        self.mix_ratio = mix_ratio
        self._last_action = 0
        # RNG separado do RNG do jogo (self.game.rng, usado pra fisica da
        # bola) -- reusar o mesmo stream pro sorteio de reforco denso
        # acoplava os dois, mudando a trajetoria da bola so por causa do
        # modo de reforco, mesmo sem nenhuma diferenca real de peso/plasticidade
        # (bug real encontrado no Achado 11, ver README). Seed fixo e
        # independente do seed do jogo -- nao precisa variar entre
        # experimentos, so precisa nao ser o mesmo stream da fisica da bola.
        self._reward_rng = random.Random(0xF17B0A7)

        self.signal_mode = signal_mode
        self.synthetic_gain = synthetic_gain
        self.synthetic_weight = SYNTHETIC_W_INIT
        self.synthetic_delta = 0.0
        self.synthetic_weight_log = []
        # DNa10 dentro de cada grupo motor (interseccao com motor_up/down) --
        # o sinal sintetico so entra nesses neuronios, nunca nas Giant Fiber.
        self.dna10_up_idx = np.intersect1d(self.net.motor_up_idx, self.net.dna10_idx)
        self.dna10_down_idx = np.intersect1d(self.net.motor_down_idx, self.net.dna10_idx)

        self.point_outcomes = []  # 'bounce' ou 'miss', um por ponto perdido/ganho da nossa perspectiva
        self.bounce_rate_curve = []
        self.plastic_weight_log = []
        self.dense_reward_log = []  # so preenchido quando o reforco denso dispara (Achado 11)
        self.motor_baseline_diff = self._calibrate_motor_baseline()

    def _effective_dense_probability(self):
        if self.reward_mode == "sparse":
            return 0.0
        if self.reward_mode == "dense":
            return 1.0
        return self.mix_ratio

    def _direction_signal(self):
        """sign(ball_y - centro_do_paddle), com zona morta -- mesma logica de
        `_dense_reward_signal` mas usada como ESTIMULO sensorial (Achado 13),
        nao como reforco."""
        paddle_center = self.game.paddle_left_y + PADDLE_H / 2
        diff = self.game.ball_y - paddle_center
        if abs(diff) < DENSE_REWARD_DEADZONE:
            return 0.0
        return 1.0 if diff > 0 else -1.0

    def _synthetic_current_vector(self, direction_signal):
        """Corrente extra nos neuronios DNa10 (Achado 13): quando a bola esta
        abaixo do paddle (direction_signal>0, acao correta = desce), excita
        o grupo "desce" e inibe o "sobe"; inverte quando a bola esta acima.
        Zero se nenhum DNa10 foi encontrado no subgrafo (ex. dados
        sinteticos) ou se o modo de sinal e "real"."""
        current = np.zeros(self.net.n)
        if self.signal_mode == "real" or direction_signal == 0.0 or not len(self.net.dna10_idx):
            return current
        w = self.synthetic_gain if self.signal_mode == "synthetic_strong" else self.synthetic_weight
        if direction_signal > 0:
            current[self.dna10_down_idx] += w
            current[self.dna10_up_idx] -= w
        else:
            current[self.dna10_up_idx] += w
            current[self.dna10_down_idx] -= w
        return current

    def _dense_reward_signal(self, action):
        """+1 se `action` bate com a direcao correta (sinal de ball_y menos o
        centro do paddle), -1 se bate com a direcao oposta, 0 na zona morta
        perto do centro ou se `action` for 0 (parado)."""
        paddle_center = self.game.paddle_left_y + PADDLE_H / 2
        diff = self.game.ball_y - paddle_center
        if abs(diff) < DENSE_REWARD_DEADZONE or action == 0:
            return 0.0
        correct_action = 1 if diff > 0 else -1
        return 1.0 if action == correct_action else -1.0

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
        direction_signal = self._direction_signal()
        synthetic_current = self._synthetic_current_vector(direction_signal)

        self.motor_spike_accum[:] = 0.0
        last_spikes_subset = None
        for _ in range(SUBSTEPS_PER_FRAME):
            external = self._reward_current_vector()
            external[self.net.photoreceptor_idx] += stim
            external += synthetic_current
            spikes = self.net.step(external)
            self.motor_spike_accum += spikes
            last_spikes_subset = spikes

        if self.signal_mode == "synthetic_plastic" and direction_signal != 0.0 and netmod.PLASTIC_LR != 0.0:
            # Regra de tres fatores simplificada pro canal sintetico: "pre"
            # esta sempre ativo quando direction_signal!=0 (e um estimulo
            # constante, nao um spike), entao a elegibilidade e so o post
            # spike do lado que deveria responder. Atualizacao a nivel de
            # frame (nao substep) -- granularidade mais grossa que a regra
            # real (sim/network.py), mas usa o mesmo PLASTIC_LR e o mesmo
            # teto PLASTIC_DELTA_MAX/SYNTHETIC_DELTA_MAX pra comparabilidade.
            post_idx = self.dna10_down_idx if direction_signal > 0 else self.dna10_up_idx
            post_spiked = float(self.motor_spike_accum[post_idx].sum() > 0) if len(post_idx) else 0.0
            dw = netmod.PLASTIC_LR * self.net.dopamine_level * post_spiked
            self.synthetic_delta = float(np.clip(
                self.synthetic_delta + dw, -SYNTHETIC_DELTA_MAX, SYNTHETIC_DELTA_MAX
            ))
            self.synthetic_weight = SYNTHETIC_W_INIT + self.synthetic_delta
        self.synthetic_weight_log.append(self.synthetic_weight)

        action = read_motor_action(
            self.motor_spike_accum, self.net.motor_up_idx, self.net.motor_down_idx,
            baseline_diff=self.motor_baseline_diff,
        )
        self._last_action = action

        if self._reward_rng.random() < self._effective_dense_probability():
            dense_r = self._dense_reward_signal(action)
            if dense_r != 0:
                self.reward_valence = dense_r
                self.reward_pulse_remaining = REWARD_PULSE_STEPS
                self.dense_reward_log.append(dense_r)

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
