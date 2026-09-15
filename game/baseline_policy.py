"""Baselines sem rede neural pra comparacao com o circuito do conectoma
(Achado 14). Jogam o mesmo Pong, mesma metrica de taxa de rebatida (janela
movel de 20, mesmo criterio inicio/fim de
main.py:FlyPongRunner.validation_report).

Dois baselines, deliberadamente diferentes:

- `run_trivial_baseline`: oraculo (le ball_y exato, sem atraso). Serve so
  como teto teorico (Achado 14, Fase 4 original) -- nao e comparavel ao
  circuito real de forma justa, ganha por construcao.
- `run_delayed_linear_baseline` (Verificacao 2): sujeito as duas mesmas
  limitacoes que o circuito neural real tem, pra ser uma comparacao que
  significa alguma coisa:
    1. Le a posicao da bola so atraves do mesmo estimulo populacional que os
       fotorreceptores simulados recebem (`ball_to_photoreceptor_stimulus`),
       nao o `ball_y` exato do estado do jogo -- resolucao limitada a
       `N_RETINA` posicoes discretas, com a mesma forma gaussiana de
       ativacao que o circuito real usa.
    2. Age com atraso de `delay_frames` frames (padrao 1): o circuito real
       precisa de varios saltos sinapticos (fotorreceptor -> interneuronio
       -> target/motion -> object -> descending) pra um estimulo chegar no
       motor, e com so 4 substeps por frame isso normalmente exige mais de
       um frame pra atravessar toda a cadeia -- entao a acao de um frame
       tende a refletir a posicao da bola de pelo menos 1 frame atras, nao a
       posicao atual.
"""
import numpy as np

from game.pong import PongGame, PADDLE_H, WIDTH, HEIGHT
from game.sensory_map import ball_to_photoreceptor_stimulus

BOUNCE_RATE_WINDOW = 20
N_RETINA = 1783  # mesma contagem de fotorreceptores do subgrafo real (Achado 9)


def _validation_report(point_outcomes, bounce_rate_curve):
    n = len(bounce_rate_curve)
    if n < 4:
        return {"n_attempts": n, "verdict": "dados insuficientes"}
    k = max(1, n // 5)
    before = sum(bounce_rate_curve[:k]) / k
    after = sum(bounce_rate_curve[-k:]) / k
    return {
        "n_attempts": n,
        "bounce_rate_inicio": before,
        "bounce_rate_fim": after,
        "delta": after - before,
    }


def run_trivial_baseline(seed, frames=15000, deadzone=0.0):
    """Oraculo: le ball_y exato, sem atraso. Teto teorico, nao comparavel de
    forma justa ao circuito real (ver docstring do modulo)."""
    game = PongGame(seed=seed)
    point_outcomes = []
    bounce_rate_curve = []

    for _ in range(frames):
        paddle_center = game.paddle_left_y + PADDLE_H / 2
        diff = game.ball_y - paddle_center
        action = 0 if abs(diff) < deadzone else (1 if diff > 0 else -1)
        event = game.step(action)

        if event["bounce"]:
            point_outcomes.append("bounce")
        elif event["score"] == "right":
            point_outcomes.append("miss")

        if event["bounce"] or event["score"] == "right":
            recent = point_outcomes[-BOUNCE_RATE_WINDOW:]
            bounce_rate_curve.append(recent.count("bounce") / len(recent))

    return _validation_report(point_outcomes, bounce_rate_curve)


def run_delayed_linear_baseline(seed, frames=15000, delay_frames=1, deadzone=0.0,
                                 n_retina=N_RETINA):
    """Verificacao 2 (README, Achado 14): mesma politica linear de 1 parametro,
    mas lendo a posicao da bola so via estimulo populacional (resolucao
    limitada) e com atraso de `delay_frames` frames, pra ficar sujeita as
    mesmas duas restricoes do circuito neural real."""
    game = PongGame(seed=seed)
    positions = np.linspace(0, 1, n_retina)
    point_outcomes = []
    bounce_rate_curve = []
    stim_history = []

    for _ in range(frames):
        stim = ball_to_photoreceptor_stimulus(game.ball_y, game.ball_x, HEIGHT, WIDTH, positions)
        stim_history.append(stim)

        if len(stim_history) > delay_frames:
            delayed_stim = stim_history[-(delay_frames + 1)]
            estimated_pos_frac = float(np.sum(positions * delayed_stim) / max(np.sum(delayed_stim), 1e-9))
            estimated_ball_y = estimated_pos_frac * HEIGHT
        else:
            estimated_ball_y = HEIGHT / 2  # sem historico suficiente ainda: fica parado

        paddle_center = game.paddle_left_y + PADDLE_H / 2
        diff = estimated_ball_y - paddle_center
        action = 0 if abs(diff) < deadzone else (1 if diff > 0 else -1)
        event = game.step(action)

        if event["bounce"]:
            point_outcomes.append("bounce")
        elif event["score"] == "right":
            point_outcomes.append("miss")

        if event["bounce"] or event["score"] == "right":
            recent = point_outcomes[-BOUNCE_RATE_WINDOW:]
            bounce_rate_curve.append(recent.count("bounce") / len(recent))

    return _validation_report(point_outcomes, bounce_rate_curve)


if __name__ == "__main__":
    import json
    for seed in [42, 1, 2, 3, 4, 5]:
        print(json.dumps({"baseline": "trivial", "seed": seed, **run_trivial_baseline(seed)}))
    for seed in [42, 1, 2, 3, 4, 5]:
        print(json.dumps({"baseline": "delayed_linear", "seed": seed, **run_delayed_linear_baseline(seed)}))
