"""Baseline trivial pra comparacao honesta com o circuito do conectoma
(Achado 14, Fase 4, inspirado no padrao de reportar do FLYT3): uma politica
linear de 1 parametro, sem nenhuma rede neural, jogando o mesmo Pong com a
mesma metrica de taxa de rebatida (janela movel de 20, mesmo criterio
inicio/fim de main.py:FlyPongRunner.validation_report)."""
from game.pong import PongGame, PADDLE_H

BOUNCE_RATE_WINDOW = 20


def run_trivial_baseline(seed, frames=15000, deadzone=0.0):
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


if __name__ == "__main__":
    import json
    for seed in [42, 1, 2, 3, 4, 5]:
        print(json.dumps({"seed": seed, **run_trivial_baseline(seed)}))
