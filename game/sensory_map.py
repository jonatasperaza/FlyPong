"""Mapeia o estado da bola do Pong para um estimulo de intensidade nos
fotorreceptores selecionados do conectoma (retina 1D simulada)."""
import numpy as np


def ball_to_photoreceptor_stimulus(ball_y, ball_x, height, width, positions,
                                    gain=6.0, sigma_frac=0.08):
    """Retorna um vetor de corrente externa (mesmo tamanho de `positions`).

    `positions`: posicao normalizada [0,1] de cada fotorreceptor na retina 1D
    (ver ConnectomeNetwork.photoreceptor_positions -- vem de somaLocation real
    quando disponivel, ou de ordem de indice como fallback).

    A posicao Y da bola vira uma "banda" gaussiana de ativacao ao longo da
    retina 1D (analogo a coluna de retina onde a bola esta). A intensidade e
    escalada pela proximidade da bola ao paddle controlado (x pequeno = mais
    perto = mais brilhante), seguindo a especificacao do projeto.
    """
    positions = np.asarray(positions)
    center = ball_y / height
    sigma = sigma_frac
    bump = np.exp(-0.5 * ((positions - center) / sigma) ** 2)

    proximity = 1.0 - np.clip(ball_x / width, 0.0, 1.0)
    proximity = 0.3 + 0.7 * proximity  # nunca deixa o estimulo zerar

    return gain * proximity * bump


def egocentric_ball_y(ball_y, paddle_center, height):
    """Altura "vista" pela retina num referencial centrado no paddle.

    O olho da mosca se move com o corpo: o centro da retina e a altura do
    paddle, e a bola aparece deslocada pela diferenca bola - paddle. A
    diferenca vai de -height a +height; dividir por 2 a mantem dentro da
    retina ([0, height]). O modo alocentrico original (so a posicao absoluta
    da bola) nao informa a posicao do paddle, entao a direcao correta nao
    pode ser inferida a partir dele (ver serve_eval.py).
    """
    return height / 2 + (ball_y - paddle_center) / 2
