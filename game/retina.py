"""Retina realista (item 2): luminancia vista por cada omatidio.

A mosca e o paddle esquerdo, vista de cima: o Pong e um plano horizontal,
a mosca olha para o campo (+x) e o paddle se mover e a mosca se deslocar
para o lado. A bola fica no horizonte (elevacao 0) e o desvio lateral em
relacao a mosca vira azimute: bola "abaixo" do paddle na tela fica a
direita da mosca (olho direito), "acima" fica a esquerda. A bola e um disco
escuro sobre fundo cinza; o tamanho angular cresce quando ela se aproxima.
(Uma primeira versao punha o plano do Pong vertical a frente da mosca: a
bola ficava sempre em azimute 0 e so ~2% dos omatidios a viam.)
Cada omatidio tem uma direcao (az_deg, el_deg; scripts/add_visual_directions.py)
e sensibilidade gaussiana com largura a meia altura de 5 graus.

Escala da cena (escolha documentada): para a bola ser visivel em todo o
campo, ela e tratada como um objeto de raio BALL_VISUAL_RADIUS (unidades do
jogo), 5 vezes o raio desenhado: ~2,4 graus no fundo do campo, ~5 graus no
meio, ~22 graus perto da mosca.
"""
from __future__ import annotations

import numpy as np

from game.constants import BALL_SIZE, PADDLE_H, PADDLE_W

BACKGROUND = 0.5
BALL_LUMINANCE = 0.0
ACCEPTANCE_FWHM_DEG = 5.0
BALL_VISUAL_RADIUS = 5 * BALL_SIZE / 2
_SIGMA = ACCEPTANCE_FWHM_DEG / 2.355


def ball_direction(game):
    """(azimute em graus, raio angular em graus) da bola vista do paddle.
    Azimute positivo = a direita da mosca (y do jogo cresce para baixo)."""
    fly_x = PADDLE_W
    fly_y = game.paddle_left_y + PADDLE_H / 2
    bx = game.ball_x + BALL_SIZE / 2
    by = game.ball_y + BALL_SIZE / 2
    dx = max(bx - fly_x, 1.0)
    az = np.degrees(np.arctan2(by - fly_y, dx))
    dist = float(np.hypot(dx, by - fly_y))
    rho = np.degrees(np.arctan2(BALL_VISUAL_RADIUS, dist))
    return float(az), float(rho)


def luminance(az_deg, el_deg, az_ball, rho_ball, el_ball=0.0):
    """Luminancia em cada omatidio: disco de raio angular rho suavizado pela
    sensibilidade gaussiana do omatidio (aproximacao de convolucao)."""
    d2 = (az_deg - az_ball) ** 2 + (el_deg - el_ball) ** 2
    s2 = _SIGMA ** 2
    r2 = rho_ball ** 2
    weight = (r2 / (r2 + 2 * s2)) * np.exp(-d2 / (2 * s2 + r2))
    return BACKGROUND + (BALL_LUMINANCE - BACKGROUND) * weight
