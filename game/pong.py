"""Pong minimalista. O paddle esquerdo e controlado pelo conectoma (via
main.py); o paddle direito e uma IA trivial que segue a bola com atraso,
ruido e erro ocasional.

Alteracoes em relacao ao original (para tornar a tarefa aprendivel):
  1. ball_vy tambem cresce a cada rebatida (com teto BALL_VY_MAX),
     quebrando a premissa de que o bot sempre alcanca a bola.
  2. A IA reage a uma posicao atrasada da bola (AI_REACTION_FRAMES).
  3. A IA adiciona ruido gaussiano e "pisca" com probabilidade AI_MISS_PROB.
  4. Curriculo: a dificuldade da IA cresce conforme o lado esquerdo marca.
"""

import random

WIDTH, HEIGHT = 480, 320
PADDLE_H = 60
PADDLE_W = 10
BALL_SIZE = 8
PADDLE_SPEED = 4.0
AI_SPEED = 3.0
BALL_SPEED = 3.5
BALL_SPEED_MAX = 2.5 * BALL_SPEED
BALL_VY_MAX = 6.0  # teto para a velocidade vertical

# --- Parametros da IA (ajuste para controlar dificuldade) ---
AI_REACTION_FRAMES = 8  # quantos frames a IA "ve" no passado
AI_ERROR_SIGMA = 12.0  # desvio padrao do ruido no alvo (px)
AI_MISS_PROB = 0.10  # prob. de a IA nao se mover no frame
AI_CURRICULUM_STEP = 0.05  # quanto a dificuldade sobe por ponto
AI_MIN_SPEED_FACTOR = 0.5  # velocidade minima com dificuldade 0
AI_MAX_SIGMA_FACTOR = 1.0  # sigma com dificuldade 0
AI_MIN_SIGMA_FACTOR = 0.3  # sigma com dificuldade 1


class PongGame:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        self.ball_x = WIDTH / 2
        self.ball_y = HEIGHT / 2
        angle = self.rng.uniform(-0.4, 0.4)
        direction = self.rng.choice([-1, 1])
        self.ball_vx = direction * BALL_SPEED
        self.ball_vy = BALL_SPEED * angle
        self.paddle_left_y = HEIGHT / 2 - PADDLE_H / 2
        self.paddle_right_y = HEIGHT / 2 - PADDLE_H / 2
        self.score_left = 0
        self.score_right = 0

        # Buffer de reacao da IA: preenchido com a posicao atual da bola.
        # A IA consulta a posicao mais antiga do buffer (bola "vista" ha
        # AI_REACTION_FRAMES frames).
        self.ai_ball_history = [self.ball_y] * AI_REACTION_FRAMES

        # Dificuldade do curriculo: 0.0 = facil, 1.0 = dificil.
        self.ai_difficulty = 0.0

    def step(self, action):
        """action: -1 (sobe), 0 (parado), 1 (desce). Retorna dict de evento:
        {'bounce': bool, 'score': None|'left'|'right'}"""
        # --- Paddle esquerdo (agente) ---
        self.paddle_left_y += action * PADDLE_SPEED
        self.paddle_left_y = max(0, min(HEIGHT - PADDLE_H, self.paddle_left_y))

        # --- Paddle direito (IA) ---
        # 1) Atualiza o buffer de historico e obtem a posicao "percebida".
        self.ai_ball_history.append(self.ball_y)
        self.ai_ball_history.pop(0)
        perceived_ball_y = self.ai_ball_history[0]

        # 2) Aplica ruido e possibilidade de "piscar".
        if self.rng.random() < AI_MISS_PROB:
            pass  # IA nao se move neste frame
        else:
            sigma = AI_ERROR_SIGMA * (
                AI_MAX_SIGMA_FACTOR
                - (AI_MAX_SIGMA_FACTOR - AI_MIN_SIGMA_FACTOR) * self.ai_difficulty
            )
            speed_factor = (
                AI_MIN_SPEED_FACTOR + (1.0 - AI_MIN_SPEED_FACTOR) * self.ai_difficulty
            )
            effective_speed = AI_SPEED * speed_factor

            noisy_target = perceived_ball_y + self.rng.gauss(0, sigma)
            target = noisy_target - PADDLE_H / 2
            if target > self.paddle_right_y:
                self.paddle_right_y += effective_speed
            elif target < self.paddle_right_y:
                self.paddle_right_y -= effective_speed
        self.paddle_right_y = max(0, min(HEIGHT - PADDLE_H, self.paddle_right_y))

        # --- Fisica da bola ---
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        bounce = False
        score = None

        # Colisao com topo/base
        if self.ball_y <= 0 or self.ball_y >= HEIGHT - BALL_SIZE:
            self.ball_vy *= -1
            self.ball_y = max(0, min(HEIGHT - BALL_SIZE, self.ball_y))

        # Colisao com paddle esquerdo (agente)
        if self.ball_x <= PADDLE_W:
            if self.paddle_left_y <= self.ball_y <= self.paddle_left_y + PADDLE_H:
                self.ball_vx = min(-self.ball_vx * 1.05, BALL_SPEED_MAX)
                # *** PATCH 1: vy tambem cresce, com teto ***
                self.ball_vy = max(-BALL_VY_MAX, min(BALL_VY_MAX, self.ball_vy * 1.05))
                self.ball_x = PADDLE_W
                bounce = True
            elif self.ball_x < 0:
                self.score_right += 1
                score = "right"
                self.reset_ball(direction=1)

        # Colisao com paddle direito (IA)
        if self.ball_x >= WIDTH - PADDLE_W - BALL_SIZE:
            if self.paddle_right_y <= self.ball_y <= self.paddle_right_y + PADDLE_H:
                self.ball_vx = max(-self.ball_vx * 1.05, -BALL_SPEED_MAX)
                self.ball_x = WIDTH - PADDLE_W - BALL_SIZE
            elif self.ball_x > WIDTH:
                self.score_left += 1
                score = "left"
                # *** PATCH 4: sobe a dificuldade quando a mosca marca ***
                self.ai_difficulty = min(1.0, self.ai_difficulty + AI_CURRICULUM_STEP)
                self.reset_ball(direction=-1)

        return {"bounce": bounce, "score": score}

    def reset_ball(self, direction):
        self.ball_x = WIDTH / 2
        self.ball_y = HEIGHT / 2
        angle = self.rng.uniform(-0.4, 0.4)
        self.ball_vx = direction * BALL_SPEED
        self.ball_vy = BALL_SPEED * angle
        # Ao reposicionar a bola, reinicia o buffer para a IA nao
        # reagir a uma posicao antiga que nao existe mais.
        self.ai_ball_history = [self.ball_y] * AI_REACTION_FRAMES
