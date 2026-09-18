"""Pong minimalista. O paddle esquerdo e controlado pelo conectoma (via
main.py); o paddle direito e controlado por uma estrategia adversaria
opcional (game.opponent.OpponentStrategy).

Se nenhum adversario for fornecido, usa-se a IA trivial original.
"""

import random

from game.constants import *
from game.opponent import OpponentStrategy


class PongGame:
    def __init__(self, seed=None, opponent_strategy: OpponentStrategy | None = None):
        self.opponent_strategy = opponent_strategy
        self._use_default_ai = opponent_strategy is None
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
        self.ai_ball_history = [self.ball_y] * AI_REACTION_FRAMES
        self.ai_difficulty = 0.0
        if self.opponent_strategy is not None:
            self.opponent_strategy.reset()

    def _update_ai_paddle(self):
        """Update the right paddle using either the default AI or a pluggable strategy."""
        self.ai_ball_history.append(self.ball_y)
        self.ai_ball_history.pop(0)
        perceived_ball_y = self.ai_ball_history[0]

        if self._use_default_ai:
            self._update_ai_default(perceived_ball_y)
        elif self.opponent_strategy is not None:
            self._update_ai_strategy(perceived_ball_y)

    def _update_ai_default(self, perceived_ball_y: float):
        """Original AI logic (backward compatible)."""
        if self.rng.random() < AI_MISS_PROB:
            pass
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

    def _update_ai_strategy(self, perceived_ball_y: float):
        """Use the pluggable opponent strategy."""
        target = self.opponent_strategy.compute_target(
            ball_y=self.ball_y,
            ball_y_history=self.ai_ball_history,
            paddle_y=self.paddle_right_y,
        )
        effective_speed = AI_SPEED * self.opponent_strategy.speed_factor
        if target > self.paddle_right_y:
            self.paddle_right_y = min(self.paddle_right_y + effective_speed, HEIGHT - PADDLE_H)
        elif target < self.paddle_right_y:
            self.paddle_right_y = max(self.paddle_right_y - effective_speed, 0)

    def step(self, action):
        """action: -1 (sobe), 0 (parado), 1 (desce). Retorna dict de evento:
        {'bounce': bool, 'score': None|'left'|'right'}"""
        self.paddle_left_y += action * PADDLE_SPEED
        self.paddle_left_y = max(0, min(HEIGHT - PADDLE_H, self.paddle_left_y))

        self._update_ai_paddle()

        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        bounce = False
        score = None

        if self.ball_y <= 0 or self.ball_y >= HEIGHT - BALL_SIZE:
            self.ball_vy *= -1
            self.ball_y = max(0, min(HEIGHT - BALL_SIZE, self.ball_y))

        if self.ball_x <= PADDLE_W:
            if self.paddle_left_y <= self.ball_y <= self.paddle_left_y + PADDLE_H:
                self.ball_vx = min(-self.ball_vx * 1.05, BALL_SPEED_MAX)
                self.ball_vy = max(-BALL_VY_MAX, min(BALL_VY_MAX, self.ball_vy * 1.05))
                self.ball_x = PADDLE_W
                bounce = True
            elif self.ball_x < 0:
                self.score_right += 1
                score = "right"
                self.reset_ball(direction=1)

        if self.ball_x >= WIDTH - PADDLE_W - BALL_SIZE:
            if self.paddle_right_y <= self.ball_y <= self.paddle_right_y + PADDLE_H:
                self.ball_vx = max(-self.ball_vx * 1.05, -BALL_SPEED_MAX)
                self.ball_x = WIDTH - PADDLE_W - BALL_SIZE
            elif self.ball_x > WIDTH:
                self.score_left += 1
                score = "left"
                self.ai_difficulty = min(1.0, self.ai_difficulty + AI_CURRICULUM_STEP)
                self.reset_ball(direction=-1)

        return {"bounce": bounce, "score": score}

    def reset_ball(self, direction):
        self.ball_x = WIDTH / 2
        self.ball_y = HEIGHT / 2
        angle = self.rng.uniform(-0.4, 0.4)
        self.ball_vx = direction * BALL_SPEED
        self.ball_vy = BALL_SPEED * angle
        self.ai_ball_history = [self.ball_y] * AI_REACTION_FRAMES
        self.ai_difficulty = 0.0