"""Two-brain Pong: both paddles are agent-controlled, for fly-vs-fly duels.

This is a sibling of game.pong.PongGame, not a subclass or an edit of it.
game/pong.py stays exactly as it is (Achado 18 and the rest of the project
depend on its current fixed-AI-opponent behavior); this module imports its
shared constants so the two never drift out of sync on court geometry or
ball speed, but reimplements `step` to take two actions instead of driving
the right paddle with the built-in AI.

Score/collision semantics are otherwise identical to PongGame: "left"
scoring means the right paddle missed, "right" scoring means the left
paddle missed, `bounce` fires only for the fly-controlled paddle in the
single-brain game -- here it fires per side, since both are fly-controlled.
"""
from __future__ import annotations

import random

from game.pong import (
    WIDTH,
    HEIGHT,
    PADDLE_H,
    PADDLE_W,
    BALL_SIZE,
    PADDLE_SPEED,
    BALL_SPEED,
    BALL_SPEED_MAX,
    BALL_VY_MAX,
)


class DuelPongGame:
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

    def reset_ball(self, direction):
        self.ball_x = WIDTH / 2
        self.ball_y = HEIGHT / 2
        angle = self.rng.uniform(-0.4, 0.4)
        self.ball_vx = direction * BALL_SPEED
        self.ball_vy = BALL_SPEED * angle

    def step(self, action_left: int, action_right: int) -> dict:
        """action_*: -1 (up), 0 (still), 1 (down) for each fly's paddle.

        Returns {'bounce_left': bool, 'bounce_right': bool,
                 'score': None|'left'|'right'} -- 'left'/'right' names the
        SCORING side, mirroring PongGame's convention (the side that just
        conceded reset the ball).
        """
        self.paddle_left_y += action_left * PADDLE_SPEED
        self.paddle_left_y = max(0, min(HEIGHT - PADDLE_H, self.paddle_left_y))
        self.paddle_right_y += action_right * PADDLE_SPEED
        self.paddle_right_y = max(0, min(HEIGHT - PADDLE_H, self.paddle_right_y))

        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        bounce_left = False
        bounce_right = False
        score = None

        if self.ball_y <= 0 or self.ball_y >= HEIGHT - BALL_SIZE:
            self.ball_vy *= -1
            self.ball_y = max(0, min(HEIGHT - BALL_SIZE, self.ball_y))

        if self.ball_x <= PADDLE_W:
            if self.paddle_left_y <= self.ball_y <= self.paddle_left_y + PADDLE_H:
                self.ball_vx = min(-self.ball_vx * 1.05, BALL_SPEED_MAX)
                self.ball_vy = max(-BALL_VY_MAX, min(BALL_VY_MAX, self.ball_vy * 1.05))
                self.ball_x = PADDLE_W
                bounce_left = True
            elif self.ball_x < 0:
                self.score_right += 1
                score = "right"
                self.reset_ball(direction=1)

        if self.ball_x >= WIDTH - PADDLE_W - BALL_SIZE:
            if self.paddle_right_y <= self.ball_y <= self.paddle_right_y + PADDLE_H:
                self.ball_vx = max(-self.ball_vx * 1.05, -BALL_SPEED_MAX)
                self.ball_vy = max(-BALL_VY_MAX, min(BALL_VY_MAX, self.ball_vy * 1.05))
                self.ball_x = WIDTH - PADDLE_W - BALL_SIZE
                bounce_right = True
            elif self.ball_x > WIDTH:
                self.score_left += 1
                score = "left"
                self.reset_ball(direction=-1)

        return {"bounce_left": bounce_left, "bounce_right": bounce_right, "score": score}
