"""Pong minimalista. O paddle esquerdo e controlado pelo conectoma (via
main.py); o paddle direito e uma IA trivial que segue a bola com atraso."""
import random

WIDTH, HEIGHT = 480, 320
PADDLE_H = 60
PADDLE_W = 10
BALL_SIZE = 8
PADDLE_SPEED = 4.0
AI_SPEED = 3.0
BALL_SPEED = 3.5
BALL_SPEED_MAX = 2.5 * BALL_SPEED  # sem teto, o fator 1.05 por rebatida faz a bola acelerar sem limite em runs longos


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

    def step(self, action):
        """action: -1 (sobe), 0 (parado), 1 (desce). Retorna dict de evento:
        {'bounce': bool, 'score': None|'left'|'right'}"""
        self.paddle_left_y += action * PADDLE_SPEED
        self.paddle_left_y = max(0, min(HEIGHT - PADDLE_H, self.paddle_left_y))

        target = self.ball_y - PADDLE_H / 2
        if target > self.paddle_right_y:
            self.paddle_right_y += AI_SPEED
        elif target < self.paddle_right_y:
            self.paddle_right_y -= AI_SPEED
        self.paddle_right_y = max(0, min(HEIGHT - PADDLE_H, self.paddle_right_y))

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
                self.reset_ball(direction=-1)

        return {"bounce": bounce, "score": score}

    def reset_ball(self, direction):
        self.ball_x = WIDTH / 2
        self.ball_y = HEIGHT / 2
        angle = self.rng.uniform(-0.4, 0.4)
        self.ball_vx = direction * BALL_SPEED
        self.ball_vy = BALL_SPEED * angle
