"""Opponent strategy module for Etapa 3 — Estilos de Jogo.

Seven adversarial opponent styles to prevent overfitting and test
generalization. Each style implements a different behavioral pattern,
making the evolutionary pressure diverse and non-trivial.

Styles:
  DEFENSIVA    — stays near center, conservative, reacts late
  AGRESSIVA    — chases the ball aggressively, fast reactions
  RAPIDA       — very fast reaction, low noise, high speed
  IMPREVISIVEL — random movements, high noise, variable speed
  CONSERVADORA — stays near edges, rarely moves
  RUIDO_ALTO   — high noise, moderate speed, unpredictable
  TRAJETORIA_ESTRANHA — follows sine-wave-like patterns

Usage:
    from game.opponent import OpponentStyle, OpponentStrategy
    from game.pong import PongGame

    # Use a specific style
    game = PongGame(seed=42, opponent_style=OpponentStyle.AGRESSIVA)

    # Or use a custom strategy
    strategy = OpponentStrategy(defensive=True, reaction_delay=12)
    game = PongGame(seed=42, opponent_strategy=strategy)
"""
from __future__ import annotations

import enum
import math
import random

from game.constants import (
    WIDTH, HEIGHT, PADDLE_H, PADDLE_W,
    BALL_SIZE, PADDLE_SPEED, AI_SPEED, BALL_SPEED, BALL_VY_MAX,
)


class OpponentStyle(enum.Enum):
    """Predefined opponent styles."""
    DEFENSIVA = "defensiva"
    AGRESSIVA = "agressiva"
    RAPIDA = "rapida"
    IMPREVISIVEL = "imprevisivel"
    CONSERVADORA = "conservadora"
    RUIDO_ALTO = "ruído_alto"
    TRAJETORIA_ESTRANHA = "trajetória_estranha"


class OpponentStrategy:
    """Pluggable opponent AI strategy.

    Each strategy computes a paddle target position based on
    the current game state (ball position, history, etc.).
    """

    def __init__(
        self,
        style: OpponentStyle = OpponentStyle.DEFENSIVA,
        reaction_delay: int = 8,
        noise_sigma: float = 12.0,
        miss_prob: float = 0.10,
        speed_factor: float = 1.0,
        **kwargs,
    ):
        self.style = style
        self.reaction_delay = reaction_delay
        self.noise_sigma = noise_sigma
        self.miss_prob = miss_prob
        self.speed_factor = speed_factor
        self._rng = random.Random(hash(style.name))
        self._frame_count = 0
        self._ball_history: list[float] = []
        self._target_offset = 0.0
        self._sine_phase = 0.0

    def reset(self):
        """Reset the strategy's internal state."""
        self._frame_count = 0
        self._target_offset = 0.0
        self._sine_phase = 0.0
        self._ball_history = []
        self._rng = random.Random(hash(self.style.name + str(self._frame_count)))

    def compute_target(self, ball_y: float, ball_y_history: list[float], paddle_y: float) -> float:
        """Compute the target paddle position for the given ball state.

        Returns the desired Y position for the right paddle.
        """
        self._frame_count += 1

        if self.style == OpponentStyle.DEFENSIVA:
            return self._defensive(ball_y, ball_y_history, paddle_y)
        elif self.style == OpponentStyle.AGRESSIVA:
            return self._aggressive(ball_y, ball_y_history, paddle_y)
        elif self.style == OpponentStyle.RAPIDA:
            return self._fast(ball_y, ball_y_history, paddle_y)
        elif self.style == OpponentStyle.IMPREVISIVEL:
            return self._unpredictable(ball_y, ball_y_history, paddle_y)
        elif self.style == OpponentStyle.CONSERVADORA:
            return self._conservative(ball_y, ball_y_history, paddle_y)
        elif self.style == OpponentStyle.RUIDO_ALTO:
            return self._noisy(ball_y, ball_y_history, paddle_y)
        elif self.style == OpponentStyle.TRAJETORIA_ESTRANHA:
            return self._strange_trajectory(ball_y, ball_y_history, paddle_y)
        else:
            return self._defensive(ball_y, ball_y_history, paddle_y)

    def get_perceived_ball_y(self, ball_y: float, history: list[float]) -> float:
        """Get the delayed ball position from the history buffer."""
        if not history:
            return ball_y
        idx = min(self.reaction_delay, len(history) - 1)
        return history[-1 - idx] if idx < len(history) else ball_y

    def _should_miss(self) -> bool:
        """Whether the opponent misses this frame (simulating human error)."""
        return self._rng.random() < self.miss_prob

    def _add_noise(self, target: float) -> float:
        """Add Gaussian noise to the target."""
        return target + self._rng.gauss(0, self.noise_sigma)

    def _clamp_paddle(self, target: float) -> float:
        """Clamp target to valid paddle range."""
        return max(0, min(HEIGHT - PADDLE_H, target))

    def _move_toward(self, current: float, target: float) -> float:
        """Move paddle toward target with the strategy's speed."""
        effective_speed = AI_SPEED * self.speed_factor
        if target > current:
            return min(current + effective_speed, HEIGHT - PADDLE_H)
        elif target < current:
            return max(current - effective_speed, 0)
        return current

    # --- Individual style implementations ---

    def _defensive(self, ball_y: float, history: list[float], paddle_y: float) -> float:
        """Defensive: stays near center, reacts late, low speed."""
        perceived = self.get_perceived_ball_y(ball_y, history)
        target = perceived - PADDLE_H / 2 + self._target_offset
        target = self._add_noise(target)
        return self._clamp_paddle(target)

    def _aggressive(self, ball_y: float, history: list[float], paddle_y: float) -> float:
        """Aggressive: tracks the ball immediately, high speed, no noise."""
        perceived = self.get_perceived_ball_y(ball_y, history)
        target = perceived - PADDLE_H / 2
        # Move faster than normal
        effective_speed = AI_SPEED * self.speed_factor * 1.5
        if target > paddle_y:
            return min(paddle_y + effective_speed, HEIGHT - PADDLE_H)
        elif target < paddle_y:
            return max(paddle_y - effective_speed, 0)
        return paddle_y

    def _fast(self, ball_y: float, history: list[float], paddle_y: float) -> float:
        """Fast: very low reaction delay, minimal noise, high speed."""
        perceived = self.get_perceived_ball_y(ball_y, history)
        target = perceived - PADDLE_H / 2
        target = self._add_noise(target * 0.1)  # very little noise
        return self._clamp_paddle(target)

    def _unpredictable(self, ball_y: float, history: list[float], paddle_y: float) -> float:
        """Unpredictable: random jumps, high noise, variable speed."""
        # Occasionally make a random large jump
        if self._rng.random() < 0.2:
            self._target_offset = self._rng.uniform(-PADDLE_H * 2, PADDLE_H * 2)
        perceived = self.get_perceived_ball_y(ball_y, history)
        target = perceived - PADDLE_H / 2 + self._target_offset
        target = self._add_noise(target)
        speed_mult = self._rng.uniform(0.3, 2.0)
        effective_speed = AI_SPEED * speed_mult
        if target > paddle_y:
            return min(paddle_y + effective_speed, HEIGHT - PADDLE_H)
        elif target < paddle_y:
            return max(paddle_y - effective_speed, 0)
        return paddle_y

    def _conservative(self, ball_y: float, history: list[float], paddle_y: float) -> float:
        """Conservative: stays near edges, rarely moves, only reacts to close balls."""
        perceived = self.get_perceived_ball_y(ball_y, history)
        # Only move if the ball is close to the paddle
        if abs(perceived - paddle_y - PADDLE_H / 2) > 50:
            return paddle_y  # Stay put
        target = perceived - PADDLE_H / 2
        target = self._add_noise(target * 0.5)
        return self._clamp_paddle(target)

    def _noisy(self, ball_y: float, history: list[float], paddle_y: float) -> float:
        """High noise: lots of randomness, moderate speed."""
        perceived = self.get_perceived_ball_y(ball_y, history)
        # Random target offset that changes frequently
        if self._frame_count % 10 == 0:
            self._target_offset = self._rng.gauss(0, PADDLE_H * 2)
        target = perceived - PADDLE_H / 2 + self._target_offset
        # High noise
        target = self._add_noise(target * 2.0)
        return self._clamp_paddle(target)

    def _strange_trajectory(self, ball_y: float, history: list[float], paddle_y: float) -> float:
        """Strange trajectory: follows a sine-wave-like pattern."""
        self._sine_phase += 0.1
        sine_offset = math.sin(self._sine_phase) * PADDLE_H * 2
        # Also track the ball but with the sine offset
        perceived = self.get_perceived_ball_y(ball_y, history)
        target = perceived - PADDLE_H / 2 + sine_offset + self._target_offset
        # The target oscillates independently of the ball sometimes
        if self._rng.random() < 0.15:
            self._target_offset = self._rng.gauss(0, PADDLE_H)
        target = self._add_noise(target)
        return self._clamp_paddle(target)


class OpponentStyleConfig:
    """Configuration for each opponent style."""
    CONFIGS = {
        OpponentStyle.DEFENSIVA: {
            "reaction_delay": 15,
            "noise_sigma": 8.0,
            "miss_prob": 0.15,
            "speed_factor": 0.6,
        },
        OpponentStyle.AGRESSIVA: {
            "reaction_delay": 3,
            "noise_sigma": 2.0,
            "miss_prob": 0.02,
            "speed_factor": 1.5,
        },
        OpponentStyle.RAPIDA: {
            "reaction_delay": 2,
            "noise_sigma": 3.0,
            "miss_prob": 0.05,
            "speed_factor": 1.8,
        },
        OpponentStyle.IMPREVISIVEL: {
            "reaction_delay": 5,
            "noise_sigma": 20.0,
            "miss_prob": 0.20,
            "speed_factor": 1.0,
        },
        OpponentStyle.CONSERVADORA: {
            "reaction_delay": 10,
            "noise_sigma": 5.0,
            "miss_prob": 0.05,
            "speed_factor": 0.3,
        },
        OpponentStyle.RUIDO_ALTO: {
            "reaction_delay": 8,
            "noise_sigma": 30.0,
            "miss_prob": 0.15,
            "speed_factor": 0.8,
        },
        OpponentStyle.TRAJETORIA_ESTRANHA: {
            "reaction_delay": 6,
            "noise_sigma": 10.0,
            "miss_prob": 0.10,
            "speed_factor": 1.0,
        },
    }

    @classmethod
    def get_config(cls, style: OpponentStyle) -> dict:
        return cls.CONFIGS.get(style, cls.CONFIGS[OpponentStyle.DEFENSIVA]).copy()


def create_opponent_strategy(style: OpponentStyle) -> OpponentStrategy:
    """Factory function: create an OpponentStrategy for the given style."""
    config = OpponentStyleConfig.get_config(style)
    return OpponentStrategy(style=style, **config)


def create_opponent_strategies() -> dict[OpponentStyle, OpponentStrategy]:
    """Create all 7 opponent strategies."""
    return {style: create_opponent_strategy(style) for style in OpponentStyle}
