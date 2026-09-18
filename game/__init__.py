"""Game module — Pong, opponent strategies, and neural interfaces."""

from game.constants import *
from game.pong import PongGame
from game.opponent import OpponentStyle, OpponentStrategy, OpponentStyleConfig, create_opponent_strategy
from game.sensory_map import ball_to_photoreceptor_stimulus
from game.motor_read import read_motor_action
from game.baseline_policy import *

__all__ = [
    "WIDTH", "HEIGHT", "PADDLE_H", "PADDLE_W", "BALL_SIZE",
    "PADDLE_SPEED", "AI_SPEED", "BALL_SPEED", "BALL_SPEED_MAX", "BALL_VY_MAX",
    "PongGame", "OpponentStyle", "OpponentStrategy", "OpponentStyleConfig",
    "create_opponent_strategy", "ball_to_photoreceptor_stimulus",
    "read_motor_action",
]
