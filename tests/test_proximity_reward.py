"""Unit tests for the opt-in ball-paddle proximity shaping signal."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import FlyPongRunner


class ProximityRewardTests(unittest.TestCase):
    def make_runner(self):
        runner = FlyPongRunner.__new__(FlyPongRunner)
        runner.game = SimpleNamespace(ball_y=100.0, paddle_left_y=60.0)
        runner.proximity_progress_deadzone = 0.5
        runner._previous_approach_distance = None
        return runner

    def test_only_rewards_progress_while_ball_approaches(self):
        runner = self.make_runner()

        self.assertEqual(runner._proximity_reward_signal(True), 0.0)
        runner.game.ball_y = 95.0
        self.assertEqual(runner._proximity_reward_signal(True), 1.0)
        runner.game.ball_y = 110.0
        self.assertEqual(runner._proximity_reward_signal(True), -1.0)

        self.assertEqual(runner._proximity_reward_signal(False), 0.0)
        self.assertIsNone(runner._previous_approach_distance)

    def test_deadzone_ignores_negligible_distance_change(self):
        runner = self.make_runner()
        self.assertEqual(runner._proximity_reward_signal(True), 0.0)
        runner.game.ball_y = 99.75
        self.assertEqual(runner._proximity_reward_signal(True), 0.0)

    def test_schedule_reward_preserves_requested_strength(self):
        runner = self.make_runner()
        runner._schedule_reward(-1.0, current=3.0)
        self.assertEqual(runner.reward_valence, -1.0)
        self.assertEqual(runner.reward_current, 3.0)
        self.assertGreater(runner.reward_pulse_remaining, 0)


if __name__ == "__main__":
    unittest.main()
