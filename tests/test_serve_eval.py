"""Tests for the independent-serve evaluation protocol (serve_eval.py)."""

from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import serve_eval
from game.pong import PongGame, WIDTH, HEIGHT, PADDLE_H


class WilsonIntervalTests(unittest.TestCase):
    def test_contains_point_estimate_and_stays_in_unit_interval(self):
        for hits, n in [(0, 10), (5, 10), (10, 10), (37, 300)]:
            lo, hi = serve_eval.wilson_interval(hits, n)
            self.assertLessEqual(0.0, lo)
            self.assertLessEqual(hi, 1.0)
            self.assertLessEqual(lo, hits / n)
            self.assertGreaterEqual(hi, hits / n)

    def test_known_value(self):
        lo, hi = serve_eval.wilson_interval(50, 100)
        self.assertAlmostEqual(lo, 0.4038, places=3)
        self.assertAlmostEqual(hi, 0.5962, places=3)


class ServeProtocolTests(unittest.TestCase):
    def test_serve_heads_to_connectome_paddle_from_center(self):
        game = PongGame(seed=0)
        rng = random.Random(7)
        for _ in range(50):
            serve_eval.serve(game, rng)
            self.assertEqual(game.ball_x, WIDTH / 2)
            self.assertEqual(game.ball_y, HEIGHT / 2)
            self.assertLess(game.ball_vx, 0.0)
            self.assertTrue(0.0 <= game.paddle_left_y <= HEIGHT - PADDLE_H)

    def test_serves_depend_only_on_eval_seed(self):
        a = serve_eval.run_serves(serve_eval.still_policy, 100, 1001)
        b = serve_eval.run_serves(serve_eval.still_policy, 100, 1001)
        c = serve_eval.run_serves(serve_eval.still_policy, 100, 1002)
        self.assertEqual(a, b)
        self.assertNotEqual(a["hits"], c["hits"])

    def test_oracle_beats_still_paddle(self):
        still = serve_eval.run_serves(serve_eval.still_policy, 200, 1001)
        oracle = serve_eval.run_serves(serve_eval.oracle_policy, 200, 1001)
        self.assertGreater(oracle["hit_rate"], 0.95)
        self.assertLess(still["hit_rate"], 0.5)
        self.assertLess(still["ci95_high"], oracle["ci95_low"])

    def test_train_and_eval_seeds_are_disjoint(self):
        self.assertFalse(set(serve_eval.TRAIN_SEEDS) & set(serve_eval.EVAL_SEEDS))


class InvertReadoutTests(unittest.TestCase):
    def test_swaps_groups_and_negates_baseline_and_is_involution(self):
        up = np.array([0, 1])
        down = np.array([2, 3])
        runner = SimpleNamespace(
            net=SimpleNamespace(motor_up_idx=up, motor_down_idx=down),
            motor_baseline_diff=0.25,
        )
        serve_eval.invert_readout(runner)
        self.assertIs(runner.net.motor_up_idx, down)
        self.assertIs(runner.net.motor_down_idx, up)
        self.assertEqual(runner.motor_baseline_diff, -0.25)

        serve_eval.invert_readout(runner)
        self.assertIs(runner.net.motor_up_idx, up)
        self.assertIs(runner.net.motor_down_idx, down)
        self.assertEqual(runner.motor_baseline_diff, 0.25)

    def test_inversion_flips_every_readout_decision(self):
        from game.motor_read import read_motor_action

        rng = np.random.default_rng(0)
        up, down, baseline = np.array([0, 1, 2]), np.array([3, 4, 5]), 0.3
        for _ in range(100):
            counts = rng.integers(0, 5, size=6).astype(float)
            normal = read_motor_action(counts, up, down, baseline_diff=baseline)
            inverted = read_motor_action(counts, down, up, baseline_diff=-baseline)
            self.assertEqual(inverted, -normal)


if __name__ == "__main__":
    unittest.main()
