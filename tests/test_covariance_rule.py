"""Tests for the reward-prediction-error covariance rule and egocentric retina."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sim.covariance_rule import CovarianceRPELearner
from game.sensory_map import egocentric_ball_y


def make_net(w=2.0):
    """Two pre neurons (0, 1) -> two post neurons (2, 3), all plastic."""
    pre = np.array([0, 1, 0, 1])
    post = np.array([2, 2, 3, 3])
    W = sp.csr_matrix((np.full(4, w), (post, pre)), shape=(4, 4))
    data_idx = np.array([W.indptr[po] + np.flatnonzero(W.indices[W.indptr[po]:W.indptr[po + 1]] == pr)[0]
                         for pr, po in zip(pre, post)])
    return SimpleNamespace(n=4, W=W, plastic_pre=pre, plastic_post=post,
                           plastic_data_idx=data_idx)


def weights(net):
    return net.W.data[net.plastic_data_idx].copy()


class PredictionErrorTests(unittest.TestCase):
    def test_constant_reward_gives_zero_change_once_predicted(self):
        net = make_net()
        learner = CovarianceRPELearner(net, eta=0.1)
        learner.reward_mean = 1.0  # recompensa ja prevista
        rng = np.random.default_rng(0)
        w0 = weights(net)
        for _ in range(200):
            counts = rng.integers(0, 3, size=4).astype(float)
            xi = rng.normal(size=4)
            learner.update(counts, 1.0, perturbation=xi)
        np.testing.assert_allclose(weights(net), w0)

    def test_reward_unrelated_to_activity_gives_no_systematic_drift(self):
        # Cada execucao e um passeio aleatorio; o teste e sobre a media de
        # muitas execucoes independentes (desvio padrao esperado ~0.015).
        rng = np.random.default_rng(1)
        drifts = []
        for _ in range(400):
            net = make_net(w=4.0)
            learner = CovarianceRPELearner(net, eta=0.01, w_cap_factor=10.0)
            w0 = weights(net)
            for _ in range(200):
                counts = rng.integers(0, 3, size=4).astype(float)
                learner.update(counts, float(rng.choice([-1.0, 1.0])),
                               perturbation=rng.normal(size=4))
            drifts.append(weights(net) - w0)
        self.assertLess(abs(np.mean(drifts)), 0.08)

    def test_reward_correlated_with_perturbation_strengthens_that_pathway(self):
        net = make_net()
        learner = CovarianceRPELearner(net, eta=0.01)
        rng = np.random.default_rng(2)
        counts = np.array([1.0, 0.0, 0.0, 0.0])  # so o pre 0 ativo
        for _ in range(2000):
            xi = rng.normal(size=4)
            learner.update(counts, 1.0 if xi[2] > 0 else -1.0, perturbation=xi)
        w = weights(net)  # ordem: (0->2), (1->2), (0->3), (1->3)
        self.assertGreater(w[0], 2.5)          # pre ativo -> post recompensado
        self.assertAlmostEqual(w[1], 2.0)      # pre inativo nao muda
        self.assertAlmostEqual(w[3], 2.0)
        self.assertLess(abs(w[2] - 2.0), abs(w[0] - 2.0))

    def test_covariance_mode_gives_opposite_signs_to_high_and_low_post(self):
        net = make_net()
        learner = CovarianceRPELearner(net, eta=0.01, post_mean_decay=0.0)
        learner.post_mean[:] = 1.0
        learner.reward_mean = 0.0
        learner.update(np.array([1.0, 0.0, 2.0, 0.0]), 1.0)
        w = weights(net)
        self.assertGreater(w[0], 2.0)  # post 2 disparou acima da media
        self.assertLess(w[2], 2.0)     # post 3 disparou abaixo da media

    def test_accumulate_without_apply_keeps_weights(self):
        net = make_net()
        learner = CovarianceRPELearner(net, eta=0.1, elig_decay=0.9)
        w0 = weights(net)
        learner.update(np.ones(4), 1.0, perturbation=np.ones(4), apply=False)
        np.testing.assert_allclose(weights(net), w0)
        self.assertTrue(np.all(learner.elig > 0))

    def test_weights_stay_within_bounds(self):
        net = make_net()
        learner = CovarianceRPELearner(net, eta=5.0, w_cap_factor=3.0)
        rng = np.random.default_rng(3)
        for _ in range(200):
            learner.update(np.ones(4), float(rng.normal() * 10), perturbation=rng.normal(size=4))
        w = weights(net)
        self.assertTrue(np.all(w >= 0.0))
        self.assertTrue(np.all(w <= 3.0 * 2.0 + 1e-9))


class HomeostasisTests(unittest.TestCase):
    def test_scales_down_inputs_of_neurons_above_target(self):
        net = make_net()
        learner = CovarianceRPELearner(net, eta=0.0, homeostasis_rate=0.1,
                                       target_rate=0.3, post_mean_decay=0.5)
        for _ in range(50):
            learner.update(np.array([1.0, 1.0, 1.0, 0.0]), 0.0)
        w = weights(net)
        self.assertLess(w[0], 2.0)     # post 2 acima do alvo
        self.assertGreater(w[2], 2.0)  # post 3 abaixo do alvo


class EgocentricRetinaTests(unittest.TestCase):
    def test_ball_level_with_paddle_hits_retina_center(self):
        self.assertEqual(egocentric_ball_y(100.0, 100.0, 320), 160.0)

    def test_sign_and_range(self):
        self.assertLess(egocentric_ball_y(50.0, 200.0, 320), 160.0)
        self.assertGreater(egocentric_ball_y(250.0, 100.0, 320), 160.0)
        for ball, pad in [(0, 320), (320, 0)]:
            self.assertTrue(0 <= egocentric_ball_y(ball, pad, 320) <= 320)


if __name__ == "__main__":
    unittest.main()
