"""Ruido antissimetrico: +xi no lado direito, -xi no esquerdo, por tipo."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import serve_train as st


def _net(types):
    return SimpleNamespace(neurons_df=pd.DataFrame({"type": types}))


class AntisymmetricPairsTests(unittest.TestCase):
    def test_pairs_by_type(self):
        net = _net(["DNa10", "DNp01", "DNp01", "DNa10"])
        net.motor_up_idx = np.array([0, 1])
        net.motor_down_idx = np.array([2, 3])
        r, l = st.antisymmetric_pairs(net)
        self.assertEqual(r.tolist(), [0, 1])
        self.assertEqual(l.tolist(), [3, 2])

    def test_unpaired_type_is_rejected(self):
        net = _net(["DNa10", "DNp01", "DNa10"])
        net.motor_up_idx = np.array([0, 1])
        net.motor_down_idx = np.array([2])
        with self.assertRaises(ValueError):
            st.antisymmetric_pairs(net)

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            st.train(None, None, n_serves=1, train_seed=0, noise_std=1.0,
                     reward_mode="dense", noise_mode="bogus")


if __name__ == "__main__":
    unittest.main()
