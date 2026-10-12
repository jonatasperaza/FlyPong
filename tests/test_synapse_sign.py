"""synapse_sign="nt": sinapses de neuronios GABA/glutamato ficam negativas."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import fetch_connectome as fc
from sim.covariance_rule import HomeostaticScaler
from sim.network import ConnectomeNetwork


class SynapseSignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        d = cls.tmp.name
        neurons, edges = fc.fetch_synthetic(seed=0)
        rng = np.random.default_rng(0)
        neurons["nt"] = "acetylcholine"
        inter = neurons.index[neurons["role"] == "interneuron"]
        neurons.loc[rng.choice(inter, len(inter) // 3, replace=False), "nt"] = "gaba"
        neurons.to_parquet(os.path.join(d, "neurons.parquet"), index=False)
        edges.to_parquet(os.path.join(d, "edges.parquet"), index=False)
        with open(os.path.join(d, "metadata.json"), "w") as f:
            json.dump({"source": "synthetic"}, f)
        cls.plain = ConnectomeNetwork(d)
        cls.signed = ConnectomeNetwork(d, synapse_sign="nt")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_inhibitory_presynaptic_rows_are_negative(self):
        net = self.signed
        W = net.W.tocoo()
        inhib = np.isin(W.col, net.inhibitory_idx)
        self.assertTrue(inhib.any())
        self.assertTrue((W.data[inhib] < 0).all())
        self.assertTrue((W.data[~inhib] > 0).all())
        np.testing.assert_allclose(np.abs(net.W.toarray()), self.plain.W.toarray())

    def test_default_is_all_excitatory(self):
        self.assertTrue((self.plain.W.data > 0).all())
        self.assertEqual(len(self.plain.inhibitory_idx), 0)

    def test_plastic_synapses_are_excitatory(self):
        self.assertTrue((self.signed.W.data[self.signed.plastic_data_idx] > 0).all())

    def test_homeostasis_scales_only_excitatory_inputs(self):
        net = self.signed
        before = net.W.data.copy()
        idx = net.role_idx["interneuron"]
        sc = HomeostaticScaler(net, idx, rate=0.5, target_rate=0.0, mean_decay=0.0)
        sc.update(np.ones(net.n))
        changed = net.W.data != before
        self.assertTrue(changed.any())
        self.assertTrue((before[changed] > 0).all())
        net.W.data[:] = before


if __name__ == "__main__":
    unittest.main()
