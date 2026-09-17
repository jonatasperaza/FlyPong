"""Regression tests for the causal-pruning on-disk graph invariant."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sim.network import ConnectomeNetwork


class CausalPruneTests(unittest.TestCase):
    def setUp(self):
        self.neurons = pd.DataFrame(
            [
                {"bodyId": 101, "type": "R1", "instance": "R1", "role": "photoreceptor"},
                {"bodyId": 102, "type": "Tm4", "instance": "Tm4", "role": "interneuron"},
                {"bodyId": 103, "type": "DNa10", "instance": "DNa10_R", "role": "descending"},
                {"bodyId": 104, "type": "PAM01", "instance": "PAM01", "role": "dopaminergic"},
                {"bodyId": 105, "type": "Other", "instance": "Other", "role": "interneuron"},
            ]
        )
        self.edges = pd.DataFrame(
            [
                {"pre": 101, "post": 102, "weight": 3},
                {"pre": 102, "post": 103, "weight": 4},
                {"pre": 105, "post": 105, "weight": 9},
            ]
        )

    def test_pruned_edges_keep_body_ids_and_network_connectivity(self):
        nodes, edges = ConnectomeNetwork._causal_prune(self.neurons, self.edges)

        self.assertSetEqual(set(nodes["bodyId"]), {101, 102, 103, 104})
        self.assertListEqual(edges["pre"].tolist(), [101, 102])
        self.assertListEqual(edges["post"].tolist(), [102, 103])

        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            nodes.to_parquet(data_dir / "neurons.parquet", index=False)
            edges.to_parquet(data_dir / "edges.parquet", index=False)
            network = ConnectomeNetwork(str(data_dir), prune_causal=True, record_history=False)

        self.assertEqual(network.n, 4)
        self.assertEqual(network.W.nnz, 2)
        self.assertEqual(network.W[network.body_id_to_idx[103], network.body_id_to_idx[102]], 1.2)


if __name__ == "__main__":
    unittest.main()
