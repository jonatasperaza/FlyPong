"""Tests for the per-eye elevation retina axis (retina_axis="elevation")."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sim.network import ConnectomeNetwork


def write_graph(tmp: Path, with_y: bool = True):
    # Dois olhos separados no eixo PCA (como no MaleCNS real), com elevacao
    # (retina_y) variando dentro de cada olho.
    rows = []
    for i, (inst, pca, y) in enumerate([
        ("R1_L", -30.0, 300.0), ("R1_L", -31.0, 100.0), ("R1_L", -29.0, 200.0),
        ("R1_R", 30.0, 150.0), ("R1_R", 31.0, 50.0), ("R1_R", 29.0, 250.0),
    ]):
        row = {"bodyId": 100 + i, "type": "R1", "instance": inst,
               "role": "photoreceptor", "retina_pos": pca}
        if with_y:
            row["retina_y"] = y
        rows.append(row)
    rows.append({"bodyId": 200, "type": "DNa10", "instance": "DNa10_R",
                 "role": "descending", "retina_pos": np.nan,
                 **({"retina_y": np.nan} if with_y else {})})
    pd.DataFrame(rows).to_parquet(tmp / "neurons.parquet", index=False)
    pd.DataFrame([{"pre": 100, "post": 200, "weight": 1}]).to_parquet(
        tmp / "edges.parquet", index=False)


class RetinaAxisTests(unittest.TestCase):
    def test_pca_axis_splits_eyes_but_elevation_spans_each_eye(self):
        with tempfile.TemporaryDirectory() as d:
            write_graph(Path(d))
            pca = ConnectomeNetwork(d).photoreceptor_positions
            elev = ConnectomeNetwork(d, retina_axis="elevation").photoreceptor_positions
        # PCA: olho esquerdo todo abaixo de 0.1, direito todo acima de 0.9.
        self.assertTrue(np.all(pca[:3] < 0.1) and np.all(pca[3:] > 0.9))
        # Elevacao: cada olho cobre [0, 1] na ordem de retina_y.
        np.testing.assert_allclose(elev[:3], [1.0, 0.0, 0.5])
        np.testing.assert_allclose(elev[3:], [0.5, 0.0, 1.0])

    def test_elevation_requires_retina_y(self):
        with tempfile.TemporaryDirectory() as d:
            write_graph(Path(d), with_y=False)
            with self.assertRaises(ValueError):
                ConnectomeNetwork(d, retina_axis="elevation")


if __name__ == "__main__":
    unittest.main()
