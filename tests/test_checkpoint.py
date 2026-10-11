"""Retomar de checkpoint deve dar resultado identico a rodar sem interrupcao."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import fetch_connectome as fc
import serve_train as st


class _Crash(Exception):
    pass


def _write_synthetic(d: str):
    neurons, edges = fc.fetch_synthetic(seed=0)
    neurons.to_parquet(os.path.join(d, "neurons.parquet"), index=False)
    edges.to_parquet(os.path.join(d, "edges.parquet"), index=False)
    with open(os.path.join(d, "metadata.json"), "w") as f:
        json.dump({"source": "synthetic"}, f)


def _comparable(row):
    keys = ("hits", "hit_rate", "train_hit_rate", "curve", "weights")
    return {k: row[k] for k in keys}


class CheckpointResumeTests(unittest.TestCase):
    KW = dict(condition="inverted", policy="trained", train_seed=7, eval_seed=2001,
              eval_trials=30, n_serves=60, dev_serves=40, curve_every=20,
              curve_trials=15, verbose=False, homeostasis_scope="all_visual",
              checkpoint_every=10)

    def test_resume_after_crash_matches_uninterrupted_run(self):
        with tempfile.TemporaryDirectory() as d:
            _write_synthetic(d)
            reference = st.run_condition(d, **self.KW)

            ck = os.path.join(d, "ck", "job.pkl")
            real_save = st._save_checkpoint
            calls = {"n": 0}

            def crashing_save(path, obj):
                real_save(path, obj)
                calls["n"] += 1
                if calls["n"] in (3, 7):  # uma queda no desenvolvimento, outra no treino
                    raise _Crash()

            st._save_checkpoint = crashing_save
            try:
                with self.assertRaises(_Crash):
                    st.run_condition(d, checkpoint_path=ck, **self.KW)
                with self.assertRaises(_Crash):
                    st.run_condition(d, checkpoint_path=ck, **self.KW)
                resumed = st.run_condition(d, checkpoint_path=ck, **self.KW)
            finally:
                st._save_checkpoint = real_save

        self.assertEqual(_comparable(reference), _comparable(resumed))


if __name__ == "__main__":
    unittest.main()
