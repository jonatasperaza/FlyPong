"""Connectome simulation with an optional causal-pruned fast path.

The default path is intentionally compatible with the original FlyPong model.
The optimized path can prune nodes that cannot causally influence the motor
readout, disable per-frame history, and reuse hot-path buffers.
"""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd
import scipy.sparse as sp

from sim.neurons import LIFPopulation

SYNAPSE_GAIN = 0.3
DOPAMINE_DECAY = 0.90
TRACE_DECAY = 0.85
PLASTIC_LR = 0.02
PLASTIC_DELTA_MAX = 3.0
PLASTICITY_MODE = "original"
FREQ_EMA_DECAY = 0.999
FREQ_NORM_MAX_SCALE = 10.0
FREQ_NORM_EPS = 1e-4
TONIC_BIAS = 0.05


class ConnectomeNetwork:
    """Load and simulate the selected connectome subgraph.

    Parameters added for performance work:
      prune_causal: keep only nodes that can lie on a causal path from an
                    externally stimulated source to a descending readout,
                    while always retaining photoreceptors, dopaminergic
                    neurons and descending neurons used by the readout.
      record_history: disable weight-history allocation during screening.
    """

    def __init__(self, data_dir: str, *, prune_causal: bool = False, record_history: bool = True,
                 retina_axis: str = "pca"):
        neurons_path = os.path.join(data_dir, "neurons.parquet")
        edges_path = os.path.join(data_dir, "edges.parquet")
        neurons_df = pd.read_parquet(neurons_path)
        edges_df = pd.read_parquet(edges_path)

        if prune_causal:
            neurons_df, edges_df = self._causal_prune(neurons_df, edges_df)

        self.neurons_df = neurons_df.reset_index(drop=True)
        self.edges_df = edges_df.reset_index(drop=True)
        self.record_history = record_history

        self.n = len(self.neurons_df)
        self.body_id_to_idx = {bid: i for i, bid in enumerate(self.neurons_df["bodyId"])}
        self.role_idx = {
            role: self.neurons_df.index[self.neurons_df["role"] == role].to_numpy(dtype=np.int64)
            for role in self.neurons_df["role"].unique()
        }

        self.photoreceptor_idx = self.role_idx.get("photoreceptor", np.empty(0, dtype=np.int64))
        self.motion_idx = self.role_idx.get("motion", np.empty(0, dtype=np.int64))
        self.object_idx = self.role_idx.get("object", np.empty(0, dtype=np.int64))
        self.target_idx = self.role_idx.get("target", np.empty(0, dtype=np.int64))
        self.descending_idx = self.role_idx.get("descending", np.empty(0, dtype=np.int64))
        self.dopaminergic_idx = self.role_idx.get("dopaminergic", np.empty(0, dtype=np.int64))

        desc_types = self.neurons_df["type"].astype(str).to_numpy()
        is_dna10 = np.array([t.startswith("DNa10") for t in desc_types[self.descending_idx]])
        self.dna10_idx = self.descending_idx[is_dna10]
        if retina_axis not in {"pca", "elevation"}:
            raise ValueError(f"unknown retina_axis: {retina_axis}")
        self.retina_axis = retina_axis
        self.photoreceptor_positions = self._compute_photoreceptor_positions()
        self.motor_up_idx, self.motor_down_idx = self._compute_motor_groups()

        types = self.neurons_df["type"].astype(str)
        is_pam = types.str.startswith("PAM")
        dop_set = set(self.dopaminergic_idx.tolist())
        if is_pam.any():
            pam_idx = self.neurons_df.index[is_pam].to_numpy(dtype=np.int64)
            pam_idx = np.array([i for i in pam_idx if i in dop_set], dtype=np.int64)
            pam_set = set(pam_idx.tolist())
            ppl1_idx = np.array([i for i in self.dopaminergic_idx if i not in pam_set], dtype=np.int64)
        else:
            half = len(self.dopaminergic_idx) // 2
            pam_idx = self.dopaminergic_idx[:half]
            ppl1_idx = self.dopaminergic_idx[half:]
        self.dopamine_positive_idx = pam_idx
        self.dopamine_negative_idx = ppl1_idx

        # `pre`/`post` are always bodyIds in the on-disk format, including
        # after causal pruning. Keeping that invariant prevents a pruned graph
        # from silently losing all of its edges when it is mapped to indices.
        pre = self.edges_df["pre"].map(self.body_id_to_idx).to_numpy()
        post = self.edges_df["post"].map(self.body_id_to_idx).to_numpy()
        weight = self.edges_df["weight"].to_numpy(dtype=np.float64) * SYNAPSE_GAIN
        valid = ~(np.isnan(pre) | np.isnan(post))
        if not valid.all():
            bad_edges = int((~valid).sum())
            raise ValueError(
                f"{bad_edges} edge(s) reference bodyIds absent from neurons.parquet"
            )
        pre = pre[valid].astype(np.int64)
        post = post[valid].astype(np.int64)
        weight = weight[valid]

        self.W = sp.csr_matrix((weight, (post, pre)), shape=(self.n, self.n))

        upstream_of_descending = (
            set(self.motion_idx.tolist())
            | set(self.object_idx.tolist())
            | set(self.target_idx.tolist())
        )
        descending_set = set(self.descending_idx.tolist())
        plastic_mask = np.fromiter(
            (pr in upstream_of_descending and po in descending_set for pr, po in zip(pre, post)),
            dtype=bool,
            count=len(pre),
        )
        self.plastic_pre = pre[plastic_mask]
        self.plastic_post = post[plastic_mask]
        self.plastic_data_idx = self._find_data_indices(self.plastic_pre, self.plastic_post)
        self.plastic_w_init = self.W.data[self.plastic_data_idx].copy()
        self.plastic_delta = np.zeros(len(self.plastic_data_idx), dtype=np.float64)
        self.plastic_weight_history: list[float] = []

        self.lif = LIFPopulation(self.n)
        self.dopamine_level = 0.0
        self.pre_trace = np.zeros(self.n, dtype=np.float64)
        self.post_trace = np.zeros(self.n, dtype=np.float64)
        self.last_spikes = np.zeros(self.n, dtype=bool)
        self.last_spikes_float = np.zeros(self.n, dtype=np.float64)
        self.current = np.zeros(self.n, dtype=np.float64)
        self.pos_rate_ema = 0.0
        self.neg_rate_ema = 0.0
        self.plasticity_scale_history: list[tuple[float, int]] = []

        self._scratch_external = np.zeros(self.n, dtype=np.float64)

    @staticmethod
    def _causal_prune(neurons_df: pd.DataFrame, edges_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Exact graph-level causal pruning for the current FlyPong interface.

        External sources are photoreceptors and dopaminergic neurons. Synthetic
        experiments may inject directly into DNa10/descending; retaining all
        descending neurons handles that case without inventing new semantics.
        Nodes are retained when they are either part of an explicit I/O role or
        lie on a source->descending path.
        """
        ids = neurons_df["bodyId"].to_numpy()
        pos = {bid: i for i, bid in enumerate(ids)}
        pre = edges_df["pre"].map(pos).to_numpy()
        post = edges_df["post"].map(pos).to_numpy()
        valid = ~(np.isnan(pre) | np.isnan(post))
        pre = pre[valid].astype(np.int64)
        post = post[valid].astype(np.int64)
        n = len(neurons_df)

        outgoing = [[] for _ in range(n)]
        incoming = [[] for _ in range(n)]
        for a, b in zip(pre, post):
            outgoing[a].append(b)
            incoming[b].append(a)

        def bfs(seeds, graph):
            seen = np.zeros(n, dtype=bool)
            stack = list(map(int, seeds))
            for s in stack:
                if 0 <= s < n:
                    seen[s] = True
            while stack:
                u = stack.pop()
                for v in graph[u]:
                    if not seen[v]:
                        seen[v] = True
                        stack.append(v)
            return seen

        roles = neurons_df["role"].astype(str).to_numpy()
        source = np.flatnonzero(np.isin(roles, ["photoreceptor", "dopaminergic"]))
        target = np.flatnonzero(roles == "descending")
        forward = bfs(source, outgoing)
        backward = bfs(target, incoming)

        keep = forward & backward
        keep |= np.isin(roles, ["photoreceptor", "dopaminergic", "descending"])

        # Preserve targets even when disconnected; the motor readout semantics
        # depend on the full descending pool.
        keep_idx = np.flatnonzero(keep)
        if len(keep_idx) == n:
            return neurons_df, edges_df

        keep_set = set(keep_idx.tolist())
        edge_mask = np.fromiter(
            (a in keep_set and b in keep_set for a, b in zip(pre, post)),
            dtype=bool,
            count=len(pre),
        )

        kept_edge_df = edges_df.iloc[np.flatnonzero(valid)[edge_mask]].copy()
        kept_nodes = neurons_df.iloc[keep_idx].copy().reset_index(drop=True)
        # Preserve the public edge schema: `pre` and `post` are bodyIds, not
        # row positions. __init__ owns the single bodyId -> local-index mapping.
        return kept_nodes, kept_edge_df.reset_index(drop=True)

    def _compute_motor_groups(self):
        n = len(self.descending_idx)
        if n == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        instances = self.neurons_df["instance"].astype(str).to_numpy()[self.descending_idx]
        is_right = np.array([bool(re.search(r"(_R$|\(R\))", inst)) for inst in instances])
        is_left = np.array([bool(re.search(r"(_L$|\(L\))", inst)) for inst in instances])
        if int((is_right | is_left).sum()) == n and n >= 2:
            return self.descending_idx[is_right], self.descending_idx[is_left]
        if n > 4:
            import warnings
            warnings.warn(
                f"{n - int((is_right | is_left).sum())} descendente(s) sem sufixo _L/_R; "
                "readout dividido pela metade dos indices (arbitrario)",
                stacklevel=2,
            )
        half = n // 2
        return self.descending_idx[:half], self.descending_idx[half:]

    def _compute_photoreceptor_positions(self):
        n = len(self.photoreceptor_idx)
        if n == 0:
            return np.empty(0, dtype=np.float64)
        if self.retina_axis == "elevation":
            return self._elevation_positions()
        if "retina_pos" in self.neurons_df.columns:
            raw = self.neurons_df["retina_pos"].to_numpy(dtype=float)[self.photoreceptor_idx]
        else:
            raw = np.full(n, np.nan)
        valid = ~np.isnan(raw)
        if valid.sum() < 2:
            return np.linspace(0.0, 1.0, n)
        median_val = np.median(raw[valid])
        raw = np.where(valid, raw, median_val)
        lo, hi = raw.min(), raw.max()
        return (raw - lo) / max(hi - lo, 1e-9)

    def _elevation_positions(self):
        """Posicao na retina 1D = elevacao, normalizada dentro de cada olho.

        No MaleCNS real, o 1o componente principal dos centroides de sinapse
        (`retina_pos`, modo "pca") separa o olho esquerdo do direito: metade
        dos fotorreceptores cai em [0, 0.11], a outra em [0.79, 1], e a faixa
        do meio fica sem nenhum. Aqui cada fotorreceptor recebe o posto (rank)
        do seu centroide no eixo y do neuPrint (`retina_y`, dorso-ventral no
        sistema de coordenadas do FlyEM) dentro do proprio olho (sufixo _L/_R
        do `instance`), entao os dois olhos veem a bola em qualquer altura.
        Exige a coluna `retina_y` (fetch_connectome.py atual ou
        scripts/add_retina_coords.py).
        """
        if "retina_y" not in self.neurons_df.columns:
            raise ValueError(
                "retina_axis='elevation' exige a coluna retina_y em neurons.parquet "
                "(rode scripts/add_retina_coords.py ou fetch_connectome.py de novo)"
            )
        idx = self.photoreceptor_idx
        y = self.neurons_df["retina_y"].to_numpy(dtype=float)[idx]
        inst = self.neurons_df["instance"].astype(str).to_numpy()[idx]
        eye = np.array(
            ["R" if re.search(r"(_R$|\(R\))", i) else "L" if re.search(r"(_L$|\(L\))", i) else "?"
             for i in inst]
        )
        pos = np.full(len(idx), 0.5)
        for e in np.unique(eye):
            m = (eye == e) & ~np.isnan(y)
            k = int(m.sum())
            if k >= 2:
                ranks = np.argsort(np.argsort(y[m]))
                pos[m] = ranks / (k - 1)
        return pos

    def _find_data_indices(self, pre_arr, post_arr):
        idx = np.empty(len(pre_arr), dtype=np.int64)
        indptr, indices = self.W.indptr, self.W.indices
        for k, (pr, po) in enumerate(zip(pre_arr, post_arr)):
            start, end = indptr[po], indptr[po + 1]
            hits = np.flatnonzero(indices[start:end] == pr)
            if len(hits) == 0:
                raise RuntimeError(f"Plastic synapse ({pr}, {po}) not found in CSR matrix")
            idx[k] = start + hits[0]
        return idx

    def reset(self):
        self.lif.reset()
        self.dopamine_level = 0.0
        self.pre_trace.fill(0.0)
        self.post_trace.fill(0.0)
        self.last_spikes.fill(False)
        self.last_spikes_float.fill(0.0)
        self.pos_rate_ema = 0.0
        self.neg_rate_ema = 0.0

    def step(self, external_current):
        # Avoid a new bool->float allocation in the hot SpMV path.
        np.copyto(self.last_spikes_float, self.last_spikes)
        np.copyto(self.current, self.W.dot(self.last_spikes_float))
        self.current += external_current
        spikes = self.lif.step(self.current)

        self.pre_trace *= TRACE_DECAY
        self.post_trace *= TRACE_DECAY
        self.pre_trace[spikes] += 1.0
        self.post_trace[spikes] += 1.0

        pos_rate = spikes[self.dopamine_positive_idx].mean() if len(self.dopamine_positive_idx) else 0.0
        neg_rate = spikes[self.dopamine_negative_idx].mean() if len(self.dopamine_negative_idx) else 0.0
        self.dopamine_level = self.dopamine_level * DOPAMINE_DECAY + (pos_rate - neg_rate)
        self.pos_rate_ema = self.pos_rate_ema * FREQ_EMA_DECAY + pos_rate * (1 - FREQ_EMA_DECAY)
        self.neg_rate_ema = self.neg_rate_ema * FREQ_EMA_DECAY + neg_rate * (1 - FREQ_EMA_DECAY)

        effective_dopamine = self.dopamine_level
        if PLASTICITY_MODE == "tonic_baseline":
            effective_dopamine += TONIC_BIAS

        if len(self.plastic_data_idx) and PLASTIC_LR != 0.0 and abs(effective_dopamine) > 1e-4:
            eligibility = self.pre_trace[self.plastic_pre] * spikes[self.plastic_post]
            scale = 1.0
            if PLASTICITY_MODE == "freq_normalized":
                freq = self.pos_rate_ema if effective_dopamine > 0 else self.neg_rate_ema
                ref = (self.pos_rate_ema + self.neg_rate_ema) / 2.0
                scale = min(FREQ_NORM_MAX_SCALE, ref / max(freq, FREQ_NORM_EPS))
                self.plasticity_scale_history.append((scale, 1 if effective_dopamine > 0 else -1))
            dw = PLASTIC_LR * effective_dopamine * eligibility * scale
            self.plastic_delta = np.clip(
                self.plastic_delta + dw, -PLASTIC_DELTA_MAX, PLASTIC_DELTA_MAX
            )
            self.W.data[self.plastic_data_idx] = self.plastic_w_init + self.plastic_delta

        self.last_spikes = spikes
        return spikes

    def plastic_weight_mean(self):
        if len(self.plastic_data_idx):
            return float(self.W.data[self.plastic_data_idx].mean())
        return 0.0

    def record_plastic_weight_snapshot(self):
        if not self.record_history:
            return
        if len(self.plastic_data_idx):
            self.plastic_weight_history.append(float(self.W.data[self.plastic_data_idx].mean()))
        else:
            self.plastic_weight_history.append(0.0)
