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
# Neurotransmissores tratados como inibitorios com synapse_sign="nt", como em
# Shiu et al. 2024 (Nature, modelo LIF do conectoma FlyWire): acetilcolina
# excita, GABA e glutamato inibem. Os demais (histamina dos fotorreceptores,
# dopamina, "unclear") ficam excitatorios. Simplificacao: no inseto real a
# histamina dos fotorreceptores hiperpolariza L1-L3; aqui a entrada da retina
# continua excitatoria, como em todos os Achados.
INHIBITORY_NT = frozenset({"gaba", "glutamate"})
# Modo neuron_model="flyvis": duracao de um passo (s) para os neuronios
# graduados. 4 passos por frame = 20 ms por frame (50 Hz).
GRADED_DT = 0.005
GRADED_ROLES = ("photoreceptor", "interneuron", "motion")
# Ganho maximo do estimulo da bola (game.sensory_map, gain=6): nos graduados
# a entrada vira luminancia em [0, 1], como no flyvis.
GRADED_INPUT_GAIN = 6.0


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
                 retina_axis: str = "pca", synapse_sign: str = "none",
                 neuron_model: str = "lif", flyvis_model: str = "000",
                 graded_to_spiking: str = "absolute"):
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
        if synapse_sign not in {"none", "nt"}:
            raise ValueError(f"unknown synapse_sign: {synapse_sign}")
        self.synapse_sign = synapse_sign
        if synapse_sign == "nt":
            # Sinal pelo neurotransmissor previsto do neuronio PRE-sinaptico
            # (coluna `nt`; ver scripts/add_neurotransmitters.py).
            if "nt" not in self.neurons_df.columns:
                raise ValueError("synapse_sign='nt' exige a coluna `nt` em neurons.parquet "
                                 "(rode scripts/add_neurotransmitters.py)")
            nt = self.neurons_df["nt"].astype(str).str.lower().to_numpy()
            sign = np.where(np.isin(nt, list(INHIBITORY_NT)), -1.0, 1.0)
            pre_ok = np.where(valid, pre, 0).astype(np.int64)
            weight = weight * np.where(valid, sign[pre_ok], 1.0)
        if neuron_model not in {"lif", "flyvis"}:
            raise ValueError(f"unknown neuron_model: {neuron_model}")
        self.neuron_model = neuron_model
        if graded_to_spiking not in {"absolute", "rest_subtracted"}:
            raise ValueError(f"unknown graded_to_spiking: {graded_to_spiking}")
        # "rest_subtracted": neuronios com spikes recebem dos graduados a
        # variacao de relu(v) em relacao ao repouso (adaptacao a entrada
        # tonica); os graduados entre si continuam com relu(v) absoluto.
        self.graded_to_spiking = graded_to_spiking
        self._rest_cache = None
        self.graded_mask = np.zeros(self.n, dtype=bool)
        if neuron_model == "flyvis":
            weight = self._flyvis_weights(pre, post, weight, valid, flyvis_model)
        self.inhibitory_idx = (
            np.flatnonzero(np.isin(self.neurons_df["nt"].astype(str).str.lower().to_numpy(),
                                   list(INHIBITORY_NT)))
            if synapse_sign == "nt" else np.empty(0, dtype=np.int64)
        )
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
        # A regra de recompensa mantem os pesos plasticos em [0, teto]: so
        # sinapses excitatorias sao plasticas.
        plastic_mask &= weight > 0
        self.plastic_pre = pre[plastic_mask]
        self.plastic_post = post[plastic_mask]
        self.plastic_data_idx = self._find_data_indices(self.plastic_pre, self.plastic_post)
        self.plastic_w_init = self.W.data[self.plastic_data_idx].copy()
        self.plastic_delta = np.zeros(len(self.plastic_data_idx), dtype=np.float64)
        self.plastic_weight_history: list[float] = []

        self.lif = LIFPopulation(self.n)
        self.output = np.zeros(self.n, dtype=np.float64)
        if neuron_model == "flyvis":
            self.v_graded = self.graded_bias.copy()
            self._settle_graded()
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
        self.output.fill(0.0)
        if self.neuron_model == "flyvis":
            self.v_graded[:] = self.v_graded_rest
            self.output[self.graded_idx] = np.maximum(self.v_graded, 0.0)
        self.dopamine_level = 0.0
        self.pre_trace.fill(0.0)
        self.post_trace.fill(0.0)
        self.last_spikes.fill(False)
        self.last_spikes_float.fill(0.0)
        self.pos_rate_ema = 0.0
        self.neg_rate_ema = 0.0

    def step(self, external_current):
        # Avoid a new bool->float allocation in the hot SpMV path.
        if self.neuron_model == "flyvis":
            # Saida de cada neuronio: spike (0/1) nos neuronios com spikes,
            # relu(v) nos graduados (flyvis). Corrente = W . saida.
            np.copyto(self.current, self.W.dot(self.output))
            if self.graded_to_spiking == "rest_subtracted":
                self.current[self.spiking_idx] -= self._rest_drive()
            self.current += external_current
            g = self.graded_idx
            # Entrada externa dos graduados (retina) em luminancia: o estimulo
            # do jogo vai de 0 a GRADED_INPUT_GAIN.
            cur_g = self.current[g] + external_current[g] * (1.0 / GRADED_INPUT_GAIN - 1.0)
            self.v_graded += self.graded_rate * (-self.v_graded + self.graded_bias + cur_g)
            spikes = self.lif.step(self.current)
            spikes[g] = False
            self.output[:] = spikes
            self.output[g] = np.maximum(self.v_graded, 0.0)
        else:
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
        if self.neuron_model == "flyvis":
            # Contagens por frame: spikes nos neuronios com spikes, atividade
            # relu(v) integrada nos graduados.
            return self.output.copy()
        return spikes

    def _flyvis_weights(self, pre, post, weight, valid, model):
        """Pesos e parametros do modo flyvis.

        Neuronios graduados: papeis photoreceptor, interneuron e motion. Tipos
        com parametros no flyvis (coluna `flyvis_type`) usam constante de
        tempo e potencial de repouso do modelo publicado; os demais usam a
        mediana das constantes de tempo do flyvis e repouso 0.
        Arestas para neuronios graduados: sinal x n_sinapses x forca do par
        de tipos no flyvis; para pares fora da especificacao do flyvis, a
        forca e a mediana das forcas nao nulas e o sinal vem do
        neurotransmissor do neuronio pre-sinaptico. Arestas para neuronios
        com spikes (LC, descendentes, dopaminergicos) ficam como no modo
        LIF (n_sinapses x SYNAPSE_GAIN, com o sinal de synapse_sign).
        """
        import sim.flyvis_params as fp
        params = fp.load(model)
        self.flyvis_params = params
        df = self.neurons_df
        roles = df["role"].astype(str).to_numpy()
        self.graded_mask = np.isin(roles, GRADED_ROLES)
        self.graded_idx = np.flatnonzero(self.graded_mask)
        self.spiking_idx = np.flatnonzero(~self.graded_mask)
        if "flyvis_type" in df.columns:
            ftype = df["flyvis_type"].astype(object).where(df["flyvis_type"].notna(), None).to_numpy()
        else:
            ftype = np.array([None] * self.n, dtype=object)
        tau_default = float(np.median(list(params.tau.values())))
        tau = np.array([params.tau.get(t, tau_default) if t is not None else tau_default for t in ftype])
        bias = np.array([params.bias.get(t, 0.0) if t is not None else 0.0 for t in ftype])
        self.graded_tau = tau[self.graded_idx]
        self.graded_bias = bias[self.graded_idx]
        self.graded_rate = GRADED_DT / np.maximum(self.graded_tau, GRADED_DT)
        nonzero = [s for s in params.strength.values() if s > 0]
        s_default = float(np.median(nonzero))
        nt = df["nt"].astype(str).str.lower().to_numpy() if "nt" in df.columns else np.array(["unknown"] * self.n)
        nt_sign = np.where(np.isin(nt, list(INHIBITORY_NT)), -1.0, 1.0)
        counts = self.edges_df["weight"].to_numpy(dtype=np.float64)
        w = weight.copy()
        k = np.flatnonzero(valid)
        k = k[self.graded_mask[post[k].astype(np.int64)]]
        a = pre[k].astype(np.int64)
        q = post[k].astype(np.int64)
        e = pd.DataFrame({"k": k, "a": a, "q": q, "c": counts[k],
                          "ta": ftype[a], "tq": ftype[q]})
        # Neuronios unicos que cobrem o campo visual inteiro (CT1, Am1) sao
        # compartimentos por coluna no flyvis e nao tem equivalente aqui.
        wide = df["type"].astype(str).isin(["CT1", "Am1"]).to_numpy()
        self.flyvis_excluded_wide = int(wide.sum())
        e = e[~wide[e.a.to_numpy()] & ~wide[e.q.to_numpy()]]
        w[k] = 0.0
        has = e.ta.notna() & e.tq.notna()
        e["pair"] = list(zip(e.ta, e.tq))
        spec = has & e.pair.isin(set(params.strength))
        # Pares do flyvis: forca e sinal do flyvis; contagens do MaleCNS
        # normalizadas por par para dar a mesma entrada media por neuronio
        # pos-sinaptico que o flyvis (sum das colunas).
        es = e[spec]
        n_post = pd.Series(ftype).value_counts()
        tot = es.groupby("pair").c.sum()
        mean_per_post = tot / np.array([n_post[p_[1]] for p_ in tot.index])
        fv_n = pd.Series({p_: params.n_syn[p_] for p_ in tot.index})
        scale = fv_n / mean_per_post
        sc = es.pair.map(scale).to_numpy()
        w[es.k.to_numpy()] = (np.array([params.sign[p_] * params.strength[p_] for p_ in es.pair])
                               * es.c.to_numpy() * sc)
        # Arestas que envolvem tipos sem parametros no flyvis: sinal do
        # neurotransmissor, forca normalizada para que a soma |entrada| desses
        # neuronios seja a mediana da soma |entrada| dos tipos do flyvis.
        ex = e[~spec & ~(has)]
        if len(ex):
            fv_in = pd.Series(np.abs(w[es.k.to_numpy()])).groupby(es.q.to_numpy()).sum()
            target_in = float(fv_in.median())
            tot_ex = ex.groupby("q").c.sum()
            sc2 = ex.q.map(target_in / tot_ex).to_numpy()
            w[ex.k.to_numpy()] = nt_sign[ex.a.to_numpy()] * ex.c.to_numpy() * sc2
        self.flyvis_edges_in_spec = int(spec.sum())
        self.flyvis_edges_extra = int(len(ex))
        self.flyvis_edges_dropped = int((~spec & has).sum())
        return w

    def _rest_drive(self):
        """W[spiking, graded] . relu(v_repouso), recalculado com os pesos
        atuais (a homeostase e a plasticidade mudam W.data no lugar)."""
        W = self.W
        c = self._rest_cache
        if c is None or c[0] is not W.data or c[1] is not self.v_graded_rest:
            rows = self.spiking_idx
            starts, ends = W.indptr[rows], W.indptr[rows + 1]
            lens = ends - starts
            K = np.concatenate([np.arange(s, e) for s, e in zip(starts, ends)]) if len(rows) else np.empty(0, np.int64)
            rid = np.repeat(np.arange(len(rows)), lens)
            rest = np.zeros(self.n)
            rest[self.graded_idx] = np.maximum(self.v_graded_rest, 0.0)
            restK = rest[W.indices[K]]
            c = (W.data, self.v_graded_rest, K, rid, restK)
            self._rest_cache = c
        _, _, K, rid, restK = c
        return np.bincount(rid, weights=self.W.data[K] * restK, minlength=len(self.spiking_idx))

    def _settle_graded(self, steps=400, external=None):
        """Estado de repouso dos graduados sem estimulo (o flyvis comeca no
        estado estacionario); usado em reset()."""
        g = self.graded_idx
        out = np.zeros(self.n)
        v = self.graded_bias.copy()
        for _ in range(steps):
            out[g] = np.maximum(v, 0.0)
            cur = self.W.dot(out)[g]
            if external is not None:
                cur = cur + external[g] / GRADED_INPUT_GAIN
            v += self.graded_rate * (-v + self.graded_bias + cur)
        self.v_graded_rest = v.copy()
        self._rest_cache = None
        self.v_graded[:] = v
        self.output[g] = np.maximum(v, 0.0)

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
