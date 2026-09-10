"""Carrega o subgrafo do conectoma e propaga atividade nele.

Pesos sinapticos = contagem de sinapses do conectoma (coluna `weight` de
edges.parquet), escalados por SYNAPSE_GAIN pra manter o LIF numericamente
estavel. Um subconjunto pequeno e explicito de sinapses (motion/object ->
descending) e marcado como "plastico" e sofre uma regra de plasticidade de
tres fatores (Hebbian gated by dopamina, cf. Fremaux & Gerstner 2016) quando
ha reforco. Todo o resto do grafo permanece fixo.
"""
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
# O delta acumulado (nao o peso absoluto) e limitado a +-PLASTIC_DELTA_MAX em
# torno do peso sinaptico ORIGINAL de cada sinapse plastica. Um teto absoluto
# (ex.: sempre <= 5.0) quebraria com dados reais, onde o peso original ja
# passa disso (contagem de sinapses real * SYNAPSE_GAIN pode chegar a ~10+) --
# nesse caso um np.clip absoluto colapsaria o peso na primeira atualizacao
# mesmo com PLASTIC_LR=0, contaminando qualquer comparacao com um controle
# "sem plasticidade" (bug real encontrado e corrigido durante a validacao com
# dados reais do MaleCNS v1.0 -- ver README).
PLASTIC_DELTA_MAX = 3.0


class ConnectomeNetwork:
    def __init__(self, data_dir):
        neurons_path = os.path.join(data_dir, "neurons.parquet")
        edges_path = os.path.join(data_dir, "edges.parquet")
        self.neurons_df = pd.read_parquet(neurons_path)
        self.edges_df = pd.read_parquet(edges_path)

        self.n = len(self.neurons_df)
        self.body_id_to_idx = {bid: i for i, bid in enumerate(self.neurons_df["bodyId"])}

        self.role_idx = {
            role: self.neurons_df.index[self.neurons_df["role"] == role].to_numpy()
            for role in self.neurons_df["role"].unique()
        }
        self.photoreceptor_idx = self.role_idx.get("photoreceptor", np.array([], dtype=int))
        self.motion_idx = self.role_idx.get("motion", np.array([], dtype=int))
        self.object_idx = self.role_idx.get("object", np.array([], dtype=int))
        self.target_idx = self.role_idx.get("target", np.array([], dtype=int))
        self.descending_idx = self.role_idx.get("descending", np.array([], dtype=int))
        self.dopaminergic_idx = self.role_idx.get("dopaminergic", np.array([], dtype=int))
        # Subconjunto de `descending` cujo tipo e DNa10 -- usado no Achado 13
        # pra testar diretamente a hipotese do Achado 8 (canal LC10a->DNa10
        # fraco): injeta um sinal sintetico so nesses neuronios, sem tocar
        # nas Giant Fiber. Vazio se nao houver DNa10 no subgrafo (ex. dados
        # sinteticos, que nao distinguem sub-populacoes de descending).
        desc_types = self.neurons_df["type"].astype(str).to_numpy()
        is_dna10 = np.array([t.startswith("DNa10") for t in desc_types[self.descending_idx]])
        self.dna10_idx = self.descending_idx[is_dna10]
        self.photoreceptor_positions = self._compute_photoreceptor_positions()

        self.motor_up_idx, self.motor_down_idx = self._compute_motor_groups()

        # PAM = valencia positiva (recompensa), PPL1 = valencia negativa
        # (aversivo), conforme Aso et al. 2014 sobre neuronios dopaminergicos
        # da mushroom body de Drosophila. No grafo sintetico (sem tipos PAM/
        # PPL1 reais) o cluster dopaminergico e simplesmente dividido ao meio.
        types = self.neurons_df["type"].astype(str)
        is_pam = types.str.startswith("PAM")
        dop_set = set(self.dopaminergic_idx.tolist())
        if is_pam.any():
            pam_idx = self.neurons_df.index[is_pam].to_numpy()
            pam_idx = np.array([i for i in pam_idx if i in dop_set])
            ppl1_idx = np.array([i for i in self.dopaminergic_idx if i not in set(pam_idx.tolist())])
        else:
            half_dop = len(self.dopaminergic_idx) // 2
            pam_idx = self.dopaminergic_idx[:half_dop]
            ppl1_idx = self.dopaminergic_idx[half_dop:]
        self.dopamine_positive_idx = pam_idx
        self.dopamine_negative_idx = ppl1_idx

        pre = self.edges_df["pre"].map(self.body_id_to_idx).to_numpy()
        post = self.edges_df["post"].map(self.body_id_to_idx).to_numpy()
        weight = self.edges_df["weight"].to_numpy(dtype=np.float64) * SYNAPSE_GAIN

        valid = ~(np.isnan(pre) | np.isnan(post))
        pre, post, weight = pre[valid].astype(int), post[valid].astype(int), weight[valid]

        self.W = sp.csr_matrix((weight, (post, pre)), shape=(self.n, self.n))

        # Plastico = qualquer sinapse de uma via "de detecao" (motion/object/
        # target) chegando em descending -- generico o bastante pra cobrir os
        # dois caminhos paralelos confirmados na auditoria real (object->GF,
        # target->DNa10, Achados 6, 8 e 9), sem hardcodar nomes especificos.
        upstream_of_descending = (
            set(self.motion_idx.tolist())
            | set(self.object_idx.tolist())
            | set(self.target_idx.tolist())
        )
        descending_set = set(self.descending_idx.tolist())
        plastic_mask = np.array([
            (pr in upstream_of_descending and po in descending_set)
            for pr, po in zip(pre, post)
        ])
        self.plastic_pre = pre[plastic_mask]
        self.plastic_post = post[plastic_mask]
        self.plastic_data_idx = self._find_data_indices(self.plastic_pre, self.plastic_post)
        self.plastic_w_init = self.W.data[self.plastic_data_idx].copy()
        self.plastic_delta = np.zeros(len(self.plastic_data_idx))
        self.plastic_weight_history = []

        self.lif = LIFPopulation(self.n)
        self.dopamine_level = 0.0
        self.pre_trace = np.zeros(self.n)
        self.post_trace = np.zeros(self.n)
        self.last_spikes = np.zeros(self.n, dtype=bool)

    def _compute_motor_groups(self):
        """Divide `descending_idx` em dois grupos de leitura motora usando a
        lateralidade real (sufixo _R/_L do campo `instance`, ex. "DNa10_R",
        "DNp01(GF)_L") quando disponivel, em vez de metade/metade por indice
        (o que causou o Achado 6: por coincidencia colocava neuronios
        anatomicamente desconectados de um lado so). Mapear
        esquerda/direita em "sobe"/"desce" de um paddle vertical e uma
        ESCOLHA DE ENGENHARIA, nao um fato biologico -- controle de direcao
        de caminhada lateral (o que DNa10/GF realmente codificam) nao e a
        mesma coisa que posicao vertical de um objeto. Ver README, Achado 9.
        """
        n = len(self.descending_idx)
        if n == 0:
            return np.array([], dtype=int), np.array([], dtype=int)

        instances = self.neurons_df["instance"].astype(str).to_numpy()[self.descending_idx]
        is_right = np.array([bool(re.search(r"(_R$|\(R\))", inst)) for inst in instances])
        is_left = np.array([bool(re.search(r"(_L$|\(L\))", inst)) for inst in instances])
        n_lateral = int((is_right | is_left).sum())

        if n_lateral == n and n >= 2:
            return self.descending_idx[is_right], self.descending_idx[is_left]

        print(f"[network] lateralidade (_R/_L) so encontrada em {n_lateral}/{n} neuronios "
              f"descendentes -- usando metade/metade por indice como fallback "
              f"(o mesmo esquema arbitrario do Achado 6).")
        half = n // 2
        return self.descending_idx[:half], self.descending_idx[half:]

    def _compute_photoreceptor_positions(self):
        """Posicao normalizada [0,1] de cada fotorreceptor na "retina" 1D
        usada por game/sensory_map.py. Usa a coluna `retina_pos` (PCA da
        somaLocation real, ver fetch_connectome.py) quando disponivel; cai de
        volta pra ordem de indice da lista quando nao (gerador sintetico, ou
        neuronios reais sem somaLocation segmentada). A ordem de indice so e
        uma proxy valida de posicao espacial no gerador sintetico -- ele foi
        construido pra que fosse; em dados reais sem retina_pos ela e
        arbitraria (ordem de retorno da query do neuPrint), o que foi
        confirmado experimentalmente como nao-informativo sobre posicao da
        bola (ver README, secao de validacao)."""
        n = len(self.photoreceptor_idx)
        if n == 0:
            return np.array([])
        if "retina_pos" in self.neurons_df.columns:
            raw = self.neurons_df["retina_pos"].to_numpy(dtype=float)[self.photoreceptor_idx]
        else:
            raw = np.full(n, np.nan)

        valid = ~np.isnan(raw)
        if valid.sum() < 2:
            print("[network] retina_pos indisponivel -- usando ordem de indice como "
                  "fallback (so e uma posicao retinotopica valida no gerador sintetico).")
            return np.linspace(0, 1, n)

        median_val = np.median(raw[valid])
        raw = np.where(valid, raw, median_val)
        n_missing = int((~valid).sum())
        if n_missing:
            print(f"[network] {n_missing}/{n} fotorreceptores sem somaLocation -- "
                  f"usando a mediana dos demais como posicao de fallback pra esses.")
        lo, hi = raw.min(), raw.max()
        return (raw - lo) / max(hi - lo, 1e-9)

    def _find_data_indices(self, pre_arr, post_arr):
        idx = np.empty(len(pre_arr), dtype=np.int64)
        indptr, indices = self.W.indptr, self.W.indices
        for k, (pr, po) in enumerate(zip(pre_arr, post_arr)):
            row_start, row_end = indptr[po], indptr[po + 1]
            row_cols = indices[row_start:row_end]
            hit = np.where(row_cols == pr)[0]
            idx[k] = row_start + hit[0]
        return idx

    def reset(self):
        self.lif.reset()
        self.dopamine_level = 0.0
        self.pre_trace[:] = 0.0
        self.post_trace[:] = 0.0
        self.last_spikes[:] = False

    def step(self, external_current):
        """external_current deve incluir qualquer estimulo de reforco nos
        indices dopamine_positive_idx/dopamine_negative_idx (ganho/perda de
        ponto) alem do estimulo sensorial nos fotorreceptores -- a dopamina
        usada na plasticidade emerge do spiking real desses neuronios, nao de
        um canal escondido."""
        current = self.W.dot(self.last_spikes.astype(np.float64)) + external_current
        spikes = self.lif.step(current)

        self.pre_trace *= TRACE_DECAY
        self.post_trace *= TRACE_DECAY
        self.pre_trace[spikes] += 1.0
        self.post_trace[spikes] += 1.0

        pos_rate = spikes[self.dopamine_positive_idx].mean() if len(self.dopamine_positive_idx) else 0.0
        neg_rate = spikes[self.dopamine_negative_idx].mean() if len(self.dopamine_negative_idx) else 0.0
        self.dopamine_level = self.dopamine_level * DOPAMINE_DECAY + (pos_rate - neg_rate)

        if len(self.plastic_data_idx) and PLASTIC_LR != 0.0 and abs(self.dopamine_level) > 1e-4:
            pre_active = self.pre_trace[self.plastic_pre]
            post_spiked = spikes[self.plastic_post].astype(np.float64)
            eligibility = pre_active * post_spiked
            dw = PLASTIC_LR * self.dopamine_level * eligibility
            self.plastic_delta = np.clip(self.plastic_delta + dw, -PLASTIC_DELTA_MAX, PLASTIC_DELTA_MAX)
            self.W.data[self.plastic_data_idx] = self.plastic_w_init + self.plastic_delta

        self.last_spikes = spikes
        return spikes

    def record_plastic_weight_snapshot(self):
        if len(self.plastic_data_idx):
            self.plastic_weight_history.append(float(self.W.data[self.plastic_data_idx].mean()))
        else:
            self.plastic_weight_history.append(0.0)
