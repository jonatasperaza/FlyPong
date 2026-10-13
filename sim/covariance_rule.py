"""Regra de covariancia com erro de predicao de recompensa (modo novo).

Forma classica de aprendizado por reforco em redes de spikes (Williams 1992;
Legenstein et al. 2008; Fremaux & Gerstner 2016):

    e_ij  <- lambda * e_ij + pre_j * (post_i - <post_i>)
    dw_ij  = eta * (R - <R>) * e_ij

  - `pre_j`, `post_i`: contagem de spikes no frame.
  - `<post_i>`: media recente (EMA) da atividade de cada neuronio pos-sinaptico.
    O desvio tem sinal oposto no grupo que disparou mais e no que disparou
    menos, entao a regra consegue separar "sobe" de "desce" -- a regra
    original (`dopamina * pre_trace * post_spike`) muda os dois com o mesmo
    sinal.
  - `<R>`: recompensa media recente. `R - <R>` e o erro de predicao (Schultz
    et al. 1997): se a recompensa nao depende da atividade, a mudanca media e
    zero, sem o decaimento sistematico dos Achados 13, 14 e 17.

A dopamina entra como escalar global aplicado direto na regra; nenhuma
corrente e injetada nos neuronios PAM/PPL1, entao o reforco nao interfere na
acao. So as sinapses plasticas ja existentes em ConnectomeNetwork mudam
(`plastic_pre` -> `plastic_post`), entre 0 e um teto de `w_cap_factor` vezes
o peso medio inicial.

Opcional: escalonamento homeostatico (`homeostasis_rate` > 0). Os
descendentes recebem corrente muito acima do limiar e disparam na taxa
maxima; nesse regime o ruido de exploracao nao muda os spikes e o gradiente
da recompensa e zero. O escalonamento traz cada neuron de volta a uma taxa
alvo, onde a recompensa consegue agir.

A taxa efetiva e normalizada pela escala tipica (RMS, media movel) da
elegibilidade e expressa em fracao do peso medio inicial, para que o mesmo
`eta` sirva no grafo sintetico e no real.
"""

from __future__ import annotations

import numpy as np


class CovarianceRPELearner:
    def __init__(
        self,
        net,
        *,
        eta: float = 0.02,
        w_cap_factor: float = 4.0,
        post_mean_decay: float = 0.98,
        reward_mean_decay: float = 0.98,
        elig_decay: float = 0.0,
        scale_decay: float = 0.99,
        homeostasis_rate: float = 0.0,
        target_rate: float = 0.3,
    ):
        self.net = net
        self.eta = float(eta)
        self.pre = net.plastic_pre
        self.post = net.plastic_post
        self.data_idx = net.plastic_data_idx
        w0 = net.W.data[self.data_idx]
        self.w_ref = float(w0.mean()) if len(w0) else 1.0
        self.w_cap = w_cap_factor * self.w_ref
        self.post_mean_decay = post_mean_decay
        self.reward_mean_decay = reward_mean_decay
        self.elig_decay = elig_decay
        self.scale_decay = scale_decay
        self.homeostasis_rate = float(homeostasis_rate)
        self.target_rate = float(target_rate)

        self.post_mean = np.zeros(net.n, dtype=np.float64)
        self.reward_mean = 0.0
        self.elig = np.zeros(len(self.data_idx), dtype=np.float64)
        self.elig_rms = 0.0
        self.n_updates = 0

    def update(self, frame_counts: np.ndarray, reward: float,
               perturbation: np.ndarray | None = None,
               apply: bool = True) -> float:
        """Aplica um passo da regra com as contagens do frame e a recompensa
        que seguiu a acao desse frame. Devolve o erro de predicao usado.

        `perturbation` (opcional): corrente de exploracao injetada em cada
        neuronio neste frame. Quando dada, substitui `post - <post>` pelo
        proprio ruido (perturbacao de no, Fiete & Seung 2006): o ruido tem
        media zero e e independente do estimulo, entao so o efeito dele sobre
        a recompensa gera mudanca sistematica. `post - <post>` com media lenta
        tambem carrega a variacao causada pelo estimulo, que se correlaciona
        com a recompensa sem relacao causal com a acao.

        `apply=False` so acumula elegibilidade (recompensa esparsa: o peso
        muda apenas no frame em que a recompensa chega, e `<R>` e a media das
        recompensas entregues).
        """
        if perturbation is not None:
            dev = perturbation[self.post]
        else:
            dev = frame_counts[self.post] - self.post_mean[self.post]
        self.elig *= self.elig_decay
        self.elig += frame_counts[self.pre] * dev

        if not apply:
            a = self.post_mean_decay
            self.post_mean *= a
            self.post_mean += (1 - a) * frame_counts
            return 0.0

        rpe = float(reward) - self.reward_mean
        rms = float(np.sqrt(np.mean(self.elig ** 2))) if len(self.elig) else 0.0
        if self.n_updates == 0:
            self.elig_rms = rms
        else:
            self.elig_rms = self.scale_decay * self.elig_rms + (1 - self.scale_decay) * rms

        if len(self.data_idx) and self.eta != 0.0 and self.elig_rms > 1e-9:
            w = self.net.W.data[self.data_idx]
            w += self.eta * self.w_ref * rpe * self.elig / self.elig_rms
            np.clip(w, 0.0, self.w_cap, out=w)
            self.net.W.data[self.data_idx] = w
        if len(self.data_idx) and self.homeostasis_rate != 0.0 and self.n_updates > 0:
            # Escalonamento sinaptico homeostatico (Turrigiano et al. 1998):
            # cada neuronio pos-sinaptico escala multiplicativamente suas
            # sinapses plasticas para manter a taxa media perto do alvo. Nao
            # depende da recompensa nem da direcao da bola.
            err = self.target_rate - self.post_mean[self.post]
            w = self.net.W.data[self.data_idx]
            w *= np.clip(1.0 + self.homeostasis_rate * err, 0.5, 1.5)
            np.clip(w, 0.0, self.w_cap, out=w)
            self.net.W.data[self.data_idx] = w

        a = self.post_mean_decay
        self.post_mean *= a
        self.post_mean += (1 - a) * frame_counts
        b = self.reward_mean_decay
        self.reward_mean = b * self.reward_mean + (1 - b) * float(reward)
        self.n_updates += 1
        return rpe

    def reset_traces(self):
        """Limpa a elegibilidade entre tentativas (as medias continuam)."""
        self.elig.fill(0.0)


class HomeostaticScaler:
    """Escalonamento sinaptico homeostatico de TODAS as entradas de um
    conjunto de neuronios (nao so das sinapses plasticas).

    Usado na fase de desenvolvimento (sem recompensa). No MaleCNS real, os
    neuronios LC (LC4/LPLC2 e LC10a) recebem centenas de sinapses e ficam
    saturados; em saturacao a altura da bola quase nao e decodificavel deles
    (~60%, contra ~95% nas camadas anteriores). Escalar as entradas para uma
    taxa alvo recupera a informacao (~90% na camada object). Como a
    homeostase do CovarianceRPELearner, nao usa recompensa nem direcao.
    """

    def __init__(self, net, post_idx, *, rate: float = 0.01,
                 target_rate: float = 0.3, mean_decay: float = 0.98):
        self.net = net
        self.rate = float(rate)
        self.target_rate = float(target_rate)
        self.mean_decay = float(mean_decay)
        W = net.W
        rows = np.repeat(np.arange(net.n), np.diff(W.indptr))
        # So as entradas excitatorias sao escaladas (escalonamento sinaptico
        # de sinapses excitatorias, Turrigiano et al. 1998). Sem sinapses
        # inibitorias (synapse_sign="none"), sao todas.
        self.entries = np.flatnonzero(np.isin(rows, np.asarray(post_idx)) & (W.data > 0))
        self.entry_post = rows[self.entries]
        self.post_mean = np.zeros(net.n, dtype=np.float64)

    def update(self, frame_counts: np.ndarray) -> None:
        a = self.mean_decay
        self.post_mean *= a
        self.post_mean += (1 - a) * frame_counts
        if self.rate == 0.0 or not len(self.entries):
            return
        err = self.target_rate - self.post_mean[self.entry_post]
        self.net.W.data[self.entries] *= np.clip(1.0 + self.rate * err, 0.5, 1.5)


class DopamineLearner(CovarianceRPELearner):
    """Plasticidade guiada por neuronios dopaminergicos reais (item 3).

    A recompensa nao entra na regra como escalar global: ela e injetada como
    corrente em neuronios dopaminergicos do conectoma (`reward_dans`, por
    exemplo PAL03), que disparam com spikes. A dopamina que chega a cada
    neuronio pre-sinaptico plastico i e

        DA_i = sum_d c_di (s_d - <s_d>) / max_i sum_d c_di

    com c_di as sinapses do conectoma do neuronio dopaminergico d para i,
    s_d os spikes de d no frame e <s_d> a media recente (o erro de predicao
    fica na adaptacao do proprio neuronio dopaminergico). Sinapses cujo
    neuronio pre-sinaptico nao recebe dopamina nao mudam.

    Como a recompensa da acao do frame t so chega aos neuronios
    dopaminergicos no frame t+1, a mudanca usa o traco de elegibilidade do
    frame anterior (pre x perturbacao do pos, como na regra global):
        dw_ij = eta * w_ref * DA_i(t+1) * e_ij(t) / rms(e)
    A homeostase dos descendentes (fase de desenvolvimento) e a da classe base.
    """

    def __init__(self, net, reward_dans, *, da_mean_decay: float = 0.98, **kw):
        super().__init__(net, **kw)
        self.reward_dans = np.asarray(reward_dans, dtype=np.int64)
        dans = self.reward_dans
        sub = net.W.tocsc()[:, dans].tocsr()  # linhas = pos, colunas = dans
        # contagens (pesos sem ganho/sinal) dos dopaminergicos para cada pre plastico
        pre_u = np.unique(self.pre)
        C = np.abs(sub[pre_u].toarray())
        tot = C.sum(1)
        norm = tot.max() if tot.max() > 0 else 1.0
        self.da_pre_ids = pre_u
        self.da_C = C / norm  # (n_pre_u, n_dans)
        self.pre_pos = np.searchsorted(pre_u, self.pre)
        self.da_mean = np.zeros(len(dans))
        self.da_mean_decay = float(da_mean_decay)
        self.elig_prev = None
        self.receives_da = tot[self.pre_pos] > 0
        self.last_da = np.zeros(len(pre_u))

    def update(self, frame_counts, reward, perturbation=None, apply=True):
        s = frame_counts[self.reward_dans]
        dev = s - self.da_mean
        a = self.da_mean_decay
        self.da_mean = a * self.da_mean + (1 - a) * s
        da = self.da_C @ dev
        self.last_da = da
        if perturbation is not None:
            pdev = perturbation[self.post]
        else:
            pdev = frame_counts[self.post] - self.post_mean[self.post]
        elig_now = frame_counts[self.pre] * pdev
        if self.elig_prev is not None and self.eta != 0.0 and len(self.data_idx):
            rms = float(np.sqrt(np.mean(self.elig_prev ** 2)))
            self.elig_rms = rms if self.n_updates == 0 else (
                self.scale_decay * self.elig_rms + (1 - self.scale_decay) * rms)
            if self.elig_rms > 1e-9:
                # so as sinapses cujo pre recebe dopamina mudam (o teto
                # tambem so se aplica a elas)
                m = self.receives_da
                idx = self.data_idx[m]
                w = self.net.W.data[idx]
                w += (self.eta * self.w_ref * da[self.pre_pos[m]] * self.elig_prev[m]
                      / self.elig_rms)
                np.clip(w, 0.0, self.w_cap, out=w)
                self.net.W.data[idx] = w
        self.elig_prev = elig_now
        # homeostase e medias da classe base, sem a regra global
        eta, self.eta = self.eta, 0.0
        try:
            super().update(frame_counts, 0.0, perturbation=perturbation, apply=apply)
        finally:
            self.eta = eta
        return float(da.mean())

    def reset_traces(self):
        super().reset_traces()
        self.elig_prev = None
