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
