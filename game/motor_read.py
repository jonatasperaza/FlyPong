"""Le a atividade dos neuronios descendentes e decide a acao do paddle."""
import numpy as np


def read_motor_action(spike_counts_window, up_idx, down_idx, threshold=0.15, baseline_diff=0.0):
    """spike_counts_window: array (n_neurons,) com contagem de spikes de cada
    neuronio ao longo da janela de leitura. Retorna -1 (sobe), 0 (parado) ou
    1 (desce).

    `baseline_diff` remove um viés estrutural constante entre os grupos
    "sobe"/"desce": como a divisao do pool descendente nesses dois grupos e
    arbitraria (ver ConnectomeNetwork), os dois grupos podem ter
    excitabilidade media diferente mesmo sem relacao com a posicao da bola.
    Sem essa correcao de baseline o paddle fica preso numa parede. Ver
    README (secao de validacao) para a medicao desse viés.
    """
    up_rate = spike_counts_window[up_idx].mean() if len(up_idx) else 0.0
    down_rate = spike_counts_window[down_idx].mean() if len(down_idx) else 0.0

    diff = (down_rate - up_rate) - baseline_diff
    if abs(diff) < threshold:
        return 0
    return 1 if diff > 0 else -1
