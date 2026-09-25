"""
fetch_connectome.py

Baixa um subgrafo do conectoma MaleCNS v1.0 (Berg et al., Cell, 2026) via neuPrint
(neuprint.janelia.org, dataset "male-cns:v1.0") e salva localmente em parquet.

Requer um token de autenticação pessoal do neuPrint (gratuito):
  1. Login em https://neuprint.janelia.org com conta Google.
  2. Menu de perfil (canto superior direito) -> "Auth Token" -> copiar.
  3. Definir a variável de ambiente NEUPRINT_TOKEN com esse valor, ou passar
     --token no comando.

Sem token, use --synthetic para gerar um grafo sintético local (mesmo schema)
que serve APENAS para desenvolver/testar o resto do pipeline offline. O grafo
sintético é claramente marcado como não-real no metadata.json de saída — nunca
deve ser confundido com dados reais do conectoma.

Regiões/tipos celulares selecionados (ver README para justificativa biológica):
  - photoreceptor : R1-6, R7, R8              (fotorreceptores da retina)
  - motion        : T4, T5 (e subtipos a-d)   (detecção de movimento local)
  - object        : LC4, LPLC2                (detecção de objeto em expansão)
  - descending    : DNa02, GF (Giant Fiber)   (vias motoras descendentes)
  - dopaminergic  : PAM*, PPL1*               (reforço: PAM ~ recompensa, PPL1 ~ aversivo)

Fonte: Berg et al., "A connectome of the adult male Drosophila central nervous
system", Cell, 2026 (MaleCNS v1.0). Dataset acessado via neuPrint
(neuprint.janelia.org, dataset male-cns:v1.0).
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "connectome_data")

ROLE_TYPE_REGEX = {
    # neuPrint/Neo4j faz FULLMATCH no regex (nao busca substring) -- por isso
    # todo padrao que deve cobrir subtipos (T4a/T4b/..., PAM01/PAM02/...)
    # precisa terminar em ".*", senao so bate com a string exata "T4"/"PAM"
    # (que normalmente nao existe como tipo, so os subtipos existem).
    "photoreceptor": r"R[1-8](-6)?[dpy]?",
    # Camada intermediaria lamina/medula (L1-L5, Mi1/Mi4/Mi9, Tm1/Tm2/Tm4/Tm9):
    # no circuito visual real da mosca, R1-8 NAO fazem sinapse direta em T4/T5
    # -- o sinal passa por essas interneuronios primeiro (Takemura et al. 2013,
    # 2017). Confirmado empiricamente neste projeto: com dados reais do
    # MaleCNS v1.0, o subgrafo induzido so com photoreceptor+motion tinha ZERO
    # arestas entre os dois papeis (ver README, secao de validacao). Essa
    # camada foi adicionada depois desse achado pra fechar o circuito.
    "interneuron": r"L1.*|L2.*|L3.*|L4.*|L5.*|Mi1.*|Mi4.*|Mi9.*|Tm1.*|Tm2.*|Tm4.*|Tm9.*",
    "motion": r"T4.*|T5.*",
    "object": r"LC4|LPLC2",
    # LC10a: neuronio de campo visual pequeno associado a rastreamento de alvo
    # durante perseguicao de corte (Ribeiro et al., "Visual Projection Neurons
    # Mediating Directed Courtship in Drosophila", Cell, 2018). Adicionado no
    # Achado 9 (README) depois de confirmar via auditoria real de conectividade
    # que e o unico candidato testado com aresta direta e nao-trivial pra um
    # neuronio descendente (DNa10, 801 sinapses) -- LC4/LPLC2 tinham ZERO
    # arestas pra DNa02 (Achado 6), e LC10a tambem tem ZERO pra DNa02/GF
    # (Achado 8): e um canal visual diferente, com alvo descendente diferente.
    "target": r"LC10a.*",
    # DNa02 removido (Achado 8: recebe milhares de sinapses reais, mas NENHUMA
    # de qualquer papel visual selecionado neste subgrafo -- e dominado por
    # circuito de heading do central complex, PFL3/LAL/AOTU, fora do escopo
    # deste projeto). DNa10 adicionado: recebe de LC10a (via achado 9).
    "descending": r"DNa10.*|GF|GiantFiber.*|DNp01.*",
    "dopaminergic": r"PAM.*|PPL1.*",
}

# Conjuntos de neuronios descendentes (role "descending"). "original" e o
# usado ate o Achado 19. "achado20" acrescenta os tipos com par bilateral
# (L e R) que recebem >= 1000 de peso sinaptico dos LC ja presentes no
# subgrafo (LC4, LPLC2, LC10a), segundo scripts/find_dn_candidates.py no
# male-cns:v0.9 (docs/dn_candidatos_lc_do_subgrafo_male-cns-v0.9.csv).
# Criterio fixado antes de qualquer teste com esses neuronios.
DN_SETS = {
    "original": ROLE_TYPE_REGEX["descending"],
    "achado20": ROLE_TYPE_REGEX["descending"]
    + "|DNp04|DNp103|DNp02|DNg40|DNp11|DNp06|DNp03|DNpe056|DNp05|DNp71|DNp35|DNpe025",
}

# Conjuntos de tipos acrescentados a papeis existentes (Achado 21). Os LC do
# modelo recebiam so 13-49% da entrada real vinda do subgrafo
# (docs/lc_candidatos_male-cns-v0.9_e_v1.0.txt). "achado21" inclui os tipos do
# lobo optico mais fortes como entrada dos LC e os LC10c, pela regra fixada
# antes de testar: entra o tipo com >= 35% da propria entrada vinda do
# subgrafo ampliado (scripts/input_completeness.py,
# docs/input_completeness_male-cns-v0.9.txt). TmY21, Li39 e LC10d ficaram de
# fora por essa regra. Regex do neuPrint casa o nome inteiro: "T2" nao pega
# "T2a", "Tm3" nao pega "Tm37".
INPUT_SETS = {
    "original": {},
    "achado21": {
        "interneuron": "TmY3|T2|Tm5Y|Tm3|Li22|Tm37|Tm5a|Y3|Tm39|Li28",
        "target": "LC10c-1|LC10c-2",
    },
}

NEUPRINT_SERVER = "https://neuprint.janelia.org"
DATASET = "male-cns:v1.0"


def connect(token: str, dataset: str = DATASET):
    """Cliente do neuPrint, ou None (com mensagem) se o dataset nao existir.

    O servidor ja mudou a lista de datasets disponiveis (male-cns:v1.0 sumiu e
    male-cns:v0.9 continuou); o erro da biblioteca mostra a lista, mas nao
    explica o risco de trocar de versao.
    """
    from neuprint import Client
    try:
        return Client(NEUPRINT_SERVER, dataset=dataset, token=token)
    except RuntimeError as e:
        print(f"ERRO: {e}", file=sys.stderr)
        print("Use --dataset com um dos datasets listados. Atencao: bodyIds podem "
              "mudar entre versoes; nao misture dados de versoes diferentes no mesmo "
              "connectome_data/.", file=sys.stderr)
        return None


def _pca_project(coords_by_body: dict, body_ids: list) -> tuple[np.ndarray, int]:
    """Projeta coordenadas 3D (dict bodyId -> [x,y,z]) no eixo de maior
    variancia (1o componente principal via SVD), na ordem de `body_ids`.
    NaN para bodyIds sem coordenada. Serve como proxy de posicao
    retinotopica: neuronios fisicamente proximos no neuropilo do lobo optico
    correspondem a pontos proximos no campo visual, entao a coordenada
    espacial real e uma aproximacao razoavel de posicao na retina -- sem
    alegar corresponder exatamente ao eixo dorso-ventral verdadeiro (nao
    verificamos a orientacao dos eixos do dataset, so que existe um gradiente
    espacial coerente entre os fotorreceptores).
    """
    positions = np.full(len(body_ids), np.nan)
    valid_mask = np.array([bid in coords_by_body for bid in body_ids])
    n_valid = int(valid_mask.sum())
    if n_valid < 3:
        return positions, n_valid
    coords = np.array([coords_by_body[bid] for bid in np.array(body_ids)[valid_mask]], dtype=float)
    coords_centered = coords - coords.mean(axis=0)
    _, _, vt = np.linalg.svd(coords_centered, full_matrices=False)
    positions[valid_mask] = coords_centered @ vt[0]
    return positions, n_valid


def fetch_photoreceptor_centroids(ndf: pd.DataFrame, client) -> pd.DataFrame:
    """Centroide (x, y, z) das sinapses de saida de cada fotorreceptor, em
    coordenadas do neuPrint, indexado por bodyId."""
    from neuprint import fetch_synapses, SynapseCriteria as SC, NeuronCriteria as NC

    body_ids = ndf["bodyId"].tolist()
    syn_df = fetch_synapses(NC(bodyId=body_ids, client=client), SC(type="pre", client=client), client=client)
    if len(syn_df) == 0:
        return pd.DataFrame(columns=["x", "y", "z"])
    return syn_df.groupby("bodyId")[["x", "y", "z"]].mean()


def add_centroid_columns(ndf: pd.DataFrame, centroid: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta retina_x/retina_y/retina_z (NaN quando nao houver sinapse).
    Guardar as coordenadas brutas permite escolher o eixo da retina depois
    (ver ConnectomeNetwork, retina_axis="elevation") sem baixar tudo de novo."""
    ndf = ndf.copy()
    for axis in ("x", "y", "z"):
        ndf[f"retina_{axis}"] = ndf["bodyId"].map(centroid[axis]) if len(centroid) else np.nan
    return ndf


def compute_retina_positions_from_synapses(ndf: pd.DataFrame, client) -> tuple[np.ndarray, int, pd.DataFrame]:
    """Posicao retinotopica via centroide das sinapses de saida (terminais
    pre-sinapticos) de cada fotorreceptor. Preferido sobre somaLocation: na
    pratica, a maioria dos fotorreceptores do MaleCNS v1.0 nao tem soma
    segmentado (testado: so 11/1783 tinham somaLocation), mas todo neuronio
    incluido no conectoma tem sinapses -- e essas sinapses ficam no
    neuropilo, fisicamente organizado por posicao retinotopica.
    """
    centroid = fetch_photoreceptor_centroids(ndf, client)
    if len(centroid) == 0:
        return np.full(len(ndf), np.nan), 0, centroid
    coords_by_body = {bid: row.tolist() for bid, row in centroid.iterrows()}
    positions, n_valid = _pca_project(coords_by_body, ndf["bodyId"].tolist())
    return positions, n_valid, centroid


def fetch_real(token: str, dataset: str = DATASET,
               role_regex: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    from neuprint import NeuronCriteria as NC, fetch_adjacencies, fetch_neurons

    client = connect(token, dataset)
    if client is None:
        sys.exit(2)
    role_regex = ROLE_TYPE_REGEX if role_regex is None else role_regex

    role_frames = []
    for role, regex in role_regex.items():
        crit = NC(type=regex, regex=True, client=client)
        ndf, _ = fetch_neurons(crit, client=client)
        if role == "photoreceptor":
            positions, n_valid, centroid = compute_retina_positions_from_synapses(ndf, client)
            ndf = add_centroid_columns(ndf, centroid)
            ndf["retina_pos"] = positions
            print(f"[fetch_connectome] retina_pos (PCA do centroide de sinapses) calculado para "
                  f"{n_valid}/{len(ndf)} fotorreceptores")
        else:
            ndf = ndf.copy()
            ndf["retina_pos"] = np.nan
            for axis in ("x", "y", "z"):
                ndf[f"retina_{axis}"] = np.nan
        ndf = ndf[["bodyId", "type", "instance", "status", "retina_pos",
                   "retina_x", "retina_y", "retina_z"]].copy()
        ndf["role"] = role
        role_frames.append(ndf)
        print(f"[fetch_connectome] role={role:14s} regex={regex!r:20s} -> {len(ndf)} neurons")

    neurons_df = pd.concat(role_frames, ignore_index=True).drop_duplicates(subset="bodyId")
    if len(neurons_df) == 0:
        raise RuntimeError("Nenhum neuronio encontrado -- verifique o token e os tipos celulares.")

    all_crit = NC(bodyId=neurons_df["bodyId"].tolist(), client=client)
    _, conn_df = fetch_adjacencies(sources=all_crit, targets=all_crit, client=client)
    conn_df = conn_df.rename(columns={"bodyId_pre": "pre", "bodyId_post": "post", "weight": "weight"})
    conn_df = conn_df.groupby(["pre", "post"], as_index=False)["weight"].sum()

    return neurons_df, conn_df


def fetch_synthetic(seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Grafo sintético (NAO e dado real) so para desenvolvimento/teste offline.

    Mantem o mesmo schema do grafo real (roles, contagens na mesma ordem de
    grandeza relativa) mas conectividade e pesos sao aleatorios. Necessario
    porque baixar o MaleCNS v1.0 real exige um token pessoal do neuPrint que
    nao esta disponivel neste ambiente (ver docstring do modulo).
    """
    rng = np.random.default_rng(seed)

    role_sizes = {
        "photoreceptor": 120,
        "interneuron": 200,
        "motion": 260,
        "object": 60,
        "target": 60,
        "descending": 20,
        "dopaminergic": 30,
    }

    rows = []
    body_id = 1_000_000_000
    role_ids = {}
    for role, n in role_sizes.items():
        ids = list(range(body_id, body_id + n))
        role_ids[role] = ids
        body_id += n
        for i, bid in enumerate(ids):
            # instance com sufixo _R/_L pra descending, espelhando a
            # convencao real do neuPrint (ex. "DNa10_R") -- usado pelo split
            # motor por lateralidade em sim/network.py (Achado 9). Primeira
            # metade = R, segunda = L, o que aqui coincide com os dois
            # extremos do eixo retinotopico sintetico (ver conectividade
            # topografica abaixo), preservando o comportamento de
            # rastreamento ja validado no gerador sintetico.
            side = "R" if i < n / 2 else "L"
            instance = f"synthetic_{role}_{i}_{side}" if role == "descending" else f"synthetic_{role}_{i}"
            rows.append({
                "bodyId": bid,
                "type": f"{role}_{i % 8}",
                "instance": instance,
                "status": "Traced",
                "role": role,
                # sem somaLocation sintetica -- sim/network.py cai de volta pra
                # ordem de indice, que aqui E a posicao retinotopica de
                # verdade (foi assim que a conectividade topografica abaixo
                # foi construida).
                "retina_pos": np.nan,
                # No grafo sintetico a ordem de indice E a posicao
                # retinotopica; retina_y = i permite testar
                # retina_axis="elevation" tambem aqui (um "olho" so).
                "retina_x": np.nan,
                "retina_y": float(i) if role == "photoreceptor" else np.nan,
                "retina_z": np.nan,
            })
    neurons_df = pd.DataFrame(rows)

    # Conectividade feedforward: photoreceptor -> motion -> object -> descending.
    # As vias visuais reais da mosca (lobo optico) sao RETINOTOPICAS: cada
    # coluna preserva a posicao espacial ao longo das camadas. Um grafo
    # bipartido puramente aleatorio (Erdos-Renyi) destroi essa estrutura e
    # torna o readout motor cego a posicao da bola (isso foi de fato testado
    # e falhou -- ver secao de validacao do README). Por isso o gerador
    # sintetico usa uma probabilidade de conexao localizada (kernel gaussiano
    # sobre a posicao normalizada do neuronio), aproximando a topografia real
    # sem alegar corresponder a nenhum dado biologico especifico.
    # dopaminergic conecta-se de forma difusa (sem topografia), analogo a
    # neuromodulacao ampla dos clusters PAM/PPL1 reais.
    edges = []

    def random_bipartite(src_ids, dst_ids, p, wmax):
        for s in src_ids:
            targets = rng.random(len(dst_ids)) < p
            for d, hit in zip(dst_ids, targets):
                if hit:
                    edges.append((s, d, int(rng.integers(1, wmax))))

    def topographic_bipartite(src_ids, dst_ids, p_peak, sigma, wmax):
        src_pos = np.linspace(0, 1, len(src_ids))
        dst_pos = np.linspace(0, 1, len(dst_ids))
        for s, sp in zip(src_ids, src_pos):
            prob = p_peak * np.exp(-0.5 * ((dst_pos - sp) / sigma) ** 2)
            targets = rng.random(len(dst_ids)) < prob
            for d, hit in zip(dst_ids, targets):
                if hit:
                    edges.append((s, d, int(rng.integers(1, wmax))))

    # photoreceptor -> interneuron (lamina/medula) -> motion (T4/T5) -> object
    # (LC4/LPLC2) -> descending. Sem aresta direta motion->descending: na
    # auditoria dos dados reais do MaleCNS v1.0 essa aresta simplesmente nao
    # existe (T4/T5 projetam pra neuronios colunares do lobulo como LC4/LPLC2,
    # nao direto pra vias descendentes) -- ver README.
    topographic_bipartite(role_ids["photoreceptor"], role_ids["interneuron"], p_peak=0.35, sigma=0.12, wmax=15)
    topographic_bipartite(role_ids["interneuron"], role_ids["motion"], p_peak=0.35, sigma=0.12, wmax=15)
    topographic_bipartite(role_ids["motion"], role_ids["object"], p_peak=0.35, sigma=0.15, wmax=20)
    topographic_bipartite(role_ids["object"], role_ids["descending"], p_peak=0.40, sigma=0.20, wmax=30)
    # Segundo caminho paralelo (Achado 9): interneuron -> target (LC10a) ->
    # descending, espelhando a conexao real Tm4 -> LC10a -> DNa10 encontrada
    # na auditoria de conectividade (docs/achado-8-auditoria-dna02.md).
    topographic_bipartite(role_ids["interneuron"], role_ids["target"], p_peak=0.35, sigma=0.12, wmax=15)
    topographic_bipartite(role_ids["target"], role_ids["descending"], p_peak=0.40, sigma=0.20, wmax=30)
    random_bipartite(role_ids["dopaminergic"], role_ids["target"], p=0.10, wmax=10)
    random_bipartite(role_ids["dopaminergic"], role_ids["motion"], p=0.10, wmax=10)
    random_bipartite(role_ids["dopaminergic"], role_ids["object"], p=0.10, wmax=10)
    random_bipartite(role_ids["dopaminergic"], role_ids["descending"], p=0.10, wmax=10)

    conn_df = pd.DataFrame(edges, columns=["pre", "post", "weight"])
    conn_df = conn_df.groupby(["pre", "post"], as_index=False)["weight"].sum()

    return neurons_df, conn_df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--token", default=os.environ.get("NEUPRINT_TOKEN"),
                     help="Token de autenticacao do neuPrint (ou defina NEUPRINT_TOKEN).")
    ap.add_argument("--synthetic", action="store_true",
                     help="Gera grafo SINTETICO local em vez de baixar dados reais.")
    ap.add_argument("--seed", type=int, default=0, help="Seed do gerador sintetico.")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--dataset", default=DATASET,
                     help=f"dataset do neuPrint (padrao: {DATASET})")
    ap.add_argument("--dn-set", choices=sorted(DN_SETS), default="original",
                     help="conjunto de neuronios descendentes (ver DN_SETS)")
    ap.add_argument("--input-set", choices=sorted(INPUT_SETS), default="original",
                     help="tipos extras de entrada dos LC (ver INPUT_SETS)")
    args = ap.parse_args()
    role_regex = {**ROLE_TYPE_REGEX, "descending": DN_SETS[args.dn_set]}
    for role, extra in INPUT_SETS[args.input_set].items():
        role_regex[role] = f"{role_regex[role]}|{extra}"

    os.makedirs(args.out_dir, exist_ok=True)

    if args.synthetic:
        print("[fetch_connectome] Gerando grafo SINTETICO (nao e dado real do conectoma).")
        neurons_df, conn_df = fetch_synthetic(seed=args.seed)
        source_label = "synthetic"
    else:
        if not args.token:
            print("ERRO: nenhum token fornecido. Use --token, defina NEUPRINT_TOKEN, "
                  "ou rode com --synthetic para desenvolvimento offline.", file=sys.stderr)
            sys.exit(1)
        print(f"[fetch_connectome] Conectando a {NEUPRINT_SERVER} dataset={args.dataset} "
              f"dn_set={args.dn_set} input_set={args.input_set} ...")
        neurons_df, conn_df = fetch_real(args.token, args.dataset, role_regex)
        source_label = "real"

    neurons_path = os.path.join(args.out_dir, "neurons.parquet")
    edges_path = os.path.join(args.out_dir, "edges.parquet")
    meta_path = os.path.join(args.out_dir, "metadata.json")

    neurons_df.to_parquet(neurons_path, index=False)
    conn_df.to_parquet(edges_path, index=False)

    counts = neurons_df["role"].value_counts().to_dict()
    meta = {
        "source": source_label,
        "dataset": args.dataset if source_label == "real" else "SYNTHETIC (nao e dado real)",
        "dn_set": args.dn_set,
        "input_set": args.input_set,
        "citation": "Berg et al., 'A connectome of the adult male Drosophila central nervous system', Cell, 2026 (MaleCNS).",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "n_neurons": int(len(neurons_df)),
        "n_edges": int(len(conn_df)),
        "counts_by_role": counts,
        "role_type_regex": role_regex,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[fetch_connectome] neuronios: {len(neurons_df)}  arestas: {len(conn_df)}")
    print(f"[fetch_connectome] por role: {counts}")
    print(f"[fetch_connectome] salvo em: {args.out_dir}  (source={source_label})")


if __name__ == "__main__":
    main()
