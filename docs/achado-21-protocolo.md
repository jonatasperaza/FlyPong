# Achado 21 — protocolo pré-registrado

Escrito e commitado **antes** de baixar os dados ampliados e de qualquer
execução com eles.

## Pergunta

No modelo, os neurônios LC recebem só 13–49% da entrada que recebem no
conectoma real a partir do subgrafo (`docs/lc_candidatos_male-cns-v0.9_e_v1.0.txt`).
Se completarmos essa entrada com os tipos do lobo óptico que faltam, a mesma
regra aprende mais, principalmente na condição **normal**? Nessa condição o
DNa10 não tinha entradas "bola acima".

## Dados

- `male-cns:v0.9` (mesma versão do Achado 20), com 4 descendentes (GF e
  DNa10).
- **Ampliado:** `fetch_connectome.py --dataset male-cns:v0.9 --dn-set original
  --input-set achado21`.
- Tipos novos, escolhidos pela regra "≥ 35% da própria entrada vinda do
  subgrafo ampliado" (`docs/input_completeness_male-cns-v0.9.txt`):
  - interneurônios: TmY3, T2, Tm5Y, Tm3, Li22, Tm37, Tm5a, Y3, Tm39 e Li28;
  - target: LC10c-1 e LC10c-2.
- Excluídos pela mesma regra: TmY21, Li39 e LC10d.
- **Comparação:** os resultados de 4 descendentes na v0.9 já obtidos no
  Achado 20 (`docs/achado-19-resultados-real-male-cns-v0.9-elevation-visual-1500serves.jsonl`):
  treinada 36,6% na invertida e 30,7% na normal.

## Configuração (sem nenhum ajuste novo)

```
python scripts/run_achado19.py --data-dir <ampliado> --retina-axis elevation \
    --homeostasis-scope visual --serves 1500
```

- Hiperparâmetros: eta 0,001, ruído 6, 300 saques de desenvolvimento e
  1.500 de treino.
- Seeds de treino: 42 e 1–5. Seeds de teste: 1001–1006, com 300 saques
  cada.
- Políticas: treinada, controle homeostase e controle inato.
- Os LC10c entram no papel `target`, então as sinapses deles para os
  descendentes são plásticas e as entradas deles passam pela homeostase dos
  LC, como as dos LC10a.

## Critérios (definidos agora)

1. **Principal:** na condição **normal**, o IC95 da treinada fica acima do
   IC95 de cada controle em ≥ 5 de 6 seeds.
2. **Invertida:** o mesmo critério, em ≥ 5 de 6 seeds.
3. **Efeito da completude:** em cada condição, a taxa somada da treinada
   (1.800 saques) fica acima da de 4 descendentes na v0.9 sem os tipos novos,
   com IC95 de Wilson sem sobreposição.

Antes das execuções são permitidos diagnósticos que não usam as seeds de
teste (informação por camada e sintonia das entradas dos descendentes). Eles
não mudam nenhum hiperparâmetro nem critério. Qualquer resultado entra no
README como Achado 21, inclusive se for negativo.
