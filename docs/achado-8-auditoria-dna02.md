# Achado 8: auditoria de conectividade real de DNa02, Giant Fiber, e LC10a

Dados brutos, coletados via `neuprint-python` (dataset `male-cns:v1.0`,
`fetch_adjacencies`, não `fetch_neurons` com filtro de tipo, pra não presumir
quais parceiros iam aparecer). CSVs brutos em `docs/raw_*.csv`.

## 1. Todos os parceiros pré-sinápticos de DNa02 e Giant Fiber (sem filtro de tipo)

| neurônio alvo | bodyId | total de sinapses de entrada | nº de parceiros distintos |
|---|---|---|---|
| DNa02_R | 10360 | 24 174 | 1203 |
| DNa02_L | 523769 | 23 971 | 1141 |
| DNp01(GF)_R | 10001 | 16 054 | 697 |
| DNp01(GF)_L | 10010 | 20 727 | 768 |

DNa02 recebe mais sinapses totais que Giant Fiber, não é um neurônio "pouco
conectado" de forma geral. O problema (Achado 6) é que nenhuma dessas ~24 mil
sinapses vem de `object` (LC4/LPLC2) ou `motion` (T4/T5), os únicos papéis
visuais que este projeto tinha selecionado até então.

Top parceiros por peso (amostra, ver CSV bruto pra lista completa):

**DNa02_R (10360)**: AN03A008 (717), PS049 (528), LAL018 (319), CB0431 (307),
DNa03 (297), AOTU019 (289), PS013 (266), PS059 (263+259), DNae005 (255),
VES072 (243), GNG521 (229), LAL046 (219), SAD085 (208), AOTU025 (196).

**DNp01(GF)_R (10001)**: DNp70 (537), PVLP010 (297), PVLP122 (278+233),
GNG300 (213), SAD064 (186+184+178), SAD091 (181), SAD109 (179), LHAD1g1
(161), PVLP062 (160), SAD073 (159), AN12B001 (156).

Os tipos que dominam a entrada de DNa02 (LAL = lateral accessory lobe, AOTU =
anterior optic tubercle, PS0xx, CB0431 = central body) são todos regiões do
central complex, ligadas ao circuito de heading. Isso é consistente com a
literatura sobre DNa02 receber comando de direção do circuito de navegação
(Rayshubskiy et al. 2020; Westeinde et al. 2024), não de detectores de objeto
do lobo óptico.

## 2. Busca por candidatos específicos em DNa02/GF

| tipo candidato | citação | sinapses encontradas em DNa02/GF |
|---|---|---|
| LC10a | Ribeiro et al. 2018 | **0** |
| LC11 | — | **0** |
| PFL1 | central complex, controle motor | **0** |
| PFL2 | central complex, controle motor | 5 (DNa02_R apenas) |
| PFL3 | Rayshubskiy et al. 2020 | 736 (380 em DNa02_R, 356 em DNa02_L) |
| PFN* | central complex, ring attractor | **0** |
| hDeltaB | central complex | **0** |
| EPG | central complex, ring attractor (heading) | **0** |

LC10a não aparece em DNa02 nem em Giant Fiber: a hipótese de trabalho
original (LC10a→DNa02) é refutada pelos dados. PFL3 aparece com peso
comparável ao que depois encontramos para LC10a→DNa10 (ver seção 3), mas PFL3
é um neurônio de heading do central complex, não codifica posição
retinotópica de um objeto, então não é um substituto direto adequado pra
"rastrear a bola", mesmo estando conectado.

## 3. Pra onde LC10a projeta de verdade

275 neurônios `LC10a` encontrados no dataset. Top alvos por peso total:

| tipo-alvo | peso total | nº de neurônios-alvo |
|---|---|---|
| LC10a (recorrente) | 25 244 | 275 |
| TuTuA_2 | 23 354 | 2 |
| AOTU008 | 16 114 | 25 |
| AOTU041 | 14 304 | 4 |
| AOTU019 | 14 210 | 2 |
| LC10d | 13 852 | 214 |
| LT52 | 12 858 | 34 |
| (…mais AOTU0xx…) | | |

A esmagadora maioria da saída de LC10a vai pro AOTU (anterior optic
tubercle), o que bate exatamente com a via de corte de Ribeiro et al. 2018
(LC10a → AOTU → bulb → central complex). Essa via é polissináptica (passa
pelo central complex inteiro antes de qualquer coisa parecida com controle
motor), não um circuito de 1-2 saltos como o resto deste projeto usa.

Candidatos descendentes (prefixo DN) entre os alvos de LC10a, com peso bem
menor que AOTU, mas os únicos não-zero encontrados:

| tipo descendente | peso total | nº de neurônios-alvo |
|---|---|---|
| **DNa10** | **801** | **2** |
| DNp11 | 184 | 2 |
| DNa16 | 169 | 1 |
| DNp63 | 146 | 2 |
| DNp09 | 65 | 1 |
| DNd05 | 42 | 1 |
| DNp27 | 21 | 2 |

DNa10 é o candidato mais forte: 801 sinapses via só 2 neurônios-alvo (um par
bilateral limpo, ver seção 4), ordens de grandeza acima dos outros
candidatos e não-zero (diferente de LC4/LPLC2→DNa02).

## 4. DNa10: o candidato escolhido

| bodyId | tipo | instance |
|---|---|---|
| 10379 | DNa10 | DNa10_R |
| 523329 | DNa10 | DNa10_L |

Par bilateral limpo (`_R`/`_L` no campo `instance`), usado no Achado 9 pro
split motor por lateralidade.

| DNa10 | total de entrada | entrada de LC10a | fração de LC10a |
|---|---|---|---|
| 10379 (R) | 15 146 | 424 | 2.8% |
| 523329 (L) | 15 078 | 377 | 2.5% |

LC10a é uma fração pequena (~2.7%) do input total de DNa10; o resto vem
principalmente de PLP213, AOTU063_a/b, PS200, PS355, IB117/IB018, de novo
região de integração multissensorial/central complex-adjacente. DNa10 não é
um canal visual "limpo" dominado por LC10a: é uma conexão real e não-trivial,
mas minoritária frente ao resto do input de DNa10 que este projeto não
modela.

## 5. Confirmação da cadeia completa (photoreceptor → ... → DNa10)

Checando os parceiros de ORIGEM de LC10a (não os alvos), pra confirmar que o
caminho tem aresta em todo elo, não só nas pontas:

| tipo de origem | peso total pra LC10a |
|---|---|
| Tm4 (já em `interneuron`) | **2657** |
| Tm2 (já em `interneuron`) | 77 |
| Tm1 (já em `interneuron`) | 9 |
| T4 (motion) | 1 |
| Tm3, Tm5Y, TmY21, Li22, LC9, LC10c-1/2 (não incluídos) | (maior peso, mas fora do escopo atual) |

Confirmado: `interneuron` (especificamente Tm4) → `LC10a` → `DNa10` tem
aresta real e substancial em todo elo da cadeia (Tm4→LC10a: 2657; LC10a→
DNa10: 801). O papel `target` (LC10a) foi adicionado ao `fetch_connectome.py`
com esse resultado.

## 6. Confirmação de novo: motion/object → descending

Re-confirmando o Achado 6 com a lista completa de parceiros (não só ausência
pontual): nenhuma das ~24 mil sinapses de DNa02 nem das ~16-20 mil de GF vem
de `motion` (T4/T5) ou `object` (LC4/LPLC2) além do que já era conhecido (GF
recebe de `object`, DNa02 não recebe de nenhum dos dois). Sem surpresas aqui.

## Decisão pra Fase 2

- **DNa02 removido** do papel `descending`: confirmado sem conexão de
  nenhuma via visual selecionada (nem mesmo com LC10a adicionado).
- **DNa10 adicionado**: conexão real e não-trivial via LC10a (801 sinapses,
  2.7% do seu input total), com cadeia completa confirmada
  `interneuron(Tm4) → target(LC10a) → descending(DNa10)`.
- Giant Fiber (`DNp01`/`GF`) mantido: já tinha conexão real via `object`
  (Achado 6).
