# Achado 16: auditoria de viabilidade — o complexo central como substrato de aprendizado visual

Investigação só de consulta ao neuPrint (`neuprint-python`), sem simulação:
nenhuma mudança em `sim/network.py`, `main.py`, ou qualquer código de
simulação nesta rodada. CSVs brutos em `docs/raw_aotu_*.csv`,
`docs/raw_tubu_*.csv`, `docs/raw_pfl_*.csv`, `docs/raw_ring_*.csv`.

## Contexto

A literatura (Wolf, Heisenberg et al.) mostra que aprendizado visual em
*Drosophila* depende do complexo central, não do corpo cogumelo. O Achado 8
já tinha esbarrado nisso sem perseguir: `PFL3` aparece com 736 sinapses em
`DNa02`, mas `DNa02` foi descartado do papel `descending` no Achado 9 por
não receber nada da via visual (`object`/`motion`/`target`). Esta auditoria
verifica se existe um caminho real ponta a ponta entre a via visual já
modelada no projeto e o complexo central, e estima o tamanho do trabalho de
fato modelar isso — decidir se vale tentar, não tentar ainda.

## Fase 1: via visual até o anel EPG — existe, com um elo fraco

Cadeia auditada: `LC10a` (papel `target` já usado) → `AOTU0xx` → `TuBu`/`ER`
(bulb/ring) → `EPG` (anel da elipsoide).

**LC10a → AOTU**: já documentado no Achado 9 (99 neurônios AOTU de tipos já
identificados, 40 017 arestas de saída totais desses neurônios).

**AOTU → TuBu/ER** (auditado agora): existe, mas é fraco. Entre os alvos de
saída dos 99 neurônios AOTU, os tipos `TuBu01`-`TuBu10` e `ER3w_a`/`ER2_b`
aparecem, mas com pesos de 1 a 17 — ordens de grandeza abaixo dos alvos
dominantes de AOTU (`LC10d`: 19 326, `LC10a`: 17 578, outros `AOTU0xx`:
1800-5000). AOTU projeta majoritariamente de volta pra dentro da própria
família LC10/AOTU, não pro anel.

| alvo | peso total | nº de neurônios-alvo |
|---|---|---|
| TuBu03 | 17 | 9 |
| TuBu02 | 16 | 7 |
| TuBu05 | 11 | 4 |
| TuBu07 | 9 | 7 |
| TuBu08 | 7 | 2 |
| TuBu04 | 4 | 2 |
| TuBu10 | 4 | 2 |
| ER3w_a | 4 | 1 |
| TuBu06 | 4 | 4 |
| TuBu01 | 3 | 1 |
| TuBu09 | 2 | 2 |
| ER2_b | 1 | 1 |

**TuBu/ER → EPG** (auditado agora): forte. 167 neurônios TuBu/ER
encontrados; o alvo dominante é `EPG` (6508 de peso total, 46 dos 46
neurônios EPG do dataset recebem essa entrada), maior até que qualquer
outro alvo individual de TuBu/ER (`ExR1`: 6303, outros ring neurons:
1000-3000 cada).

**Conclusão da Fase 1**: a cadeia completa `LC10a → AOTU → TuBu/ER → EPG`
existe com conectividade real em cada elo, não quebra em nenhum ponto. O
elo fraco é especificamente `AOTU → TuBu/ER` (peso 1-17 por tipo), não a
ausência de conexão — o sinal visual que chegaria no anel seria pequeno
frente ao resto do que AOTU processa, mas existe.

## Fase 2: caminho de volta até GF ou DNa10 — não existe

Auditados todos os alvos de saída de `PFL2` (12 neurônios) e `PFL3` (24
neurônios): nenhum dos dois projeta em `GF`/`DNp01` ou `DNa10`. A única
conexão forte de PFL2/PFL3 pra um neurônio descendente é `PFL3 → DNa02`
(741 de peso, confirmando o número já visto no Achado 9/8), e `DNa02` está
fora do escopo do projeto desde o Achado 9.

Busca ampla (não presumindo o tipo, mesmo método do Achado 9 quando achou
`LC10a→DNa10` sem presumir): auditados TODOS os parceiros de entrada já
conhecidos de `GF`/`DNp01` e `DNa10`, procurando qualquer tipo com prefixo
de complexo central (`EPG`, `PEN`, `PFN`, `PFL`, `hDelta`, `FC`, `FB`,
`ExR`, etc.):

- **GF/DNp01 (10001, 10010): zero conexão de qualquer tipo de complexo
  central.** As correspondências iniciais de `PFL2`/`PFL3`/`ExR6` que
  pareciam apontar pra GF eram, na verdade, pra `DNa02` (erro de rótulo
  corrigido durante a auditoria — os bodyIds 10360/523769 são DNa02, não
  GF; confirmado reagrupando por `bodyId_post` exato).
- **DNa10 (10379, 523329): uma única sinapse de um único neurônio `FB4M`**
  (peso 1, pro lado direito, 10379). Isso é ruído de reconstrução, não uma
  via funcional — pra comparação, o canal já usado (`LC10a→DNa10`) tem 801
  sinapses via 275 neurônios de origem.

**Conclusão da Fase 2**: não existe caminho de volta do anel pros neurônios
descendentes que o projeto já usa e validou (`GF`, `DNa10`). O único
neurônio descendente real que o complexo central alimenta com força
(`DNa02`, via `PFL3`) é exatamente o que foi excluído no Achado 9 por não
ter nenhuma entrada visual direta. Fechar esse circuito exigiria trocar
`GF`/`DNa10` por `DNa02` como readout motor, abandonando toda a
infraestrutura de leitura validada nos Achados 9-15 (lateralidade real,
teste de teto, tentativas de correção de plasticidade), não estendê-la.

## Fase 3: o anel recorrente é real e denso — mas é um mecanismo novo

Contagem de neurônios do anel no dataset (829 no total):

| grupo | tipos | total |
|---|---|---|
| EPG | EPG, EPGt | 50 |
| PEN | PEN_a(PEN1), PEN_b(PEN2) | 42 |
| PFN | PFNp_a/b/c/d/e, PFNa, PFNd, PFNm_a/b, PFNv | 416 |
| hDelta | hDeltaA-M | 189 |
| FC2 | FC2A/B/C | 92 |

Arestas internas auditadas entre esses 829 neurônios: **47 629 arestas,
peso total 263 369** — densidade de conexão comparável ou maior que outras
partes já modeladas do projeto. Agrupando por família (maiores primeiro):

| origem → destino | peso total | nº de arestas |
|---|---|---|
| PFN → PFN | 77 644 | 26 701 |
| PFN → hDelta | 36 322 | 3 167 |
| hDelta → hDelta | 34 389 | 4 521 |
| hDelta → FC2 | 26 700 | 2 088 |
| PEN → EPG | 24 251 | 811 |
| PEN → PEN | 17 916 | 1 365 |
| EPG → PEN | 12 526 | 769 |
| EPG → EPG | 11 772 | 1 122 |
| PFN → FC2 | 7 427 | 2 401 |
| EPG → PFN | 5 427 | 956 |

Essa é exatamente a topologia clássica do anel de heading (`EPG↔PEN`,
`PFN↔hDelta`, retroalimentação `EPG↔EPG`) documentada na literatura de
outros conectomas (FlyWire, hemibrain), confirmada aqui como real neste
dataset especificamente, não presumida por analogia.

**Conclusão da Fase 3**: a estrutura recorrente existe de fato e é robusta.
Não é o elo fraco desta investigação — os elos fracos são a entrada visual
(Fase 1, `AOTU→TuBu` com peso 1-17) e a saída motora (Fase 2, sem conexão
com o readout já validado).

## Estimativa de esforço e escopo

Neurônios novos necessários: 829 (anel) + 167 (TuBu/ER) + os ~99 AOTU já
parcialmente auditados (podem já estar cobertos se o papel `target` for
expandido, ou exigir um papel novo) — na ordem de 900-1100 neurônios novos,
bem menor em contagem bruta que a expansão do Achado 9 (que adicionou
28 699 neurônios só no papel `interneuron`). Em termos de contagem de nós e
arestas, essa expansão é **menor** que o redesign do Achado 9.

Mas o tipo de trabalho é qualitativamente diferente e maior: todo o projeto
até aqui (Achados 1-15) modela uma cadeia **feedforward** (mesmo os "dois
caminhos paralelos" do Achado 9 convergem sem realimentação). O anel exige
modelar uma dinâmica de **atrator recorrente** (bump attractor) — um regime
dinâmico que o simulador LIF atual nunca precisou sustentar, com ajuste de
ganho/peso provavelmente muito mais sensível (um atrator de bump instável
não produz "sinal fraco", produz caos ou colapso). Isso é comparável em
esforço a somar os redesigns dos Achados 9, 13 e 14 juntos, só pra chegar
ao ponto de ter uma representação de heading estável — antes de sequer
testar se ela ajuda a jogar Pong.

E o retorno esperado é incerto por construção: a entrada visual no anel é
mais fraca (peso 1-17) que a que já foi validada como insuficiente sozinha
no `DNa10` atual (2,7% do input, Achado 8); e a única saída teria que ser
`DNa02`, uma via de controle de direção de caminhada — não claramente mais
adequada pra "posição vertical de uma bola" que a lateralidade de `DNa10`
já usada (Achado 9 já documentou essa mesma ressalva pra `DNa10`/`GF`).

## Recomendação

**Não prosseguir por ora.** As três cadeias existem com conectividade real
(nenhuma quebra por ausência de dado), então a razão não é anatômica. É
custo/benefício: o elo de entrada é mais fraco que um já testado e
insuficiente; o elo de saída exige abandonar o readout validado nos Achados
9-15 em favor de um neurônio (`DNa02`) sem relação direta com a tarefa; e o
mecanismo central (atrator recorrente) é uma categoria de modelagem nova,
com custo de implementação e ajuste provavelmente maior que todos os
redesigns anteriores do projeto somados. Isso fica registrado como decisão
de escopo, não como beco sem saída anatômico — se o projeto expandir no
futuro pra tarefas onde heading/direção seja mais central que posição de um
objeto na tela, essa auditoria já documenta que a via existe e onde estão
os elos fracos.
