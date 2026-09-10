# FlyPong

Pong controlado por um subgrafo do conectoma **MaleCNS v1.0** de *Drosophila
melanogaster* (Berg et al., "A connectome of the adult male Drosophila central
nervous system", *Cell*, 2026), acessado via [neuPrint](https://neuprint.janelia.org)
(dataset `male-cns:v1.0`).

**Isto NÃO é uma alegação de que a mosca "está jogando" ou de qualquer forma de
consciência/senciência.** É um modelo computacional aproximado: um pequeno
subgrafo do conectoma real, simulado como neurônios leaky integrate-and-fire
(LIF), recebendo um estímulo sensorial derivado da posição da bola e lido por
um readout motor simples. A seção [Validação](#validação-honesta) reporta os
resultados reais dos testes com dados REAIS do MaleCNS v1.0 — o resultado é
**negativo em dois níveis**: (1) não há evidência de que a plasticidade
dopaminérgica implementada produza aprendizado; (2) mesmo o rastreamento
básico "posição da bola → ação motora" não funciona de forma confiável com o
subgrafo de neurônios selecionado, por razões anatômicas e de dados que estão
documentadas abaixo, não escondidas.

## Inspiração / trabalhos relacionados

Este projeto segue o padrão de outros projetos abertos que usam conectomas
reais pra controlar jogos, reaproveitando a ideia geral (estimular sensorial →
propagar no conectoma → ler motor) mas trocando o jogo por Pong (rastreamento
contínuo de 1 objeto, 2 ações motoras) pra reduzir a complexidade:

- [`nftechie/doomfly`](https://github.com/nftechie/doomfly) — Doom controlado
  pelo conectoma, reforço dopaminérgico, relatórios de validação honestos
  (inclusive quando o modelo não aprendeu). É a principal referência de
  metodologia e de honestidade científica deste projeto.
- [`blendi-remade/fly-brain-minecraft`](https://github.com/blendi-remade/fly-brain-minecraft) —
  mod de Minecraft, neurônios LIF, script de fetch via neuPrint anônimo.
- [`evnsnclr/neurocraft-fly-public`](https://github.com/evnsnclr/neurocraft-fly-public) —
  outro mod de Minecraft, foco em atividade neural inspecionável.

## Setup

```bash
pip install -r requirements.txt
```

### 1. Obter o subgrafo do conectoma

**Dados reais** exigem um token pessoal do neuPrint (gratuito):

1. Login em https://neuprint.janelia.org com conta Google.
2. Menu de perfil → "Auth Token" → copiar.
3. Rodar:

```bash
python fetch_connectome.py --token SEU_TOKEN_AQUI
# ou: export NEUPRINT_TOKEN=SEU_TOKEN_AQUI  &&  python fetch_connectome.py
```

**Sem token** (desenvolvimento/teste offline), use o gerador sintético:

```bash
python fetch_connectome.py --synthetic
```

O grafo sintético **não é dado real do conectoma** — é gerado localmente com
conectividade retinotópica aproximada, só pra permitir testar o resto do
pipeline sem acesso à API do neuPrint. Isso fica marcado explicitamente em
`connectome_data/metadata.json` (`"source": "synthetic"`). **O projeto foi
rodado até o fim com dados reais** (token pessoal do usuário) — ver seção de
validação.

Papéis / tipos celulares selecionados:

| role | tipos celulares (regex) | função biológica |
|---|---|---|
| `photoreceptor` | R1-6, R7, R8 | fotorreceptores da retina |
| `interneuron` | L1-L5, Mi1/Mi4/Mi9, Tm1/Tm2/Tm4/Tm9 | lâmina + medula: ponte entre fotorreceptor e T4/T5 (Achado 2 — adicionada depois de descobrir que faltava) |
| `motion` | T4, T5 (subtipos a-d) | detecção de movimento local |
| `object` | LC4, LPLC2 | detecção de objeto em expansão/loom |
| `target` | LC10a | rastreamento de alvo pequeno (Achado 9 — substitui parcialmente `object` como via pra `descending`) |
| `descending` | DNa10, Giant Fiber (GF), DNp01 | vias motoras descendentes (DNa02 removido no Achado 9, ver por quê) |
| `dopaminergic` | PAM*, PPL1* | reforço (PAM ~ recompensa, PPL1 ~ aversivo, cf. Aso et al. 2014) |

**Contagem real final** (dataset `male-cns:v1.0`, depois do Achado 9): **44 989
neurônios, 1 157 952 sinapses** — `photoreceptor`: 1783, `interneuron`:
28 699, `motion`: 13 585, `object`: 311, `target`: 275, `dopaminergic`: 332,
`descending`: **4** (2× GF/DNp01 + 2× DNa10). Grafo sintético (dev/teste
offline): 750 neurônios, 11 681 sinapses. Ver `connectome_data/metadata.json`
pra contagem exata da execução mais recente.

O pool `descending` real continua pequeno (4 neurônios) — DNa10, Giant Fiber e
DNp01 são vias com poucas cópias (tipicamente 1-2 por lado do corpo),
diferente de interneurônios do lobo óptico que se repetem por coluna
retinotópica. Isso deixa o readout motor com só 2 neurônios por grupo
("sobe"/"desce"), mas agora agrupados por **lateralidade real**
(`_R`/`_L`), não por índice arbitrário — ver Achado 9.

### 2. Rodar o jogo

```bash
python main.py                              # janela com Pong + painel ao vivo
python main.py --headless --frames 30000    # sem janela, so gera relatorio de validacao em runs/
```

## Arquitetura

```
fetch_connectome.py   # baixa/gera o subgrafo do conectoma (neurons.parquet, edges.parquet)
sim/neurons.py         # modelo LIF vetorizado (NumPy)
sim/network.py          # carrega o grafo, propaga spikes (scipy.sparse), plasticidade de 3 fatores
game/pong.py            # Pong (paddle esquerdo = conectoma, direito = IA que rastreia a bola)
game/sensory_map.py    # posicao Y da bola -> estimulo gaussiano nos fotorreceptores
game/motor_read.py      # diferenca de disparo (grupo "desce" - grupo "sobe") -> acao
dashboard.py            # painel pygame: placar, raster de spikes, peso sinaptico plastico
main.py                 # loop principal + modo headless de validacao
```

Fluxo por frame: `ball_to_photoreceptor_stimulus` → `ConnectomeNetwork.step`
(N substeps, propagando `photoreceptor → interneuron → {motion → object,
target}` em dois caminhos paralelos que convergem em `descending` — ver
Achado 9) → `read_motor_action` → `PongGame.step(action)` → se houve
contato ou perda de bola, aplica um pulso de corrente nos neurônios
dopaminérgicos, cuja atividade resultante modula (regra de três fatores tipo
STDP, cf. Frémaux & Gerstner 2016) **apenas** os pesos das sinapses
`motion`/`object`/`target` → `descending` marcadas como plásticas — o resto do
grafo é fixo. A dopamina é modelada como sinal global (volume transmission),
não exige aresta anatômica dopaminérgico→alvo.

A divisão do pool `descending` em dois grupos de leitura ("sobe"/"desce") usa
**lateralidade real** (sufixo `_R`/`_L` do campo `instance`, ex. `DNa10_R`) —
neurônios do lado direito num grupo, do lado esquerdo no outro
(`sim/network.py`, `_compute_motor_groups`). Isso substituiu o esquema
original de metade/metade por índice depois que o Achado 6 mostrou que ele
podia, por coincidência, isolar um subconjunto anatomicamente desconectado
num grupo inteiro. Mapear lateralidade (o que esses neurônios controlam de
fato — direção de caminhada) em "sobe"/"desce" de um paddle vertical **ainda
é uma escolha de engenharia, não um fato biológico** (Achado 9). Mesmo com a
lateralidade real, os dois grupos podem ter viés estrutural de excitabilidade
sem relação com a posição da bola; o código mede esse viés numa fase de
calibração (bola parada no centro, 200 frames) e o subtrai do sinal motor
(`game/motor_read.py`, parâmetro `baseline_diff`).

A posição de cada fotorreceptor na "retina" 1D simulada (usada por
`ball_to_photoreceptor_stimulus`) vem do centróide de suas sinapses de saída,
projetado no eixo de maior variância (1º componente principal via SVD) —
ver Achado 3 abaixo pra por que isso foi necessário e o que significa/não
significa.

## Validação honesta

Esta seção documenta, na ordem em que foram encontrados, cinco problemas reais
que surgiram ao rodar o projeto com dados reais do MaleCNS v1.0 — três eram
bugs corrigíveis, dois são limitações estruturais que não têm uma correção
simples. Nenhum foi escondido; o resultado final é negativo e isso é
reportado como tal, seguindo o padrão do `doomfly`.

**Métrica**: taxa de rebatida = fração das tentativas de retorno (contato
paddle-bola bem-sucedido vs. bola perdida) numa janela móvel de 20 tentativas.
Não é "taxa de pontos ganhos": o oponente da direita é uma IA que rastreia a
bola com precisão total a cada frame, então vencer pontos contra ela é
praticamente impossível por construção do jogo, independente da qualidade do
controle neural.

**Achado 1 (bug de regex — corrigido):** na primeira execução com token do
neuPrint, os grupos `motion` e `dopaminergic` vieram com **0 neurônios**.
`neuprint-python` traduz `regex=True` para o operador `=~` do Cypher/Neo4j,
que exige **fullmatch** (a string inteira precisa bater com o padrão) — não é
busca de substring. Padrões como `^T4|^T5` só batiam com a string exata
`"T4"`, que não existe como tipo (só os subtipos, tipo `"T4a"`). Corrigido pra
`T4.*|T5.*` etc. em `fetch_connectome.py`.

**Achado 2 (lacuna anatômica — corrigido adicionando uma camada):** mesmo
depois do Achado 1, o subgrafo `photoreceptor + motion + object + descending +
dopaminergic` tinha **zero arestas entre `photoreceptor` e qualquer outro
papel** (auditoria direta das arestas por par de papéis confirmou isso).
Causa: no circuito visual real da mosca, R1-8 não fazem sinapse direta em
T4/T5 — o sinal passa antes pela lâmina (L1-L5) e medula (Mi1/Mi4/Mi9,
Tm1/Tm2/Tm4/Tm9) (Takemura et al. 2013, 2017). A especificação original do
projeto não incluía essas interneurônios. Corrigido adicionando o papel
`interneuron` (ver tabela acima) — depois disso, sinal passou a chegar de
fato até `descending` (confirmado por spike count > 0 em todas as camadas).

**Achado 3 (retinotopia ausente nos metadados — parcialmente corrigido):**
com o circuito anatomicamente completo, a diferença de disparo entre os
grupos "sobe"/"desce" ainda não dependia da posição Y da bola — vinha
**constante**, porque o código original mapeava posição do fotorreceptor na
retina simplesmente pela ordem em que o neuPrint devolveu a lista (que não
tem relação com posição espacial real). Primeira tentativa de correção usou
`somaLocation` (posição do soma), mas só 11 dos 1783 fotorreceptores tinham
esse campo preenchido (0.6% — a maioria dos fotorreceptores não tem soma
segmentado nessa reconstrução). Segunda tentativa, que funcionou em termos de
cobertura: usar o **centróide das sinapses de saída** de cada fotorreceptor
(`fetch_synapses`, sempre presente), projetado no eixo de maior variância via
PCA — 1783/1783 fotorreceptores cobertos. Mas mesmo com isso, a relação entre
posição Y da bola e diferença de disparo sobe/desce não é um gradiente limpo:
é ~0 pra bola no meio da tela e só ativa (com o **mesmo valor** pra topo e
para baixo, sem distinguir direção) quando a bola está bem nos extremos. Não
foi possível confirmar que o eixo de PCA encontrado corresponde de fato ao
eixo dorso-ventral do olho (não verificamos a orientação real dos eixos do
dataset) — é plausível que a distribuição de fotorreceptores ao longo desse
eixo seja bimodal (dois clusters, não um gradiente contínuo), o que explicaria
o padrão observado.

**Achado 4 (bug de jogo — corrigido):** o fator de aceleração da bola por
rebatida (`ball_vx *= 1.05`) não tinha teto; em runs headless longas isso
inflava a velocidade exponencialmente e distorcia a métrica de tentativas.
Corrigido com `BALL_SPEED_MAX = 2.5x` em `game/pong.py`.

**Achado 5 (bug de plasticidade — corrigido):** o clip de peso sináptico
plástico usava limites absolutos (`[-1, 5]`) calibrados pro grafo sintético
(pesos pequenos). Com dados reais o peso médio já nasce em ~10.8 (contagem de
sinapse × ganho), então o `clip()` colapsava o peso na primeira atualização
elegível **mesmo com a taxa de aprendizado zerada** — invalidando qualquer
comparação "com plasticidade vs. controle sem plasticidade" feita antes dessa
correção. Corrigido pra clipar o **delta acumulado** (±3.0) em torno do peso
original de cada sinapse, não o peso absoluto (`sim/network.py`,
`PLASTIC_DELTA_MAX`).

### Resultado final (dados reais, depois de todas as correções acima)

| condição | seed | tentativas | taxa início→fim (delta) | peso plástico início→fim |
|---|---|---|---|---|
| plasticidade (LR=0.02) | 42 | 60 | 0.48 → 0.23 (-0.24) | 10.83 → 7.83 |
| controle (LR=0) | 42 | 60 | 0.48 → 0.23 (-0.24) | 10.83 → 10.83 |
| plasticidade (LR=0.02) | 1 | 61 | 0.22 → 0.22 (+0.01) | 10.83 → 7.83 |
| controle (LR=0) | 1 | 61 | 0.22 → 0.22 (+0.01) | 10.83 → 10.83 |

O controle confirma que o peso realmente fica travado no valor original
quando `LR=0` (correção do achado 5 funcionando) e que a plasticidade
realmente muda o peso quando ligada (bateu no teto do clip, -3.0, nas duas
seeds). Mesmo assim, **a curva de taxa de rebatida é idêntica bit a bit entre
plasticidade ligada e desligada, seed a seed**.

**Achado 6 (causa raiz, investigação em três hipóteses):** por que uma mudança
de peso real não tem efeito nenhum? Testamos três hipóteses concorrentes, na
ordem abaixo, parando quando uma explicou os dados.

- *Hipótese 1 — cache de peso obsoleto na propagação:* refutada. `self.W` é
  uma única `csr_matrix` mutada in-place (`self.W.data[...] = ...`); não há
  cópia separada. Confirmado empiricamente instrumentando `step()`: o peso
  somado usado no `dot()` da chamada N é sempre exatamente igual ao peso
  oficial no fim da chamada N-1 (`used[k] == official[k-1]` para todo k,
  checado numericamente).
- *Hipótese 2 — massa sináptica fixa de `motion` afogando as sinapses
  plásticas:* refutada pelos dados. Auditando `edges.parquet` diretamente:
  **não existe nenhuma aresta `motion → descending`** no subgrafo real (0
  arestas). As 311 sinapses `object → descending` já são 100% da entrada
  feedforward de `descending` (o resto é só um auto-loop `descending →
  descending` de 2 sinapses, peso 2). Não tem massa fixa nenhuma pra "afogar"
  o sinal plástico, porque não existe massa fixa competindo nesse caminho.
- *Hipótese 3 — teste de sanidade independente de dopamina:* forçamos o peso
  das 311 sinapses plásticas pra **5x o valor original**, sem nenhuma
  plasticidade (`LR=0` nas duas condições), e comparamos contra o peso
  original — isolando o efeito do peso em si, sem depender do gatilho
  dopaminérgico. **Resultado idêntico bit a bit de novo**, nas duas seeds.
  Isso descarta "dopamina fraca/rara demais" como explicação — o problema não
  é o gatilho, é a arquitetura.

Investigando mais fundo pra achar a causa desse resultado da Hipótese 3:
contamos quantas das 311 sinapses `object→descending` chegam em cada um dos 4
neurônios `descending` individualmente. Resultado:

| bodyId | tipo | sinapses de `object` recebidas |
|---|---|---|
| 10001 | DNp01 (Giant Fiber, lado direito) | 146 |
| 10010 | DNp01 (Giant Fiber, lado esquerdo) | 165 |
| 10360 | DNa02 (lado direito) | **0** |
| 523769 | DNa02 (lado esquerdo) | **0** |

E `motor_up_idx`/`motor_down_idx` são só a metade de menor/maior índice desse
array de 4 — que por acaso (índices vêm ordenados por bodyId) colocou as duas
Giant Fiber em `motor_up` e as duas DNa02 em `motor_down`. Isso bate
exatamente com o observado: só o neurônio 10001 dispara de fato (confirmado
por spike count), os outros 3 nunca disparam nem uma vez, com pesos 1x ou 5x.
**O grupo "desce" é estruturalmente mudo, sempre, independente de qualquer
peso** — porque as duas DNa02 não recebem nenhuma sinapse de `object` (nem de
`motion`, achado 2) nesse subgrafo. Nenhuma mudança de peso nas sinapses
plásticas pode fazer o grupo "desce" responder, porque ele não está
conectado ao resto do circuito que selecionamos.

Isso também faz sentido biologicamente: Giant Fiber é o alvo clássico do
circuito de escape por loom (LC4/LPLC2 → GF, von Reyn et al. 2014) — é
esperado que receba input de `object`. DNa02 é um neurônio de controle de
direção de caminhada, tipicamente dirigido por outras vias (não
especificamente LC4/LPLC2) que não foram incluídas nesta seleção de tipos
celulares. A anatomia real está correta; o problema é que nossa divisão
arbitrária "metade/metade por índice" em grupos motores colocou, por
coincidência, um circuito funcional inteiro (GF) de um lado e um circuito
funcionalmente desconectado (nesta seleção) do outro.

**Conclusão honesta:**
1. **Não há evidência de que a plasticidade dopaminérgica implementada
   produza aprendizado**, com dados sintéticos ou reais — confirmado agora
   não só pela ausência de efeito, mas pela causa raiz identificada: o grupo
   motor "desce" é estruturalmente mudo neste subgrafo, então nenhuma
   plasticidade nas sinapses selecionadas poderia jamais mudar o resultado.
2. **O rastreamento sensório-motor básico (bola → ação) não funciona de forma
   confiável com dados reais**, mesmo depois de fechar o circuito anatômico
   (achado 2) e adicionar posição espacial real aos fotorreceptores (achado
   3) — e agora sabemos por que também no lado motor: a divisão "sobe/desce"
   por índice não corresponde a nenhuma divisão funcional real, e só um dos 4
   neurônios descendentes tem qualquer conexão com o resto do circuito
   selecionado.

Isso foi reportado como resultado final da primeira rodada de validação — mas
o próprio texto acima recomendava, como trabalho futuro, "agrupar sobe/desce
por conectividade real" e "selecionar mais neurônios descendentes". A seção
seguinte (Achados 9 e 10) documenta uma segunda rodada que fez exatamente
isso. **A tabela e a conclusão acima ficam mantidas como registro histórico
do que foi encontrado naquele momento — não foram os números finais do
projeto.**

### Achado 9 (redesign do circuito, depois de investigar 3 hipóteses)

Antes de redesenhar, uma investigação em três hipóteses (documentada em
detalhe no histórico da sessão, resumida aqui) descartou explicações mais
simples pra "plasticidade sem efeito": não era cache de peso obsoleto
(confirmado por instrumentação direta do `step()`), não era massa sináptica
fixa de `motion` afogando o sinal plástico (auditoria de `edges.parquet`
mostrou **zero** arestas `motion→descending` — as sinapses plásticas já eram
100% da entrada feedforward), e forçar o peso a 5x manualmente (sem
plasticidade) não mudava nada. A causa raiz, encontrada auditando
conectividade real via `neuprint-python` (**Achado 8**, detalhado em
[`docs/achado-8-auditoria-dna02.md`](docs/achado-8-auditoria-dna02.md)): das
4 `descending` originais, só as 2 Giant Fiber recebiam qualquer sinapse do
resto do circuito — as 2 DNa02 recebem **milhares** de sinapses reais, mas
**nenhuma** de `object`/`motion`; são dominadas por circuito de heading do
central complex (LAL, AOTU, PS0xx, PFL3), fora do escopo deste projeto. A
divisão arbitrária sobe/desce por índice tinha colocado, por coincidência, as
2 GF (que respondem) de um lado e as 2 DNa02 (mudas) do outro.

A hipótese de trabalho testada — que `LC10a` (neurônio de rastreamento de
alvo durante corte, Ribeiro et al. 2018) seria um substituto melhor que
`LC4`/`LPLC2` (loom/fuga) — foi **parcialmente confirmada e parcialmente
refutada**: `LC10a` tem **zero** sinapses diretas em `DNa02` ou `GF` (a
hipótese original, literal, é refutada). Mas expandindo a busca pros alvos
reais de `LC10a` (275 neurônios auditados), o candidato descendente mais forte
encontrado foi **`DNa10`** (801 sinapses via só 2 neurônios, um par bilateral
`DNa10_R`/`DNa10_L`) — bem acima de qualquer outro candidato descendente
testado (`DNp11`, `DNa16`, `DNp63`, todos <200). `LC10a`, por sua vez, recebe
2657 sinapses de `Tm4` (já no papel `interneuron`), confirmando a cadeia
completa `photoreceptor → interneuron(Tm4) → target(LC10a) → descending(DNa10)`
com aresta real em todo elo (não só nas pontas — auditado explicitamente).

Mudanças de código: `DNa02` removido do papel `descending`, `DNa10`
adicionado (regex `descending`: `DNa10.*|GF|GiantFiber.*|DNp01.*`); novo papel
`target` (`LC10a.*`) adicionado; `plastic_mask` generalizado pra
`motion|object|target → descending` (antes hardcoded só `motion|object`); e o
split motor sobe/desce (`sim/network.py`, `_compute_motor_groups`) passou a
usar **lateralidade real** (sufixo `_R`/`_L` do campo `instance`) — juntando
GF+DNa10 do lado direito num grupo, GF+DNa10 do lado esquerdo no outro — em
vez de metade/metade por índice. **Isso é uma escolha de engenharia
explícita, não um fato biológico**: mapear lateralidade de controle de
caminhada em "sobe"/"desce" de um paddle vertical não é o que esses
neurônios codificam naturalmente.

Depois do redesign: subgrafo real cresceu pra 44 989 neurônios / 1 157 952
sinapses (`target`: 275). Auditoria de arestas confirmou `target→descending`
= 93 sinapses (peso 801, bate com a auditoria) e `object→descending` = 311
(peso 11224). Verificação direta de spikes: **os 4 neurônios `descending`
agora disparam** (antes, 3 de 4 nunca disparavam — Achado 6). A diferença de
disparo sobe/desce também passou a ter **sinal** dependente da posição da
bola (+0.98 com a bola no topo, -1.21 embaixo — antes era o mesmo valor pros
dois extremos, sem direção, Achado 3).

### Achado 10 (resultado do experimento de aprendizado com o circuito novo)

Rodando o mesmo protocolo de controle (plasticidade `LR=0.02` vs. `LR=0`),
4 seeds, 15 000 frames cada, com dados reais e o circuito redesenhado:

| seed | delta com plasticidade (LR=0.02) | delta controle (LR=0) | peso final (plástico) |
|---|---|---|---|
| 42 | -0.058 | -0.245 | 10.23 (subiu de 8.93) |
| 1  | +0.097 | +0.248 | 7.16 (desceu) |
| 2  | -0.070 | -0.203 | 6.61 (desceu) |
| 3  | +0.030 | +0.200 | 6.03 (desceu) |

**Pela primeira vez no projeto, plasticidade ligada e desligada produzem
resultados diferentes** (antes: idênticos bit a bit, com dados sintéticos e
reais, Achados 4-6). O redesign teve efeito real e mensurável.

Mas o padrão **não é** o de aprendizado direcionado a um objetivo: em nenhuma
das 4 seeds a plasticidade inverteu o sinal do delta (seeds 42 e 2 pioram nas
duas condições; seeds 1 e 3 melhoram nas duas). O que a plasticidade faz, de
forma consistente nas 4 seeds, é **reduzir a magnitude** do delta em relação
ao controle — tanto quando o controle estava melhorando quanto quando estava
piorando. Isso parece mais um efeito de atenuação/estabilização do que
reforço: se os eventos de "bola perdida" são mais frequentes que "bola
rebatida" na maior parte de uma rodada típica, o termo de dopamina negativa
domina ao longo do tempo, então o peso da via `object`/`target → descending`
tende a cair (3 das 4 seeds terminam com peso menor que o inicial) — um
circuito com ganho mais fraco produz oscilações menores no readout motor,
tanto pra cima quanto pra baixo, o que atenuaria o delta nas duas direções
sem exigir nenhum reforço "inteligente". Essa é uma hipótese mecanicista
plausível pra explicar o padrão observado, **não foi confirmada com
experimentos adicionais** por limite de tempo desta rodada — fica como
trabalho futuro (ex.: comparar histograma de ações tomadas entre condições).

Teste estatístico simples: com n=4 seeds e as 4 favorecendo a mesma direção
(|delta| menor com plasticidade em 4 de 4), um teste de sinal dá p=0.0625
(unicaudal) / p=0.125 (bicaudal) sob a hipótese nula de 50/50 — **sugestivo,
mas não estatisticamente significativo** com uma amostra tão pequena. Testes
com taxas de aprendizado adicionais (0.005, 0.05) e mais seeds, planejados na
especificação desta rodada, não foram executados por limite de tempo — fica
documentado aqui como não feito, não como negativo.

**Conclusão honesta desta rodada:** o redesign resolveu o problema estrutural
do Achado 6 (circuito agora responde nos dois lados, com direção correta) e
produziu o primeiro efeito mensurável e reprodutível de plasticidade em todo
o projeto. Mas esse efeito, pelo padrão observado, parece ser uma atenuação
de magnitude ligada à assimetria natural entre eventos de reforço positivo e
negativo, não evidência de que o circuito aprendeu a jogar melhor. Não
inflamos isso como "sucesso" — é um resultado real, mensurável, mas
provavelmente não é o tipo de aprendizado que o projeto original visava.

Reproduzir:

```bash
python fetch_connectome.py --token SEU_TOKEN   # ou --synthetic
python main.py --headless --frames 15000
```

## Limitações conhecidas

- **(Histórico, corrigido no Achado 9)** O pool `descending` original tinha só
  4 neurônios (2 GF + 2 DNa02), e só a metade (GF) recebia qualquer sinapse do
  resto do circuito selecionado — a divisão "sobe"/"desce" por índice colocava
  as duas GF (que disparavam) de um lado e as duas DNa02 (mudas) do outro
  (Achado 6). Corrigido substituindo DNa02 por DNa10 (Achado 9) e trocando o
  split por índice por lateralidade real (`_R`/`_L`) — os 4 neurônios
  descendentes atuais disparam e respondem com direção correta à posição da
  bola.
- LC10a contribui só ~2.7% do input total de DNa10 (achado-8) — os outros
  ~97% vêm de vias fora do escopo deste projeto (PLP213, AOTU, IB, PS0xx),
  então DNa10 não é um canal "limpo" dedicado à visão, mesmo depois do
  redesign.
- A posição retinotópica dos fotorreceptores (Achado 3) é uma aproximação via
  PCA de coordenadas de sinapse, não uma medida validada contra a anatomia
  real do olho — pode não corresponder ao eixo dorso-ventral verdadeiro.
- Mapear lateralidade (esquerda/direita, o que DNa10/GF codificam de fato —
  controle de direção de caminhada) em "sobe"/"desce" de um paddle vertical
  (Achado 9) é uma escolha de engenharia, não um fato biológico.
- O efeito de plasticidade encontrado no Achado 10 (redução de magnitude do
  delta, consistente em 4/4 seeds, p=0.125 bicaudal) não é estatisticamente
  conclusivo com amostra tão pequena, e a hipótese mecanicista proposta
  (atenuação por decaimento de peso, não reforço direcionado) não foi
  verificada com experimentos adicionais.
- O oponente (paddle direito) é uma IA que nunca erra; serve só pra manter a
  bola em jogo, não é um adversário real.

## Citação

Berg et al., "A connectome of the adult male Drosophila central nervous
system", *Cell*, 2026 (MaleCNS v1.0). Dados acessados via neuPrint
(neuprint.janelia.org, dataset `male-cns:v1.0`).
