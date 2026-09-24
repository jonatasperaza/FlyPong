# FlyPong

Pong controlado por um subgrafo do conectoma **MaleCNS v1.0** de *Drosophila
melanogaster* (Berg et al., "A connectome of the adult male Drosophila central
nervous system", *Cell*, 2026), acessado via [neuPrint](https://neuprint.janelia.org)
(dataset `male-cns:v1.0`).

Isto não é uma alegação de que a mosca "está jogando" ou de qualquer forma de
consciência ou senciência. É um modelo computacional aproximado: um pequeno
subgrafo do conectoma real, simulado como neurônios leaky integrate-and-fire
(LIF), recebendo um estímulo sensorial derivado da posição da bola e lido por
um readout motor simples. A seção [Validação](#validação-honesta) reporta os
resultados reais dos testes com dados reais do MaleCNS v1.0, ao longo de 17
achados: bugs encontrados e corrigidos, hipóteses testadas e descartadas ou
confirmadas, uma correção pós-hoc de um resultado que parecia bom demais e
não era (Achado 13), e duas tentativas de corrigir um viés de plasticidade
que não funcionaram (Achado 14).

Resumo final: o circuito real consegue rastrear a bola e responder com
direção correta (Achado 9), e um sinal forte fixo no ponto certo do circuito
produz uma melhora fraca mas real (Achado 13, Fase 1), então capacidade
sensorial bruta não parece ser o gargalo dominante. O que não funciona é a
plasticidade dopaminérgica: não há evidência de que ela produza aprendizado
com dados reais em nenhuma das variantes testadas (reforço esparso, denso,
ou um canal sintético isolado já forte o bastante). O Achado 13 (Fase 2)
aponta uma causa mecanicista específica para isso, a regra de três fatores
enfraquece sinapses sistematicamente porque eventos de reforço negativo são
mais frequentes que positivos numa rodada típica. Duas correções óbvias pra
esse viés (normalizar pela frequência de eventos, ou usar uma linha de base
de reforço não-zero) foram implementadas e testadas no Achado 14: uma piora
o problema de forma estatisticamente significativa, a outra não tem efeito
líquido distinguível de ruído. O mecanismo de atribuição de crédito da regra
de três fatores continua sem correção funcional conhecida.

## Para quem quer replicar

Se você chegou aqui querendo fazer um projeto parecido (conectoma real
controlando algum sistema, jogo ou não), leia
[`docs/GUIA-PARA-PROJETOS-SIMILARES.md`](docs/GUIA-PARA-PROJETOS-SIMILARES.md)
antes de escrever código: é um checklist prático extraído dos bugs e
armadilhas reais que apareceram construindo este projeto (regex fullmatch do
neuPrint, camadas anatômicas escondidas, dados espaciais que não existem
prontos, RNGs acoplados sem querer, etc.), organizado por etapa do trabalho,
não pela ordem cronológica dos 18 achados abaixo.

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

O grafo sintético não é dado real do conectoma: é gerado localmente com
conectividade retinotópica aproximada, só pra permitir testar o resto do
pipeline sem acesso à API do neuPrint. Isso fica marcado explicitamente em
`connectome_data/metadata.json` (`"source": "synthetic"`). O projeto foi
rodado até o fim com dados reais (token pessoal do usuário), ver seção de
validação.

Papéis / tipos celulares selecionados:

| role | tipos celulares (regex) | função biológica |
|---|---|---|
| `photoreceptor` | R1-6, R7, R8 | fotorreceptores da retina |
| `interneuron` | L1-L5, Mi1/Mi4/Mi9, Tm1/Tm2/Tm4/Tm9 | lâmina + medula: ponte entre fotorreceptor e T4/T5 (Achado 2, adicionada depois de descobrir que faltava) |
| `motion` | T4, T5 (subtipos a-d) | detecção de movimento local |
| `object` | LC4, LPLC2 | detecção de objeto em expansão/loom |
| `target` | LC10a | rastreamento de alvo pequeno (Achado 9, substitui parcialmente `object` como via pra `descending`) |
| `descending` | DNa10, Giant Fiber (GF), DNp01 | vias motoras descendentes (DNa02 removido no Achado 9, ver por quê) |
| `dopaminergic` | PAM*, PPL1* | reforço (PAM ~ recompensa, PPL1 ~ aversivo, cf. Aso et al. 2014) |

Contagem real final (dataset `male-cns:v1.0`, depois do Achado 9): **44 989
neurônios, 1 157 952 sinapses**. Por papel: `photoreceptor`: 1783,
`interneuron`: 28 699, `motion`: 13 585, `object`: 311, `target`: 275,
`dopaminergic`: 332, `descending`: 4 (2× GF/DNp01 + 2× DNa10). Grafo sintético
(dev/teste offline): 750 neurônios, 11 681 sinapses. Ver
`connectome_data/metadata.json` pra contagem exata da execução mais recente.

O pool `descending` real continua pequeno (4 neurônios): DNa10, Giant Fiber e
DNp01 são vias com poucas cópias (tipicamente 1-2 por lado do corpo),
diferente de interneurônios do lobo óptico que se repetem por coluna
retinotópica. Isso deixa o readout motor com só 2 neurônios por grupo
("sobe"/"desce"), mas agora agrupados por lateralidade real (`_R`/`_L`), não
por índice arbitrário (ver Achado 9).

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
target}` em dois caminhos paralelos que convergem em `descending`, ver
Achado 9) → `read_motor_action` → `PongGame.step(action)`. Se houve contato
ou perda de bola, um pulso de corrente é aplicado nos neurônios
dopaminérgicos, cuja atividade resultante modula (regra de três fatores tipo
STDP, cf. Frémaux & Gerstner 2016) apenas os pesos das sinapses
`motion`/`object`/`target` → `descending` marcadas como plásticas; o resto do
grafo é fixo. A dopamina é modelada como sinal global (volume transmission) e
não exige aresta anatômica dopaminérgico→alvo.

A divisão do pool `descending` em dois grupos de leitura ("sobe"/"desce") usa
lateralidade real (sufixo `_R`/`_L` do campo `instance`, ex. `DNa10_R`):
neurônios do lado direito num grupo, do lado esquerdo no outro
(`sim/network.py`, `_compute_motor_groups`). Isso substituiu o esquema
original de metade/metade por índice depois que o Achado 6 mostrou que ele
podia, por coincidência, isolar um subconjunto anatomicamente desconectado
num grupo inteiro. Mapear lateralidade (o que esses neurônios controlam de
fato, direção de caminhada) em "sobe"/"desce" de um paddle vertical continua
sendo uma escolha de engenharia, não um fato biológico (Achado 9). Mesmo com
a lateralidade real, os dois grupos podem ter viés estrutural de
excitabilidade sem relação com a posição da bola; o código mede esse viés
numa fase de calibração (bola parada no centro, 200 frames) e o subtrai do
sinal motor (`game/motor_read.py`, parâmetro `baseline_diff`).

A posição de cada fotorreceptor na "retina" 1D simulada (usada por
`ball_to_photoreceptor_stimulus`) vem do centróide de suas sinapses de saída,
projetado no eixo de maior variância (1º componente principal via SVD).
Ver Achado 3 abaixo pra por que isso foi necessário e o que significa, ou
não significa.

## Validação honesta

Esta seção documenta, na ordem em que foram encontrados, cinco problemas reais
que surgiram ao rodar o projeto com dados reais do MaleCNS v1.0: três eram
bugs corrigíveis, dois são limitações estruturais que não têm uma correção
simples. Nenhum foi escondido; o resultado final é negativo e é reportado
como tal, seguindo o padrão do `doomfly`.

**Métrica**: taxa de rebatida = fração das tentativas de retorno (contato
paddle-bola bem-sucedido vs. bola perdida) numa janela móvel de 20 tentativas.
Não é "taxa de pontos ganhos": o oponente da direita é uma IA que rastreia a
bola com precisão total a cada frame, então vencer pontos contra ela é
praticamente impossível por construção do jogo, independente da qualidade do
controle neural.

**Achado 1 (bug de regex, corrigido):** na primeira execução com token do
neuPrint, os grupos `motion` e `dopaminergic` vieram com 0 neurônios.
`neuprint-python` traduz `regex=True` para o operador `=~` do Cypher/Neo4j,
que exige fullmatch (a string inteira precisa bater com o padrão), não busca
de substring. Padrões como `^T4|^T5` só batiam com a string exata `"T4"`,
que não existe como tipo (só os subtipos, tipo `"T4a"`). Corrigido pra
`T4.*|T5.*` etc. em `fetch_connectome.py`.

**Achado 2 (lacuna anatômica, corrigido adicionando uma camada):** mesmo
depois do Achado 1, o subgrafo `photoreceptor + motion + object + descending +
dopaminergic` tinha zero arestas entre `photoreceptor` e qualquer outro papel
(auditoria direta das arestas por par de papéis confirmou isso). A causa: no
circuito visual real da mosca, R1-8 não fazem sinapse direta em T4/T5, o
sinal passa antes pela lâmina (L1-L5) e medula (Mi1/Mi4/Mi9, Tm1/Tm2/Tm4/Tm9)
(Takemura et al. 2013, 2017). A especificação original do projeto não
incluía essas interneurônios. Corrigido adicionando o papel `interneuron`
(ver tabela acima); depois disso, sinal passou a chegar até `descending`
(confirmado por spike count maior que zero em todas as camadas).

**Achado 3 (retinotopia ausente nos metadados, parcialmente corrigido):**
com o circuito anatomicamente completo, a diferença de disparo entre os
grupos "sobe"/"desce" ainda não dependia da posição Y da bola, vinha
constante, porque o código original mapeava posição do fotorreceptor na
retina simplesmente pela ordem em que o neuPrint devolveu a lista (que não
tem relação com posição espacial real). A primeira tentativa de correção
usou `somaLocation` (posição do soma), mas só 11 dos 1783 fotorreceptores
tinham esse campo preenchido (0.6%; a maioria dos fotorreceptores não tem
soma segmentado nessa reconstrução). A segunda tentativa funcionou em termos
de cobertura: usar o centróide das sinapses de saída de cada fotorreceptor
(`fetch_synapses`, sempre presente), projetado no eixo de maior variância via
PCA, cobrindo 1783 de 1783 fotorreceptores. Mesmo assim, a relação entre
posição Y da bola e diferença de disparo sobe/desce não é um gradiente
limpo: fica perto de zero pra bola no meio da tela e só ativa (com o mesmo
valor pro topo e para baixo, sem distinguir direção) quando a bola está bem
nos extremos. Não foi possível confirmar que o eixo de PCA encontrado
corresponde ao eixo dorso-ventral do olho, já que a orientação real dos
eixos do dataset não foi verificada. É plausível que a distribuição de
fotorreceptores ao longo desse eixo seja bimodal (dois clusters, não um
gradiente contínuo), o que explicaria o padrão observado.

**Achado 4 (bug de jogo, corrigido):** o fator de aceleração da bola por
rebatida (`ball_vx *= 1.05`) não tinha teto; em runs headless longas isso
inflava a velocidade exponencialmente e distorcia a métrica de tentativas.
Corrigido com `BALL_SPEED_MAX = 2.5x` em `game/pong.py`.

**Achado 5 (bug de plasticidade, corrigido):** o clip de peso sináptico
plástico usava limites absolutos (`[-1, 5]`) calibrados pro grafo sintético,
que tem pesos pequenos. Com dados reais o peso médio já nasce em ~10.8
(contagem de sinapse × ganho), então o `clip()` colapsava o peso na primeira
atualização elegível mesmo com a taxa de aprendizado zerada, invalidando
qualquer comparação "com plasticidade vs. controle sem plasticidade" feita
antes dessa correção. Corrigido pra clipar o delta acumulado (±3.0) em torno
do peso original de cada sinapse, não o peso absoluto (`sim/network.py`,
`PLASTIC_DELTA_MAX`).

### Resultado final (dados reais, depois de todas as correções acima)

| condição | seed | tentativas | taxa início→fim (delta) | peso plástico início→fim |
|---|---|---|---|---|
| plasticidade (LR=0.02) | 42 | 60 | 0.48 → 0.23 (-0.24) | 10.83 → 7.83 |
| controle (LR=0) | 42 | 60 | 0.48 → 0.23 (-0.24) | 10.83 → 10.83 |
| plasticidade (LR=0.02) | 1 | 61 | 0.22 → 0.22 (+0.01) | 10.83 → 7.83 |
| controle (LR=0) | 1 | 61 | 0.22 → 0.22 (+0.01) | 10.83 → 10.83 |

O controle confirma que o peso fica travado no valor original quando `LR=0`
(a correção do achado 5 funcionando) e que a plasticidade muda o peso quando
ligada, batendo no teto do clip (-3.0) nas duas seeds. Mesmo assim, a curva
de taxa de rebatida é idêntica bit a bit entre plasticidade ligada e
desligada, seed a seed.

**Achado 6 (causa raiz, investigação em três hipóteses):** por que uma
mudança de peso real não tem efeito nenhum? Três hipóteses concorrentes
foram testadas, na ordem abaixo, parando quando uma explicou os dados.

- *Hipótese 1, cache de peso obsoleto na propagação*: refutada. `self.W` é
  uma única `csr_matrix` mutada in-place (`self.W.data[...] = ...`); não há
  cópia separada. Confirmado empiricamente instrumentando `step()`: o peso
  somado usado no `dot()` da chamada N é sempre exatamente igual ao peso
  oficial no fim da chamada N-1 (`used[k] == official[k-1]` para todo k,
  checado numericamente).
- *Hipótese 2, massa sináptica fixa de `motion` afogando as sinapses
  plásticas*: refutada pelos dados. Auditando `edges.parquet` diretamente,
  não existe nenhuma aresta `motion → descending` no subgrafo real (0
  arestas). As 311 sinapses `object → descending` já são 100% da entrada
  feedforward de `descending` (o resto é só um auto-loop `descending →
  descending` de 2 sinapses, peso 2). Não existe massa fixa competindo nesse
  caminho, então não há o que "afogar" o sinal plástico.
- *Hipótese 3, teste de sanidade independente de dopamina*: o peso das 311
  sinapses plásticas foi forçado pra 5x o valor original, sem nenhuma
  plasticidade (`LR=0` nas duas condições), e comparado contra o peso
  original, isolando o efeito do peso em si sem depender do gatilho
  dopaminérgico. Resultado idêntico bit a bit de novo, nas duas seeds. Isso
  descarta "dopamina fraca ou rara demais" como explicação: o problema não é
  o gatilho, é a arquitetura.

Investigando mais fundo a causa desse resultado da Hipótese 3, contamos
quantas das 311 sinapses `object→descending` chegam em cada um dos 4
neurônios `descending` individualmente:

| bodyId | tipo | sinapses de `object` recebidas |
|---|---|---|
| 10001 | DNp01 (Giant Fiber, lado direito) | 146 |
| 10010 | DNp01 (Giant Fiber, lado esquerdo) | 165 |
| 10360 | DNa02 (lado direito) | **0** |
| 523769 | DNa02 (lado esquerdo) | **0** |

`motor_up_idx`/`motor_down_idx` são só a metade de menor/maior índice desse
array de 4, o que por acaso (os índices vêm ordenados por bodyId) colocou as
duas Giant Fiber em `motor_up` e as duas DNa02 em `motor_down`. Isso bate
exatamente com o observado: só o neurônio 10001 dispara de fato (confirmado
por spike count), os outros 3 nunca disparam nem uma vez, com pesos 1x ou
5x. O grupo "desce" é estruturalmente mudo, sempre, independente de qualquer
peso, porque as duas DNa02 não recebem nenhuma sinapse de `object` (nem de
`motion`, achado 2) nesse subgrafo. Nenhuma mudança de peso nas sinapses
plásticas pode fazer o grupo "desce" responder, já que ele não está
conectado ao resto do circuito selecionado.

Isso também faz sentido biologicamente: Giant Fiber é o alvo clássico do
circuito de escape por loom (LC4/LPLC2 → GF, von Reyn et al. 2014), então é
esperado que receba input de `object`. DNa02 é um neurônio de controle de
direção de caminhada, tipicamente dirigido por outras vias (não
especificamente LC4/LPLC2) que não foram incluídas nesta seleção de tipos
celulares. A anatomia real está correta; o problema é que a divisão
arbitrária "metade/metade por índice" em grupos motores colocou, por
coincidência, um circuito funcional inteiro (GF) de um lado e um circuito
funcionalmente desconectado (nesta seleção) do outro.

Conclusão:
1. Não há evidência de que a plasticidade dopaminérgica implementada produza
   aprendizado, com dados sintéticos ou reais. Isso está confirmado agora não
   só pela ausência de efeito, mas pela causa raiz identificada: o grupo
   motor "desce" é estruturalmente mudo neste subgrafo, então nenhuma
   plasticidade nas sinapses selecionadas poderia mudar o resultado.
2. O rastreamento sensório-motor básico (bola → ação) não funciona de forma
   confiável com dados reais, mesmo depois de fechar o circuito anatômico
   (achado 2) e adicionar posição espacial real aos fotorreceptores (achado
   3). Agora sabemos por que também no lado motor: a divisão "sobe/desce" por
   índice não corresponde a nenhuma divisão funcional real, e só um dos 4
   neurônios descendentes tem qualquer conexão com o resto do circuito
   selecionado.

Isso foi reportado como resultado final da primeira rodada de validação, mas
o próprio texto acima recomendava, como trabalho futuro, agrupar sobe/desce
por conectividade real e selecionar mais neurônios descendentes. A seção
seguinte (Achados 9 e 10) documenta uma segunda rodada que fez exatamente
isso. A tabela e a conclusão acima ficam mantidas como registro histórico do
que foi encontrado naquele momento; não foram os números finais do projeto.

### Achado 9 (redesign do circuito, depois de investigar 3 hipóteses)

Antes de redesenhar, uma investigação em três hipóteses (documentada em
detalhe no histórico da sessão, resumida aqui) descartou explicações mais
simples pra "plasticidade sem efeito": não era cache de peso obsoleto
(confirmado por instrumentação direta do `step()`), não era massa sináptica
fixa de `motion` afogando o sinal plástico (auditoria de `edges.parquet`
mostrou zero arestas `motion→descending`, já que as sinapses plásticas eram
100% da entrada feedforward), e forçar o peso a 5x manualmente sem
plasticidade não mudava nada. A causa raiz, encontrada auditando
conectividade real via `neuprint-python` (Achado 8, detalhado em
[`docs/achado-8-auditoria-dna02.md`](docs/achado-8-auditoria-dna02.md)): das
4 `descending` originais, só as 2 Giant Fiber recebiam qualquer sinapse do
resto do circuito. As 2 DNa02 recebem milhares de sinapses reais, mas
nenhuma de `object`/`motion`; são dominadas por circuito de heading do
central complex (LAL, AOTU, PS0xx, PFL3), fora do escopo deste projeto. A
divisão arbitrária sobe/desce por índice tinha colocado, por coincidência, as
2 GF (que respondem) de um lado e as 2 DNa02 (mudas) do outro.

A hipótese de trabalho testada, que `LC10a` (neurônio de rastreamento de
alvo durante corte, Ribeiro et al. 2018) seria um substituto melhor que
`LC4`/`LPLC2` (loom/fuga), foi parcialmente confirmada e parcialmente
refutada: `LC10a` tem zero sinapses diretas em `DNa02` ou `GF`, então a
hipótese original, literal, é refutada. Mas expandindo a busca pros alvos
reais de `LC10a` (275 neurônios auditados), o candidato descendente mais
forte encontrado foi `DNa10` (801 sinapses via só 2 neurônios, um par
bilateral `DNa10_R`/`DNa10_L`), bem acima de qualquer outro candidato
descendente testado (`DNp11`, `DNa16`, `DNp63`, todos abaixo de 200). `LC10a`,
por sua vez, recebe 2657 sinapses de `Tm4` (já no papel `interneuron`),
confirmando a cadeia completa
`photoreceptor → interneuron(Tm4) → target(LC10a) → descending(DNa10)` com
aresta real em todo elo, não só nas pontas (auditado explicitamente).

Mudanças de código: `DNa02` removido do papel `descending`, `DNa10`
adicionado (regex `descending`: `DNa10.*|GF|GiantFiber.*|DNp01.*`); novo
papel `target` (`LC10a.*`) adicionado; `plastic_mask` generalizado pra
`motion|object|target → descending` (antes hardcoded só `motion|object`); e
o split motor sobe/desce (`sim/network.py`, `_compute_motor_groups`) passou
a usar lateralidade real (sufixo `_R`/`_L` do campo `instance`), juntando
GF+DNa10 do lado direito num grupo e GF+DNa10 do lado esquerdo no outro, em
vez de metade/metade por índice. Isso é uma escolha de engenharia explícita,
não um fato biológico: mapear lateralidade de controle de caminhada em
"sobe"/"desce" de um paddle vertical não é o que esses neurônios codificam
naturalmente.

Depois do redesign, o subgrafo real cresceu pra 44 989 neurônios e 1 157 952
sinapses (`target`: 275). A auditoria de arestas confirmou `target→descending`
= 93 sinapses (peso 801, bate com a auditoria) e `object→descending` = 311
(peso 11224). Verificação direta de spikes: os 4 neurônios `descending`
agora disparam (antes, 3 de 4 nunca disparavam, Achado 6). A diferença de
disparo sobe/desce também passou a ter sinal dependente da posição da bola
(+0.98 com a bola no topo, -1.21 embaixo, contra o mesmo valor pros dois
extremos sem direção no Achado 3).

### Achado 10 (resultado do experimento de aprendizado com o circuito novo)

Rodando o mesmo protocolo de controle (plasticidade `LR=0.02` vs. `LR=0`),
4 seeds, 15 000 frames cada, com dados reais e o circuito redesenhado:

| seed | delta com plasticidade (LR=0.02) | delta controle (LR=0) | peso final (plástico) |
|---|---|---|---|
| 42 | -0.058 | -0.245 | 10.23 (subiu de 8.93) |
| 1  | +0.097 | +0.248 | 7.16 (desceu) |
| 2  | -0.070 | -0.203 | 6.61 (desceu) |
| 3  | +0.030 | +0.200 | 6.03 (desceu) |

Pela primeira vez no projeto, plasticidade ligada e desligada produzem
resultados diferentes (antes eram idênticos bit a bit, com dados sintéticos e
reais, Achados 4-6). O redesign teve efeito real e mensurável.

O padrão não é, porém, o de aprendizado direcionado a um objetivo: em
nenhuma das 4 seeds a plasticidade inverteu o sinal do delta (seeds 42 e 2
pioram nas duas condições; seeds 1 e 3 melhoram nas duas). O que a
plasticidade faz, de forma consistente nas 4 seeds, é reduzir a magnitude do
delta em relação ao controle, tanto quando o controle estava melhorando
quanto quando estava piorando. Isso parece mais um efeito de
atenuação/estabilização do que reforço: se os eventos de "bola perdida" são
mais frequentes que "bola rebatida" na maior parte de uma rodada típica, o
termo de dopamina negativa domina ao longo do tempo, então o peso da via
`object`/`target → descending` tende a cair (3 das 4 seeds terminam com peso
menor que o inicial). Um circuito com ganho mais fraco produz oscilações
menores no readout motor, tanto pra cima quanto pra baixo, o que atenuaria o
delta nas duas direções sem exigir nenhum reforço "inteligente". Essa é uma
hipótese mecanicista plausível pra explicar o padrão observado, mas não foi
confirmada com experimentos adicionais por limite de tempo desta rodada;
fica como trabalho futuro (ex.: comparar histograma de ações tomadas entre
condições).

Teste estatístico simples: com n=4 seeds e as 4 favorecendo a mesma direção
(|delta| menor com plasticidade em 4 de 4), um teste de sinal dá p=0.0625
(unicaudal) ou p=0.125 (bicaudal) sob a hipótese nula de 50/50, sugestivo mas
não estatisticamente significativo com uma amostra tão pequena. Testes com
taxas de aprendizado adicionais (0.005, 0.05) e mais seeds, planejados na
especificação desta rodada, não foram executados por limite de tempo; fica
documentado aqui como não feito, não como negativo.

Conclusão desta rodada: o redesign resolveu o problema estrutural do Achado
6 (circuito agora responde nos dois lados, com direção correta) e produziu o
primeiro efeito mensurável e reprodutível de plasticidade em todo o projeto.
Esse efeito, pelo padrão observado, parece ser uma atenuação de magnitude
ligada à assimetria natural entre eventos de reforço positivo e negativo,
não evidência de que o circuito aprendeu a jogar melhor. Isso não é reportado
como "sucesso": é um resultado real e mensurável, mas provavelmente não é o
tipo de aprendizado que o projeto original visava.

Reproduzir:

```bash
python fetch_connectome.py --token SEU_TOKEN   # ou --synthetic
python main.py --headless --frames 15000
```

### Achado 11 (reward denso: implementação, bug de RNG, e piloto de 6 seeds)

O Achado 10 atribuiu o padrão de atenuação-sem-direção a uma hipótese não
verificada: reforço esparso (só em eventos de contato/perda) dá poucas
oportunidades de atribuição de crédito, um problema clássico de RL. Isso foi
testado adicionando um segundo modo de reforço, denso, que dispara dopamina
todo frame com base em se a ação motora bateu com a direção correta
(`sign(ball_y - centro_do_paddle)`, zona morta de `PADDLE_H*0.25` px perto do
centro). Implementado em `main.py` (`FlyPongRunner.reward_mode`:
`"sparse"`/`"dense"`/`"mix"`, parâmetro `mix_ratio`) sem alterar o mecanismo
de dopamina/plasticidade já existente, só a frequência do disparo.

**Bug de RNG encontrado e corrigido:** a primeira implementação reusava
`self.game.rng` (o RNG da física da bola) pro sorteio de "aplicar reforço
denso neste frame". Isso acoplava os dois streams: mudar o modo de reforço
mudava a sequência de resets da bola mesmo sem nenhuma diferença de peso
sináptico. O sintoma que expôs o bug: no piloto inicial, a condição controle
(`LR=0`, sem plasticidade possível) mostrava melhora em 4 de 4 seeds testadas
com reward denso, estatisticamente implausível pra uma condição sem
aprendizado algum. Corrigido usando um `random.Random` separado
(`self._reward_rng`, seed fixo `0xF17B0A7`) só pro sorteio de reforço,
desacoplado da física do jogo. Depois da correção, a condição controle com
reward denso reproduziu exatamente os números de controle do Achado 10
(mesma seed, mesmo delta, até a última casa decimal), confirmando que o modo
de reforço em si não altera a trajetória do jogo quando não há plasticidade,
como deveria ser.

**Teste de sanidade barato (antes do piloto completo):** uma primeira
tentativa de checar se o grupo motor correto acompanha `ball_y - paddle_y`
acima do ruído usando correlação de Pearson encontrou uma correlação forte
(-0.156, muito acima do ruído de embaralhamento), mas o mesmo teste rodado
sem nenhuma plasticidade (`LR=0`) também deu correlação forte (-0.18). Essa
correlação vem do rastreamento de linha de base que o circuito já tem desde
o Achado 9 (conectividade fixa), não é evidência de nada relacionado ao
reforço. Essa métrica foi descartada como teste de sanidade por ser
confundida pela dinâmica de malha fechada já existente; foi usada em vez
disso a própria métrica de taxa de rebatida (mais barata rodar 1 seed do que
inventar uma métrica proxy nova), que mostrou sinal suficiente (delta
não-nulo, diferente entre plástico e controle) pra justificar completar o
piloto de 6 seeds da Fase 2.

**Piloto completo (Fase 2, 6 seeds, `reward_mode="dense"`, `LR=0.02` vs.
`LR=0`, 15 000 frames cada):**

| seed | delta plástico | delta controle | plástico − controle |
|---|---|---|---|
| 42 | -0.195 | -0.245 | +0.049 |
| 1  | -0.020 | +0.248 | -0.268 |
| 2  | +0.126 | -0.203 | +0.328 |
| 3  | +0.290 | +0.200 | +0.090 |
| 4  | +0.092 | -0.067 | +0.159 |
| 5  | +0.124 | +0.258 | -0.134 |

4 de 6 seeds favorecem o plástico sobre o controle, incluindo uma inversão de
sinal completa na seed 2. Testes estatísticos formais: teste de sinal (4 de
6) p=0.688 (bicaudal); teste t pareado t(5)=0.43, p=0.684. Nenhum dos dois é
estatisticamente significativo: a média da diferença (+0.037) tem desvio
padrão (0.212) grande o bastante pra ser inteiramente consistente com ruído.

### Achado 12 (decisão de não prosseguir pra Fase 3, e interpretação)

O critério definido antes de rodar (se a taxa de rebatida melhorar de forma
consistente e estatisticamente distinguível do controle, prosseguir pro
recozimento) não foi atingido no nível de reward mais favorável à hipótese
de reward esparso (100% denso). A Fase 3 completa (4 níveis de `mix_ratio` ×
mínimo 4 seeds × 2 condições) custaria mais de 2 horas de computação
adicionais em cima de um sinal que já não se distingue de ruído no ponto
mais favorável do espectro. Consultado, o usuário optou por não rodar a Fase
3 e documentar o resultado ambíguo da Fase 2 como está, em vez de comprometer
esse tempo; a decisão fica registrada aqui explicitamente, não pulada em
silêncio.

Interpretação (Fase 4, resultado misto, nenhuma das duas hipóteses
confirmada nem refutada com confiança):

- A hipótese de reward esparso (o circuito aprenderia melhor com reforço
  mais denso) não foi confirmada: mesmo no reward 100% denso, o efeito sobre
  a taxa de rebatida não é estatisticamente distinguível do controle sem
  plasticidade.
- Também não foi refutada com confiança: 4 de 6 seeds favoreceram a condição
  com plasticidade, incluindo uma inversão de sinal completa (seed 2), um
  padrão qualitativamente diferente da atenuação-sem-direção pura do Achado
  10 (reward esparso) e mais compatível com algum sinal de aprendizado real,
  só que fraco demais e/ou com variância grande demais pra confirmar com 6
  seeds.
- A hipótese do Achado 8 (canal sensorial fraco, LC10a é só ~2.7% do input de
  DNa10) continua sendo a explicação mais bem sustentada pra por que o
  aprendizado, se existe, é fraco: mesmo com todas as oportunidades de
  crédito que o reward denso oferece, o circuito não tem sinal direcional
  forte o bastante pra produzir uma melhora clara e reprodutível.
- Conclusão: os dados desta rodada não sustentam nem "reward esparso era o
  problema todo" nem "reward denso não muda nada" de forma limpa. O resultado
  mais defensável é que a densidade do reforço pode ter uma contribuição
  pequena e não estatisticamente demonstrada aqui, mas o gargalo dominante
  continua sendo a capacidade do canal sensorial identificada no Achado 8. Os
  4 de 6 seeds não são reportados como "evidência de aprendizado": os números
  e os testes estatísticos aqui dizem que o resultado não é conclusivo.

Reproduzir o piloto:

```bash
python -c "
import sim.network as netmod; netmod.PLASTIC_LR = 0.02  # ou 0.0 para controle
from game.pong import PongGame
import main
r = main.FlyPongRunner(main.DATA_DIR, reward_mode='dense')
r.game = PongGame(seed=42)  # trocar por 1,2,3,4,5 pras outras seeds
for _ in range(15000): r.step_frame()
print(r.validation_report())
"
```

### Achado 13 (teste decisivo de capacidade de canal: sinal sintético fixo e plástico)

O Achado 8 mediu que `LC10a` é só ~2.7% da entrada de `DNa10`, e o Achado 12
usou isso como explicação mais provável pra por que o reforço denso não
produziu aprendizado claro, mas isso nunca tinha sido testado diretamente, só
inferido de uma porcentagem estática. Esta rodada testa a hipótese de forma
direta.

**Fase 1, teste de teto (sem plasticidade):** implementado
`signal_mode="synthetic_strong"` em `main.py`/`sim/network.py`, que injeta
corrente adicional fixa, proporcional a `sign(ball_y - centro_do_paddle)`,
diretamente nos 2 neurônios `DNa10` (identificados via
`ConnectomeNetwork.dna10_idx`, novo atributo), sem tocar nas Giant Fiber. O
ganho foi calibrado empiricamente (`SYNTHETIC_GAIN=10.0`) checando spike
count antes de seguir: nesse nível, `DNa10` dispara perto do teto do modelo
LIF (~1.3 spikes por janela de 4 substeps) toda vez que o sinal está ativo,
confirmado como dominante antes de gastar tempo no experimento completo.

Com `LR=0` (nenhuma plasticidade possível), comparando `signal_mode="real"`
(circuito original) vs. `"synthetic_strong"`, 4 seeds, taxa de rebatida
**absoluta** final:

| seed | real (fim) | sintético fixo forte (fim) |
|---|---|---|
| 42 | 0.143 | 0.333 |
| 1  | 0.327 | 0.390 |
| 2  | 0.100 | **1.000** |
| 3  | 0.220 | 0.137 |

3 de 4 seeds melhoram, uma delas (seed 2) chega a taxa de rebatida perfeita
(1.0). Média: 0.465 (sintético) contra 0.198 (real), mais que o dobro. Teste
t pareado: t(3)=1.23, p=0.31, não significativo com n=4.

**Correção (investigação pós-hoc, ver
[`docs/achado-13-investigacao-seed2.md`](docs/achado-13-investigacao-seed2.md)
pro detalhe completo):** o 1.000 da seed 2 foi auditado antes de ser aceito
como evidência. Com `signal_mode="real"` na mesma seed, o desempenho é ruim
(11 de 77 tentativas, 14%), então não é uma seed estruturalmente fácil. Mas
rastreando a sequência completa de eventos, os 14 misses aconteceram todos
nos primeiros ~18 eventos, seguidos de 105 rebatidas consecutivas sem nenhum
erro. Comparando a trajetória da bola em duas janelas distantes no tempo
(frames 6000-7500 vs. 10000-11500, já com a velocidade no teto), a
correlação é de -0.9999, com a posição Y presa numa faixa de só ~12% da
tela. O sistema travou numa órbita estreita e repetitiva depois do sucesso
inicial: as 105 "rebatidas" são a mesma situação fácil repetida, não 105
desafios independentes. Recalculando sem a seed 2 (seeds 42, 1, 3):

| | com seed 2 (original) | sem seed 2 |
|---|---|---|
| média real (fim) | 0.198 | 0.230 |
| média sintético forte (fim) | 0.465 | 0.287 |
| diferença | +0.267 | +0.057 |
| teste t pareado | t(3)=1.23, p=0.31 | t(2)=0.72, p=0.55 |

Sem a seed 2, o efeito é muito mais fraco (+0.057, não +0.267) e continua
não-significativo. 2 de 3 seeds restantes ainda melhoram com sinal
sintético, mas nada perto de "quase perfeito". Conclusão revisada da Fase 1:
há um sinal de melhora fraco e não-significativo quando o canal recebe um
sinal forte fixo, o suficiente pra justificar testar a Fase 2, mas não a
alegação forte de "o resto do pipeline é capaz de desempenho quase perfeito"
que constava aqui antes desta correção. O número original (0.465/0.198) é
mantido acima como registro histórico do que foi reportado antes da
auditoria, não apagado.

**Fase 2, canal sintético plástico:** o mesmo canal, agora com peso inicial
`SYNTHETIC_W_INIT=8.93` (mesma ordem de grandeza do peso médio das sinapses
plásticas reais nos Achados 10-12), sujeito à mesma regra de três fatores
(mesmo `PLASTIC_LR`, mesmo teto `SYNTHETIC_DELTA_MAX=3.0` do
`PLASTIC_DELTA_MAX` real). É uma implementação simplificada a nível de frame,
não substep, documentada em `main.py`, usando o `dopamine_level` já calculado
pela rede real. 6 seeds, `LR=0.02` (plástico) vs. `LR=0` (controle, peso
travado no valor inicial forte), 15 000 frames:

| seed | delta plástico | delta controle | plástico − controle | peso final (plástico) |
|---|---|---|---|---|
| 42 | -0.192 | -0.177 | -0.015 | 8.28 |
| 1  | +0.004 | +0.051 | -0.047 | 8.10 |
| 2  | -0.161 | +0.293 | -0.454 | 8.07 |
| 3  | +0.124 | -0.010 | +0.134 | 8.15 |
| 4  | -0.260 | -0.029 | -0.231 | 8.18 |
| 5  | -0.004 | +0.065 | -0.069 | 7.81 |

5 de 6 seeds favorecem o controle sobre o plástico, o oposto do que a
hipótese de capacidade de canal previa. Teste de sinal: 1 de 6, p=0.219.
Teste t pareado: t(5)=-1.37, p=0.229. Nenhum dos dois é significativo, mas a
direção é consistente e mecanisticamente explicável: o peso sintético decai
de 8.93 pra 7.8-8.3 em todas as 6 seeds plásticas (mesmo padrão de
decaimento por excesso de eventos de reforço negativo identificado nos
Achados 10 e 12, já que "bola perdida" é mais comum que "bola rebatida" na
maior parte de uma rodada típica), enquanto o controle mantém o peso travado
no valor inicial forte, que a Fase 1 já mostrou ser suficiente pra bom
desempenho. A plasticidade está ativamente enfraquecendo um canal que já
começou forte o bastante pra funcionar bem: o próprio mecanismo de reforço é
o problema, não a falta de sinal.

**Interpretação (Fase 3):** o resultado bate com o terceiro cenário previsto
na especificação desta rodada. A Fase 1 funcionou (capacidade de canal não é
o teto), mas a Fase 2 não mostra aprendizado claro, e de fato mostra uma
tendência não-significativa mas consistente na direção oposta. Isso refina a
conclusão do Achado 12 em vez de simplesmente confirmá-la: não é que o canal
seja fraco demais pra sustentar aprendizado (a Fase 1 mostra que um canal
forte funciona bem), é que a regra de plasticidade de três fatores, tal como
implementada, tem um viés estrutural de enfraquecer sinapses ao longo de uma
rodada típica (por causa da assimetria bola-perdida vs. bola-rebatida), o
que é contraproducente quando a sinapse já começa forte. Isso aponta pra uma
terceira hipótese em aberto, não resolvida nesta rodada: o mecanismo de
atribuição de crédito da regra de três fatores precisaria ser reformulado
(ex. normalizar pela frequência de eventos positivos/negativos, ou usar uma
linha de base de reforço não-zero) pra não decair sistematicamente um canal
já funcional.

Reproduzir:

```bash
python -c "
import sim.network as netmod; netmod.PLASTIC_LR = 0.0  # ou 0.02 para plastico
from game.pong import PongGame
import main
r = main.FlyPongRunner(main.DATA_DIR, signal_mode='synthetic_plastic')  # ou 'synthetic_strong'/'real'
r.game = PongGame(seed=42)
for _ in range(15000): r.step_frame()
print(r.validation_report())
"
```

### Achado 14 (correção do viés de plasticidade: duas tentativas, nenhuma funcionou)

O Achado 13 (Fase 2) isolou a causa exata do decaimento sistemático: eventos
de reforço negativo são mais frequentes que positivos numa rodada típica, e
a regra de três fatores não compensa isso. Diferente dos achados anteriores,
este tinha um alvo de correção concreto, não uma limitação estrutural sem
solução óbvia. Duas correções foram implementadas em `sim/network.py`, como
modos selecionáveis via `PLASTICITY_MODE` (`"original"` continua sendo o
padrão e reproduz exatamente os números dos Achados 10-13, verificado
recomputando a seed 42 do controle e conferindo bit a bit contra o valor já
publicado):

- `"freq_normalized"`: cada atualização de peso é escalada pelo inverso da
  frequência recente do tipo de evento que a gerou, via médias móveis
  exponenciais (`FREQ_EMA_DECAY=0.999`) das taxas de disparo positiva e
  negativa, capada em `FREQ_NORM_MAX_SCALE=10.0`.
- `"tonic_baseline"`: um viés tônico positivo constante (`TONIC_BIAS=0.05`)
  é somado ao `dopamine_level` antes de cada atualização, pra que a ausência
  de eventos não puxe o peso pra baixo por padrão.

Testado no mesmo testbed isolado do Achado 13 Fase 2 (canal sintético,
`SYNTHETIC_W_INIT=8.93`, `LR=0.02`, 6 seeds, 15 000 frames), contra o mesmo
controle já documentado (peso travado em 8.93, `LR=0`, reaproveitado do
Achado 13 sem re-rodar, já que por construção não depende do modo de
plasticidade):

| seed | delta controle | delta `freq_normalized` | peso final | delta `tonic_baseline` | peso final |
|---|---|---|---|---|---|
| 42 | -0.177 | -0.323 | 7.33 | -0.178 | 7.45 |
| 1  | +0.051 | -0.159 | 7.56 | +0.309 | 9.30 |
| 2  | +0.293 | -0.203 | 7.10 | -0.203 | 6.98 |
| 3  | -0.010 | -0.099 | 7.31 | +0.596 | 8.52 |
| 4  | -0.029 | -0.273 | 7.20 | -0.241 | 7.40 |
| 5  | +0.065 | -0.164 | 7.46 | -0.112 | 7.16 |

`freq_normalized` falha de forma decisiva: 0 de 6 seeds ficam melhores que o
controle, o peso decai mais que a regra original (média 7.33 contra 8.10 do
Achado 13), e a diferença é estatisticamente significativa na direção
errada (teste t pareado t(5)=-4.13, p=0.009; teste de sinal p=0.031).

`tonic_baseline` não falha tão claramente, mas também não funciona: 2 de 6
seeds melhoram bastante (seed 1 e seed 3, essa última com o peso final
ficando bem mais perto do valor inicial e a taxa de rebatida subindo pra
0.869), 3 pioram, 1 fica igual. A diferença média contra o controle é
essencialmente zero (t(5)=-0.02, p=0.98; teste de sinal 2 de 6, p=0.69) e o
peso continua decaindo na maioria das seeds, só que com desvio padrão bem
maior (0.39 contra 0.14 do `freq_normalized`).

**Nenhuma das duas correções passa no critério mínimo** definido antes de
rodar (o peso final não deveria mais decair sistematicamente abaixo do
valor inicial). Por isso a Fase 3 (aplicar a correção no circuito real) não
foi executada — decisão prevista desde a especificação desta rodada pro
caso de as duas opções falharem no teste isolado, mais barato. O canal
sintético, que a Fase 1 do Achado 13 já tinha mostrado ser forte o bastante
pra sustentar bom desempenho com peso fixo, continua perdendo desempenho
quando a plasticidade é ligada, com as duas correções tentadas aqui. O
mecanismo de atribuição de crédito da regra de três fatores continua sendo
um problema em aberto: as duas tentativas mais óbvias de correção não
resolveram, e uma delas piorou a situação de forma mensurável.

A Fase 4 original (comparação com um baseline trivial sem rede neural
nenhuma, `game/baseline_policy.py`: `action = sign(ball_y - paddle_y)`,
mesma métrica) foi rodada de qualquer forma, já que é barata e não depende
do resultado da Fase 3. Nas 6 seeds testadas, o baseline trivial atinge
taxa de rebatida 1.0 do início ao fim, sempre: por ter latência zero e
acesso perfeito à posição da bola (sem ruído de disparo, sem os 4 substeps
de atraso do LIF), ele nunca erra. Isso não foi reportado como derrota do
circuito do conectoma naquele momento (a comparação relevante seria o quão
perto o circuito chega desse teto, não se ele vence) — mas a Verificação 2
abaixo aprofunda exatamente esse ponto.

#### Três verificações pendentes

**Verificação 1, por que `freq_normalized` piorou tanto:** a hipótese
inicial era que a média móvel de frequência (`pos_rate_ema`/`neg_rate_ema`)
cai perto de zero quando um tipo de evento fica raro, inflando o fator
`scale` até perto do teto (`FREQ_NORM_MAX_SCALE=10.0`) e amplificando
ruído. Testado instrumentando `scale` a cada passo (`plasticity_scale_history`,
novo em `sim/network.py`) numa rodada de 15 000 frames: **hipótese
refutada**. O valor de `scale` nunca passa de 2.17 em 59 170 amostras
(mediana 1.16, média 1.15), nunca chega perto do teto de 10.0. Mais
especificamente, `scale` até se comporta na direção pretendida quando
separado por sinal da dopamina: média 1.17 quando positiva (amplifica
reforço de acerto) e 0.76 quando negativa (atenua reforço de erro), o
oposto de "amplificar ruído indiscriminadamente". Mesmo assim o peso final
decai mais que a regra original. A explicação mais provável, não
aprofundada por decisão de escopo (a instrução original era não gastar mais
tempo ajustando o teto se essa hipótese caísse), é que a elegibilidade
(`pre_trace × post_spike`) tende a ser maior durante os eventos de dopamina
negativa do que durante os positivos, então mesmo com menos ocorrências e
um fator `scale` menor por ocorrência, a contribuição agregada dos eventos
negativos ainda domina. Não foi testado ajustar `FREQ_NORM_MAX_SCALE` pra
valores mais conservadores (2.0, 3.0), porque o teto nunca era o fator
limitante pra começo de conversa.

**Verificação 2, um baseline justo (`game/baseline_policy.py`,
`run_delayed_linear_baseline`):** implementado sujeitando o baseline
trivial às mesmas duas restrições do circuito real — atraso de 1 frame
(aproximando quantos frames um estímulo leva pra atravessar a cadeia
sináptica completa dado o orçamento de 4 substeps por frame) e leitura só
através do mesmo estímulo populacional que os fotorreceptores recebem
(`ball_to_photoreceptor_stimulus`, resolução de 1783 posições, igual ao
subgrafo real), em vez de `ball_y` exato. **Resultado: nas 6 seeds
testadas, o baseline com essas duas restrições continua acertando 1.0 do
início ao fim, idêntico ao baseline sem restrição nenhuma.** As duas
limitações pedidas, embora reais, não bastam pra equalizar a comparação:
`PADDLE_SPEED=4` px/frame já é mais rápido que a componente vertical típica
da velocidade da bola, e 1783 posições de retina são muito mais resolução
do que o necessário pra distinguir a direção certa em qualquer situação não
patológica. Um controlador `sign()` linear fica com a direção certa mesmo
com essas duas limitações, porque nenhuma delas ataca o gargalo real do
circuito neural: o readout de só 2 neurônios por grupo, com contagem de
spike ruidosa e binária (Achados 6, 9, 13), não a resolução sensorial ou o
atraso temporal. **Este é o achado mais importante das três verificações**:
ele mostra que "atraso" e "resolução sensorial" não são o que torna o
circuito do conectoma mais difícil de acertar que uma regra de 1 linha — o
gargalo está inteiramente do lado motor, algo que nenhum baseline linear
simples consegue replicar sem também ficar artificialmente ruidoso do
mesmo jeito. A comparação "circuito real vs. baseline justo" continua sem
uma resposta genuinamente informativa: um baseline que reproduzisse
fielmente esse gargalo de leitura deixaria de ser uma "regra de 1 linha" e
passaria a ser, na prática, uma reimplementação do problema.

**Verificação 3, auditoria da seed 3 do `tonic_baseline`:** rodada com o
mesmo método do Achado 13 (trajetória completa, sequência de eventos,
correlação entre janelas distantes no tempo). **Resultado diferente da
seed 2 do Achado 13**: a correlação entre janelas distantes
([6000,7500) vs. [10000,11500)) é de apenas -0.11 (contra -0.9999 da seed 2
do Achado 13), e a posição da bola varia por quase toda a altura da tela
(0 a 312 de 320px), não presa numa faixa estreita. Não é uma órbita
travada. Mas também não é aprendizado limpo e sustentado: dos 84 eventos
totais (35 bounces, 49 misses), há uma sequência real de 22 rebatidas
consecutivas no meio da rodada (índices 56-77), seguida de 6 misses
consecutivos bem no final (índices 78-83) — a rodada termina em queda, não
em platô. A taxa de rebatida "fim" (0.869) reflete a janela móvel de 20
eventos anteriores, que ainda captura a maior parte da sequência boa de
22 antes dos misses finais, o que é matematicamente consistente, não um
erro de cálculo. Rodando a mesma seed 3 com a regra original (controle,
`LR=0`): delta -0.01, sem evidência de aprendizado, confirmando que a seed
não era estruturalmente fácil por si só. **Conclusão: o resultado da seed 3
é real, não um artefato de órbita travada, mas também não é evidência de
aprendizado estável** — é uma janela de bom desempenho que não se sustentou
até o fim da rodada. Isso não muda a conclusão agregada do `tonic_baseline`
(ainda não-significativo, p=0.98), só esclarece que a alta variância vem de
episódios reais de desempenho instável, não de um bug de contagem como na
seed 2 do Achado 13.

Reproduzir:

```bash
python -c "
import sim.network as netmod
netmod.PLASTIC_LR = 0.02
netmod.PLASTICITY_MODE = 'tonic_baseline'  # ou 'freq_normalized'
from game.pong import PongGame
import main
r = main.FlyPongRunner(main.DATA_DIR, signal_mode='synthetic_plastic')
r.game = PongGame(seed=42)
for _ in range(15000): r.step_frame()
print(r.validation_report())
"
python -m game.baseline_policy
```

### Achado 15 (o peso-teto real é ~4,5x maior que o usado em todos os experimentos de plasticidade)

A Verificação 2 do Achado 14 mostrou que nenhum baseline "justo" (com atraso
e resolução sensorial equivalentes) consegue reproduzir o gargalo real do
circuito, que está no readout de poucos neurônios, não na informação
sensorial. Essa rodada respondeu uma pergunta diferente, sem precisar de
baseline externo: dentro da própria arquitetura de sinapse com peso (o mesmo
mecanismo `W.dot(spikes)` que a plasticidade usa, não a injeção de corrente
direta do Achado 13 Fase 1), qual é o melhor desempenho possível com peso
fixo, e a regra de aprendizado consegue chegar perto disso?

**Fase 1 (varredura grosseira, 2 seeds, `LR=0`):**

| peso | média (2 seeds) | valores |
|---|---|---|
| 2 | 0.247 | 0.306, 0.188 |
| 4 | 0.284 | 0.265, 0.302 |
| 6 | 0.279 | 0.288, 0.270 |
| 8 | 0.349 | 0.388, 0.310 |
| 8.93 | 0.314 | 0.293, 0.334 |
| 10 | 0.389 | 0.425, 0.353 |
| 12 | 0.475 | 0.270, 0.679 |
| 15 | 0.617 | 0.352, 0.883 |
| 20 | 0.666 | 0.401, 0.930 |
| 30 | 0.757 | 0.578, 0.937 |
| 35 | 0.769 | 0.537, 1.000 |
| 40 | 0.982 | 0.964, 1.000 |
| 50 | 1.000 | 1.000, 1.000 |

A média sobe de forma monotônica com o peso, sem sinal de platô até perto
de 30-35, e satura perto de 1.0 (o mesmo teto do baseline trivial oráculo)
só a partir de 40. A grade original do plano (até 30) não teria capturado
isso; dois pesos extras (40 e 50) precisaram ser testados pra confirmar a
saturação.

**Fase 2 (refinamento, 6 seeds completas, pesos 35 e 40):**

| peso | média (6 seeds) | desvio padrão | valores |
|---|---|---|---|
| 35 | 0.883 | 0.178 | 0.537, 1.000, 0.863, 1.000, 0.984, 0.913 |
| 40 | 0.994 | 0.015 | 0.964, 1.000, 1.000, 1.000, 1.000, 1.000 |

Peso 35 satura em 4 das 6 seeds mas fica bem abaixo nas outras duas
(0.537 e 0.863), variância alta. Peso 40 é uniformemente quase perfeito
nas 6 seeds (desvio padrão de só 0.015). **`SYNTHETIC_W_INIT=40` é o
peso-teto prático** — o maior valor testado onde o desempenho já não
melhora de forma relevante (50 dá exatamente o mesmo resultado que 40, sem
diferença).

**Comparação central: 40 contra os 8.93 usados em todos os experimentos de
plasticidade dos Achados 13 e 14 — uma razão de 4,48x.** Isso muda a leitura
retroativa desses achados: o "controle de peso fixo" usado lá (8.93,
média 0.314 na varredura desta rodada) nunca esteve perto do melhor
desempenho possível dessa arquitetura — não invalida a conclusão sobre a
regra de plasticidade em si (o decaimento sistemático e a falha das duas
correções continuam sendo resultados válidos, medidos corretamente contra
o controle que foi de fato usado), mas significa que "controle sem
plasticidade" nos Achados 13-14 não era o melhor controle possível. Mais
crítico ainda: com `PLASTIC_DELTA_MAX=3.0`, a plasticidade partindo de 8.93
só alcança o intervalo `[5.93, 11.93]` — mesmo no melhor caso hipotético
(toda atualização de peso na direção certa, sem nenhum decaimento), a regra
de plasticidade como configurada nos Achados 13-14 **nunca poderia chegar
perto do peso-teto de 40**, o teto de delta é pequeno demais pra isso,
independente de qualquer correção de viés.

**Fase 3 (plasticidade partindo do peso-teto) não foi executada.** A Fase 1
e a Fase 2 já demandaram mais compute do que o planejado (a varredura
precisou ser estendida de 30 até 50 pra encontrar a saturação, e um
processo de varredura em loop único foi morto duas vezes por falta de
memória do sistema, exigindo trocar pra um processo Python separado por
combinação peso/seed — mais lento, mas resistente a isso). A Fase 3 (18
execuções: 3 modos de plasticidade × 6 seeds) foi cancelada pelo usuário
antes de qualquer resultado ser produzido, por restrição de tempo. Nenhum
dado da Fase 3 existe; não é um resultado negativo, é trabalho não feito,
registrado como tal. A pergunta que a Fase 3 responderia — se a plasticidade
consegue se manter perto do peso-teto partindo dele, ou se decai pra longe
mesmo do ótimo — continua em aberto.

**Interpretação honesta:** o achado mais forte e já confirmado desta rodada
não depende da Fase 3: o teto de desempenho da arquitetura de readout (com
peso fixo, mecanismo de sinapse normal) é muito mais alto do que qualquer
experimento de plasticidade deste projeto chegou perto de testar. Isso não
significa que a plasticidade funcionaria se partisse do peso-teto — essa é
exatamente a pergunta que ficou sem resposta —, mas significa que o teto de
delta usado (`PLASTIC_DELTA_MAX=3.0`) é pequeno demais pra alcançar esse
ponto vindo de 8.93 de qualquer jeito, então os Achados 13-14 nunca tiveram
a chance de testar aprendizado perto do ótimo, só longe dele. Ver
Limitações abaixo pra como isso reformula a leitura desses achados sem
invalidar as conclusões já publicadas lá.

Reproduzir a Fase 1/2 (uma combinação peso/seed por vez, em processo
separado — evita o problema de memória visto ao rodar tudo num loop só):

```bash
python -c "
import sim.network as netmod
netmod.PLASTIC_LR = 0.0
from game.pong import PongGame
import main
main.SYNTHETIC_W_INIT = 40  # trocar pelo peso a testar
r = main.FlyPongRunner(main.DATA_DIR, signal_mode='synthetic_plastic')
r.game = PongGame(seed=42)
for _ in range(15000): r.step_frame()
import numpy as np
print(r.validation_report(), 'media geral:', np.mean(r.bounce_rate_curve))
"
```

### Achado 17 (plasticidade numa região de inclinação real da curva peso-desempenho)

O Achado 15 mapeou a curva completa de desempenho por peso fixo e mostrou
que a região onde os Achados 13-14 testaram plasticidade (`8.93 ± 3`) é
relativamente achatada. Entre os pesos ~10 e ~20 a inclinação é bem maior —
mudanças de peso ali produzem mudanças de desempenho mais visíveis. Esta
rodada testa plasticidade partindo dessa região, não da região achatada.

**Fase 1 (escolha do ponto de partida, modo `original`, 3 seeds, pesos 12 e
15):**

| peso | seed | delta plástico | peso final | delta controle |
|---|---|---|---|---|
| 12 | 42 | -0.234 | 10.90 | -0.227 |
| 12 | 1 | -0.044 | 11.13 | +0.670 |
| 12 | 2 | -0.217 | 11.34 | +0.596 |
| 15 | 42 | -0.163 | 14.47 | -0.211 |
| 15 | 1 | -0.234 | 14.02 | +0.580 |
| 15 | 2 | -0.120 | **17.41** | +0.513 |

Amplitude de movimento do peso (máximo menos mínimo do peso final entre as
3 seeds): 0.44 no peso 12, contra 3.39 no peso 15 — incluindo o primeiro
caso em todo o projeto de o peso **subir** de forma clara sob plasticidade
(seed 2, +2.41, quase no teto do delta de ±3.0). Peso 15 escolhido pra Fase
2 por esse critério (mais sinal de que a plasticidade sai da região
achatada), não pela direção do efeito, que ainda não estava definida nesta
fase.

**Fase 2 (protocolo completo, peso 15, 6 seeds, 3 modos):**

| seed | controle | `original` | `freq_normalized` | `tonic_baseline` |
|---|---|---|---|---|
| 42 | -0.211 | -0.163 | -0.333 | -0.300 |
| 1  | +0.580 | -0.234 | -0.381 | -0.075 |
| 2  | +0.513 | -0.120 | -0.347 | -0.203 |
| 3  | +0.140 | -0.113 | -0.103 | -0.159 |
| 4  | -0.160 | -0.035 | -0.257 | -0.055 |
| 5  | -0.097 | -0.093 | -0.243 | -0.092 |

Testes estatísticos (plástico vs. controle, mesmo formato dos Achados
13-14):

| modo | teste t pareado | teste de sinal (n favorável/6) |
|---|---|---|
| `original` | t(5)=-1.60, p=0.171 | 3/6, p=1.00 |
| `freq_normalized` | t(5)=-2.50, **p=0.054** | 0/6, p=0.031 |
| `tonic_baseline` | t(5)=-1.95, p=0.109 | 2/6, p=0.688 |

`freq_normalized` chega perto da significância (p=0.054) na mesma direção
ruim do Achado 14 (p=0.009 lá) — consistente entre as duas regiões
testadas, reforçando que essa correção específica não funciona
independente de onde o peso começa. Os outros dois modos não são
significativos, mas nenhum é favorável à plasticidade.

**Achado qualitativo notável**: nas seeds 1 e 2, o controle teve um
resultado bom por sorte da trajetória determinística da bola (+0.580 e
+0.513, sem nenhuma plasticidade envolvida). Nos três modos de
plasticidade, essas duas seeds especificamente pioram bastante em relação
ao controle (diferença de -0.63 a -0.96, as piores diferenças de toda a
tabela). Isso sugere que, nesta região de peso, o efeito mais visível da
plasticidade não é "aprender mal" de forma genérica — é perturbar uma
trajetória que por acaso já estava indo bem, com o mesmo custo em qualquer
um dos três modos testados.

**Correção pós-hoc (auditoria das seeds 1 e 2, mesmo método do Achado 13):**
antes de aceitar a leitura acima, as duas seeds "sortudas" foram auditadas
pelo mesmo processo que expôs a órbita travada da seed 2 do Achado 13
(`docs/achado-13-investigacao-seed2.md`): correlação entre duas janelas
distantes no tempo, ambas dentro do período estável (depois do último
evento de miss), e faixa de variação de `ball_y` nesse período.

| seed | correlação entre janelas distantes | faixa de `ball_y` | veredito |
|---|---|---|---|
| 1 | -0.714 | 97.5% da tela | trajetória genuína |
| 2 | **-0.99999999...** | 47.1% da tela | **órbita travada** |

A seed 2 reproduz a mesma assinatura da seed 2 do Achado 13 (correlação
praticamente -1 entre janelas bem separadas), embora a faixa presa desta
vez seja mais larga (47% contra 12% no Achado 13) — ainda um ciclo
periódico repetitivo, não desafios variados. A seed 1 não reproduz esse
padrão: correlação moderada (não a assinatura de -1 quase exata) e
variação por quase toda a altura da tela, consistente com rastreamento
genuíno, não um ciclo travado.

Isso corrige parcialmente o "achado qualitativo notável" acima: a
interpretação de "plasticidade perturba trajetória sortuda" continua válida
pra seed 1 (trajetória genuína, plasticidade genuinamente piora algo real).
Pra seed 2, a leitura muda — o controle não estava tendo um bom desempenho
genuíno, estava preso num ciclo degenerado (mesmo problema do Achado 13,
seed diferente do redesign, mesma seed de jogo). A plasticidade "destruindo"
esse resultado não é necessariamente ruim: pode estar só saindo de um
estado trivial, não perdendo algo de valor. As conclusões estatísticas da
Fase 2 (nenhum modo significativamente melhor que o controle) não mudam,
já que eram baseadas na tabela completa de 6 seeds, não só nessas duas —
mas a leitura qualitativa "a plasticidade estraga bom desempenho" fica
sustentada só pela metade dos casos que pareciam apoiá-la.

**Fechando o elo do Achado 13 seed a seed**: repetindo as 18 rodadas
plásticas (determinístico, mesmos parâmetros, só pra capturar contagem de
bounce/miss) e correlacionando a fração de eventos positivos
(`bounce/(bounce+miss)`) de cada seed com a diferença pareada
(plástico−controle) daquela seed:

| modo | correlação (Pearson) | p |
|---|---|---|
| `original` | r=-0.45 | 0.37 |
| `freq_normalized` | r=-0.15 | 0.78 |
| `tonic_baseline` | r=+0.86 | **0.028** |
| todos os 18 pontos juntos | r=-0.016 | 0.95 |

A hipótese (mais eventos negativos que positivos → pior desempenho da
plasticidade) **não fecha de forma robusta**: só `tonic_baseline` mostra a
correlação na direção esperada e com significância (mas com n=6, uma
amostra pequena pra confiar sem mais dados); os outros dois modos não
mostram correlação nenhuma, e o conjunto completo de 18 pontos é
estatisticamente nulo. A explicação agregada do Achado 13 (mais reforço
negativo que positivo numa rodada típica) continua sendo a leitura mais
sustentada no nível agregado, mas não se traduz num efeito consistente
seed a seed nesta região de peso — outro fator (como o achado qualitativo
da "trajetória sortuda perturbada" acima) parece pesar tanto ou mais que a
proporção bruta de eventos.

**Fase 3 (observação visual) não foi completada com descrição
qualitativa.** A CLI do `main.py` foi atualizada com os parâmetros
necessários (`--signal-mode`, `--synthetic-w-init`, `--plasticity-mode`,
`--plastic-lr`), e o comando abaixo roda a configuração desta rodada numa
janela visual — mas não há ferramenta de captura de tela pra janelas
nativas neste ambiente (só pra navegador), então não foi possível observar
e descrever o comportamento do paddle de forma honesta sem inventar o que
"parece" ter sido visto. Fica como comando pronto pra quem quiser observar
diretamente:

```bash
python main.py --signal-mode synthetic_plastic --synthetic-w-init 15 \
    --plasticity-mode original --plastic-lr 0.02
```

**Comparação com o Achado 14 (peso 8.93, região achatada)**: o padrão
geral se repete — nenhum dos três modos melhora a plasticidade sobre o
controle, e `freq_normalized` é consistentemente o pior (significativo ou
quase-significativo nas duas regiões testadas). A diferença nesta região de
maior inclinação é que os efeitos são numericamente maiores e mais
consistentes na direção negativa (todos os 18 valores de diferença nesta
rodada são negativos ou próximos de zero, contra uma mistura mais
equilibrada de sinais no Achado 14), e surge o achado qualitativo novo da
"trajetória sortuda perturbada" (confirmado pra metade dos casos que
pareciam apoiá-lo depois da auditoria acima; a outra metade era órbita
travada, não desempenho genuíno). Isso fortalece a conclusão do Achado
13/14: o problema é a regra de plasticidade em si, não a região de peso
onde ela foi testada.

### Achado 18 (peso-teto de 40: plasticidade não supera o controle perfeito)

O Achado 15 mostrou que `SYNTHETIC_W_INIT=40` é o teto de desempenho do canal
sintético fixo. A pergunta seguinte era se as três regras de plasticidade já
testadas conseguem ao menos preservar esse desempenho quando começam no teto.
O protocolo completo usa 15.000 frames, seeds 42/1/2/3/4/5, sinal
`synthetic_plastic`, peso inicial 40, `PLASTIC_LR=0.02`, quatro substeps e
sem poda causal. O controle usa o mesmo canal com `PLASTIC_LR=0`.

| modo | média taxa final | diferença final média vs. controle | plástico melhor/igual/pior |
|---|---:|---:|---:|
| controle | 1,000 | 0,000 | — |
| `original` | 0,821 | -0,179 | 0 / 3 / 3 |
| `freq_normalized` | 0,706 | -0,294 | 0 / 3 / 3 |
| `tonic_baseline` | 0,553 | -0,447 | 0 / 2 / 4 |

Nenhuma regra superou o controle em nenhuma seed. O teste t pareado
exploratório sobre a taxa final dá p=0,101 (`original`), p=0,095
(`freq_normalized`) e p=0,028 (`tonic_baseline`, pior). Isso não deve ser
tratado como um novo teste confirmatório: a taxa final foi escolhida depois de
ver os dados e o controle está saturado em 1,0. A conclusão robusta é mais
modesta: **nesta configuração, não há sinal de que qualquer uma das três
regras preserve ou melhore o desempenho de um canal que já funciona no teto**.

Há ainda um detalhe mecanístico contrário à hipótese simples "mais peso ajuda":
no modo `original`, todas as seis seeds terminaram com peso acima de 40
(40,54–43,00), mas três tiveram queda material de desempenho. A próxima
rodada não deve apenas variar taxa de aprendizado; deve pré-definir métrica,
separar seeds de desenvolvimento/teste e testar se o gargalo é crédito
temporal, observação ou readout. A receita reprodutível e retomável está em
`scripts/run_achado18.py`; a primeira rodada de 23 células foi executada
interativamente, e a célula final foi reproduzida pelo script.

**Classificação contra a tabela estática do Achado 15** (35→0,883; 40→0,994;
50→1,0): as 18 seeds plásticas terminaram com peso entre 39,21 e 43,00 —
**todas dentro ou acima da região saturada** (≥38), nenhuma chegou perto da
região de queda real (~35). Isso significa que este desenho experimental não
consegue distinguir "plasticidade parou de decair o peso" de "nunca saiu do
platô" — não é nenhum dos três cenários prometidos de forma limpa: não é (a)
puramente inconclusivo por desenho (a variação de desempenho absoluto,
0,20–1,0, é grande demais pra só refletir indiferença de platô), não é (b)
queda confirmada fora do platô (o peso nunca saiu do platô), e não é (c)
resultado positivo (nenhum modo bate o controle com significância). O achado
real é o detalhe mecanístico acima: **desempenho absoluto e peso final estão
desacoplados mesmo dentro da zona nominalmente segura** — o mecanismo mais
provável é instabilidade dinâmica durante a trajetória (o peso passa por
valores diferentes ao longo da rodada, não fica parado em 40), não a posição
final isolada.

**Auditoria pós-hoc da seed 2 (mesmo método dos Achados 13/17, órbita
travada):** a seed 2 teve os melhores números em todos os três modos
(`bounce_rate_fim`=1,0 nos três). Por padrão — já travou nos Achados 13 e 17
— foi auditada antes de aceitar isso como desempenho genuíno.

| modo | correlação entre janelas distantes | faixa de `ball_y` | veredito |
|---|---|---|---|
| `original` | **1,000000** | 23,8% da tela | **órbita travada** |
| `freq_normalized` | -0,595 | 16,5% da tela | suspeito (faixa estreita, correlação não tão extrema) |
| `tonic_baseline` | **1,000000** | 23,8% da tela | **órbita travada** |

Terceira vez que essa seed específica trava (Achados 13, 17 e agora 18),
sempre com o mesmo tipo de assinatura (correlação próxima de ±1, faixa de
`ball_y` bem abaixo dos ~97% de uma trajetória genuína — comparar com a seed
4/`freq_normalized`, pior resultado da rodada, que tem correlação -0,09 e
faixa 97,5%, ou seja, desempenho ruim genuíno, não órbita). Recalculando as
estatísticas sem a seed 2: nenhuma conclusão muda de direção (`original`
p=0,442, `freq_normalized` p=0,123, `tonic_baseline` p=0,350 — todas ainda
não-significativas), mas a média de `bounce_rate_fim` cai em todos os três
modos (0,821→0,785; 0,706→0,647; 0,553→0,463), reforçando que os números
"bons" da rodada completa eram parcialmente inflados por um artefato
determinístico, não por três seeds realmente jogando bem.

**Fase 3 (observação visual):** `python main.py --signal-mode
synthetic_plastic --synthetic-w-init 40 --plasticity-mode original
--plastic-lr 0.02` roda sem erro, mas — mesma limitação já registrada no
Achado 17 — não há ferramenta de captura de tela pra janelas nativas neste
ambiente, só pra navegador. Não é a primeira vez que essa limitação aparece,
então não deveria ser tratada como surpresa; fica só o comando pronto pra
quem quiser observar diretamente.

**Nota de reprodutibilidade:** durante esta rodada de auditoria, uma edição
externa em `main.py` (otimização de performance, não relacionada à ciência
do achado) introduziu um bug silencioso — uma checagem `hasattr` que nunca
batia por causa de *name mangling* do Python em atributos `__duplo_underscore`,
zerando o canal de reforço sintético sem gerar erro nem aviso. Foi
encontrado e corrigido durante a auditoria da Fase 2 (comparando reprodução
isolada de uma célula contra o dado já coletado) — os 24 resultados acima
foram verificados como consistentes antes e depois do bug (o bug foi
introduzido depois da coleta original, não durante). Fica como lembrete
prático: uma otimização "matematicamente equivalente" ainda pode quebrar o
experimento por um erro de implementação comum, e a defesa que funcionou foi
a mesma de sempre neste projeto — reproduzir um resultado conhecido antes de
confiar num código novo, não revisar o diff visualmente e assumir que está
certo.

### Achado 19 (aprendizado real: a rede reaprende a jogar com o readout invertido)

**Resumo:** pela primeira vez neste projeto, uma regra de plasticidade supera
com clareza um controle sem aprendizado, e sob o teste mais duro disponível.
Com os grupos "sobe"/"desce" do readout trocados, o circuito inato joga para
o lado errado (6,7% de acerto). Depois de 3.000 saques de treino com
recompensa, a mesma rede acerta **57,2%** (6 seeds de teste × 300 saques),
contra 13,1% do controle com a mesma homeostase e sem recompensa. O critério
definido antes de rodar ("IC95 da treinada acima do controle, sem
sobreposição, em ≥5 de 6 seeds") foi cumprido em **6/6 seeds, nas duas
condições**. **Ressalva principal: tudo foi medido no grafo sintético**
(`fetch_connectome.py --synthetic`). O download dos dados reais foi bloqueado
pela rede do ambiente (403 em `neuprint.janelia.org`), então o resultado
ainda precisa ser reproduzido no MaleCNS v1.0. **Atualização (seções 5 e 6):** no
MaleCNS v1.0 real, o critério **não** foi cumprido (2/6 seeds na invertida,
0/6 na normal). Há um efeito consistente e modesto na invertida: 36,9%
contra 26,3% do controle, com a treinada à frente em 6/6 seeds. Isso veio
depois de corrigir a retina (o eixo antigo separava os olhos) e a saturação
dos neurônios LC. A anatomia real limita o que a regra consegue aprender.

#### 1. Nova métrica: saques independentes (`serve_eval.py`)

A taxa de rebatida numa partida contínua permite órbitas travadas (seed 2
nos Achados 13, 17 e 18). No protocolo novo:
- cada tentativa zera o estado da rede;
- a bola sai do centro em direção ao paddle do conectoma, com ângulo
  sorteado por um RNG só da avaliação, e o paddle começa numa altura
  sorteada;
- a tentativa acaba no acerto ou no erro;
- os pesos ficam congelados (o código verifica), sem reforço;
- o resultado sai com IC de Wilson 95%.

As seeds foram separadas antes de qualquer rodada:
- treino: `TRAIN_SEEDS` = 42, 1–5;
- teste: `EVAL_SEEDS` = 1001–1006;
- escolha de hiperparâmetros: `VALIDATION_SEEDS` = 2001–2003.

Referências no grafo sintético: paddle parado 24,7%, oráculo 100%.

#### 2. Por que nada aprendia antes: três problemas que nenhuma regra resolveria

1. **A mosca não via o paddle.** `ball_to_photoreceptor_stimulus` só
   codifica a bola. Com o paddle em altura sorteada, nenhuma política cega ao
   paddle passa de ~60/260 ≈ 23%. A rede inata ficava exatamente nesse nível
   (24,2%, igual ao paddle parado). Nas partidas contínuas isso ficava
   escondido pela correlação entre trajetória da bola e posição do paddle.
   **Correção:** modo opcional `--sensory-mode egocentric`, com a retina
   centrada no paddle (o olho da mosca se move com o corpo). O modo padrão
   continua o original, então os Achados anteriores continuam reproduzíveis.
   Com a retina egocêntrica, a rede inata acerta ~50% (normal) e 5–11%
   (invertida).
2. **Os descendentes estavam saturados.**
   - Eles recebem de 5 a 23 unidades de corrente com limiar 1 e disparam na
     taxa máxima permitida pelo período refratário.
   - A camada object (LC4/LPLC2, três saltos da retina) fica quase toda ativa
     para qualquer posição da bola.
   - Em saturação, o ruído de exploração não muda os spikes, então o
     gradiente da recompensa é zero. É também por isso que o delta ±3 dos
     Achados 14–18 não mudava nada.
   - Uma configuração montada à mão, com só 10 das 387 sinapses plásticas
     (o resto zerado), acerta ~60% na condição invertida: a solução existe,
     mas é esparsa. **Correção:** uma fase de "desenvolvimento" sem
     recompensa, só com escalonamento sináptico homeostático (Turrigiano et
     al. 1998), que traz cada descendente para ~0,3 spike/frame. A homeostase
     não conhece a recompensa nem a direção da bola. Ela fica desligada
     durante o treino, porque, ligada, anulava a depressão das sinapses do
     lado errado (medido).
3. **A regra original não separa "sobe" de "desce" e não tem erro de
   predição.** Esse diagnóstico está no briefing da sessão anterior. A regra
   nova (`sim/covariance_rule.py`) usa `dw = η · (R − ⟨R⟩) · e`:
   - O erro de predição `R − ⟨R⟩` evita o decaimento sistemático: com
     recompensa constante e prevista, a mudança é zero (há teste para isso).
   - A elegibilidade `e = pre × ξ` usa o próprio ruído de exploração ξ
     injetado em cada descendente (perturbação de nó, Fiete & Seung 2006).
     A forma do briefing, `pre × (post − ⟨post⟩)`, também aprende (~50% na
     invertida, nas seeds de validação). Só que o desvio pós-sináptico, com
     média lenta, carrega a variação causada pelo estímulo, e essa variação
     se correlaciona com a recompensa sem relação causal com a ação. A
     versão com ξ é imparcial por construção.
   - A dopamina é um escalar global na regra: nenhuma corrente é injetada
     nos PAM/PPL1, então o reforço não interfere na ação.
   - Os pesos ficam entre 0 e 4× o peso médio inicial.
   - A recompensa vem só do ambiente: +1 se o paddle ficou mais perto da
     altura da bola naquele frame, −1 se ficou mais longe. Ela nunca diz
     qual grupo motor deveria disparar.

#### 3. Resultado (6 seeds de teste × 300 saques, ruído intrínseco σ=6 em todos)

| condição | treinada | controle homeostase | controle inato |
|---|---:|---:|---:|
| invertida | **57,2%** (1030/1800) | 13,1% (236/1800) | 6,7% (120/1800) |
| normal | **99,0%** (1782/1800) | 43,4% (781/1800) | 56,4% (1015/1800) |

Por seed, na condição invertida, a treinada ficou entre 52,0% e 60,7%. O
limite inferior do IC95 dela (≥0,464) ficou sempre acima do limite superior
dos dois controles (≤0,202). Na condição normal, a treinada ficou entre
97,7% e 100%. Os valores por seed estão em `docs/achado-19-resultados-synthetic.jsonl`.

Curva média de aprendizado (100 saques por ponto, pesos congelados):

| saques de treino | 0 | 500 | 1000 | 1500 | 2000 | 2500 | 3000 |
|---|---:|---:|---:|---:|---:|---:|---:|
| invertida | 10,3% | 33,0% | 41,5% | 48,2% | 52,8% | 54,7% | 57,3% |
| normal | 41,2% | 91,8% | 93,7% | 97,8% | 98,2% | 99,5% | 98,3% |

A curva invertida sobe de forma monotônica e se aproxima do teto de ~60% da
configuração montada à mão. O mecanismo foi conferido nos pesos: o treino
reforça "target de cima → grupo que agora significa subir" (0,39→2,25) e
"target de baixo → grupo que agora significa descer" (0,19→0,69). É a
religação que a inversão exige.

#### 4. O que não funcionou e limites honestos

- **Recompensa esparsa sozinha (só acerto/erro no fim do saque) não
  aprendeu** na condição invertida: 5–13% com traço de elegibilidade
  λ=0,9–1,0, η=0,02–0,05, contra ~10% do controle. Com ~10% de acertos
  iniciais, quase todo saque dá o mesmo "erro", e 3.000 episódios não bastam.
  A recompensa densa por frame continua necessária.
- **Os hiperparâmetros foram escolhidos depois de ver resultados**, mas só
  nas seeds de validação (2001/2002): η=0,001, σ=6, 300 saques de
  desenvolvimento, 3.000 de treino. Também foram testadas, sem sucesso,
  homeostase ligada durante o treino, ruído sem homeostase e η maiores. Nas
  seeds de teste, o protocolo rodou uma única vez, com tudo congelado
  (`scripts/run_achado19.py`).
- **A retina egocêntrica é uma mudança de modelo**, não só de avaliação: sem
  ela, a tarefa com paddle em altura aleatória é impossível para qualquer
  regra. Ela é a hipótese mais simples, mas não foi validada contra a
  anatomia real.
- **O ruído intrínseco (σ=6) fica ligado também na avaliação**, igual para
  todas as políticas, porque a homeostase calibrou as taxas com ele. Por
  isso o controle inato normal marca 56,4% aqui, contra ~50% sem ruído.
- **A solução ótima nesse grafo depende de pouquíssimas sinapses cruzadas**
  (a conectividade sintética é topográfica). O teto da condição invertida
  (~60%) é estrutural, não da regra. No grafo real, o teto pode ser outro,
  para cima ou para baixo.
- **Próximo passo obrigatório:** baixar os dados reais (`NEUPRINT_TOKEN` e
  acesso de rede a `neuprint.janelia.org`) e rodar
  `python scripts/run_achado19.py`. O resultado vai para
  `docs/achado-19-resultados-real.jsonl`; o script recusa misturar origens. O grafo real tem ~45 mil neurônios, contra 750 do
  sintético, então cada execução deve ser dezenas de vezes mais lenta.

#### 5. Primeiro contato com os dados reais: a retina 1D separa os olhos

Ao rodar o protocolo no MaleCNS v1.0 real, o controle da condição invertida
ficou em ~19–29%, perto do paddle parado. A investigação:
- `retina_pos` é o 1º componente principal dos centroides de sinapse de
  todos os fotorreceptores juntos. No dado real, esse eixo **separa olho
  esquerdo (880) de olho direito (903)**: metade das posições normalizadas
  fica em [0; 0,11], a outra metade em [0,79; 1], e **nenhum fotorreceptor
  fica no meio**. Esse problema já existia e vale para todos os Achados com
  dados reais, não só para o 19.
- Com a retina egocêntrica, a bola perto do paddle cai nessa faixa vazia e a
  rede fica em silêncio total (zero spikes em todas as camadas). Ela só vê a
  bola quando está a mais de ~185 px do paddle.
- A rede inata real acerta 22–24% nas duas condições, o mesmo que o paddle
  parado. Um oráculo que sabe a direção certa, mas só age quando a bola
  estimula a retina, acerta 23–31%. **Com essa retina, nenhuma regra pode
  mostrar aprendizado no dado real.**
- Os 4 descendentes reais (GF_R, GF_L, DNa10_R, DNa10_L) são divididos em
  "sobe" = lado direito e "desce" = lado esquerdo. Com a retina PCA, a bola
  no alto ativa o lado esquerdo, então o readout "normal" joga para o lado
  errado.

**Correção (opcional):** `--retina-axis elevation`.
- A posição de cada fotorreceptor passa a ser o posto do seu centroide no
  eixo y do neuPrint (dorso-ventral no sistema de coordenadas do FlyEM),
  calculado dentro do próprio olho. Assim os dois olhos veem a bola em
  qualquer altura.
- Isso exige as colunas `retina_x/y/z`, que o `fetch_connectome.py` agora
  salva. `scripts/add_retina_coords.py` as acrescenta a um
  `neurons.parquet` existente, baixando só as sinapses dos fotorreceptores.
- A orientação do eixo y ainda precisa ser conferida contra a anatomia. Um
  sinal invertido só troca qual condição (normal/invertida) começa jogando
  certo.
- No grafo sintético, `elevation` e `pca` dão exatamente as mesmas posições
  (testado), então os resultados acima não mudam.

#### 6. Dados reais com retina por elevação: efeito modesto, critério não cumprido

Com as colunas `retina_x/y/z` baixadas (`scripts/add_retina_coords.py`),
o eixo x separa os olhos (esquerdo 70–89 mil, direito 7–26 mil) e o y varia
de ~15 mil a ~50 mil dentro de cada olho, como se espera do eixo
dorso-ventral. Com `--retina-axis elevation`, a bola passa a estimular a
retina em qualquer altura. Mesmo assim, o protocolo da seção 3 **não
aprendeu** no dado real: 3.000 saques, curva entre 16% e 26% na seed de
validação, igual ao controle. O resultado foi idêntico na máquina do autor
(Windows) e aqui (Linux).

**Onde a informação se perdia.** Um decodificador linear treinado em metade
dos frames e testado na outra (bola acima/abaixo do paddle; acaso = 54,5%):

| camada | sintético | real | real + homeostase nos LC |
|---|---:|---:|---:|
| fotorreceptores | 100% | 100% | — |
| interneurônios | 99% | 97% | — |
| T4/T5 (motion) | 98% | 95% | — |
| LC4/LPLC2 (object) | 95% | **60%** | **~89–90%** |
| LC10a (target) | 96% | 65% | 58–67% |
| descendentes | 76% | 60% | — |

Os neurônios LC recebem centenas de sinapses e ficam saturados: 100% dos
LC4/LPLC2 disparam em todo frame. Estender o escalonamento homeostático,
sem recompensa, a **todas** as entradas dos LC na fase de desenvolvimento
(`--homeostasis-scope visual`, classe `HomeostaticScaler`) recupera a
informação na camada object.

**A anatomia restringe o que pode ser aprendido.** Depois da fase de
desenvolvimento, medimos a sintonia de cada entrada plástica dos 4
descendentes:
- GF_L tem 62 entradas que preferem "bola acima" e 48 "abaixo";
- GF_R tem 35 "acima" e 65 "abaixo";
- os dois DNa10 recebem **só** entradas LC10a que preferem "abaixo" (36 no
  DNa10_R, 22 no DNa10_L, nenhuma "acima").

Na condição **invertida** ("sobe" = lado esquerdo), a recompensa só precisa
reforçar o que o GF_L já oferece. Na **normal** ("sobe" = lado direito), o
DNa10_R empurra na direção errada e o "sobe" teria de ser construído quase
do zero.

**Protocolo final.** Hiperparâmetros fixos da seção 3, mais retina por
elevação, homeostase nos LC e 1.500 saques de treino. A escolha entre eta
0,001/0,003 e ruído 6/3 e o número de saques foram decididos só na seed de
validação 2001. Nas 6 seeds de teste × 300 saques:

| condição | treinada | controle homeostase | controle inato |
|---|---:|---:|---:|
| invertida | **36,9%** [34,7; 39,1] | 26,3% [24,3; 28,4] | 24,7% [22,8; 26,8] |
| normal | 29,2% [27,2; 31,4] | 25,2% [23,3; 27,3] | 26,3% [24,3; 28,4] |

(IC95 de Wilson sobre os 1.800 saques somados)

- **O critério pré-definido não foi cumprido.** A treinada ficou acima dos
  dois controles, com IC sem sobreposição na mesma seed, em **2/6 seeds**
  na invertida e **0/6** na normal. O resultado é negativo segundo a regra
  que fixamos antes de rodar.
- **Há, porém, um efeito consistente na invertida.** A treinada superou o
  controle homeostase nas **6/6 seeds** (+5 a +18 pontos, média +10,6;
  teste do sinal bilateral p = 0,031). Somando os 1.800 saques, os ICs não
  se sobrepõem. A curva média vai de 24,8% (0 saques) para 35,2% (500) e
  37,5% (1.500). Esta análise somada não era o critério pré-definido e deve
  ser lida como exploratória.
- **Na normal, o efeito é pequeno e incerto:** +4 pontos no total, 4 seeds
  positivas e 2 empates.
- **Leitura honesta:** a mesma regra que reaprende a jogar no grafo
  sintético (seção 3) produz, no conectoma real, só um ganho modesto, e
  apenas onde a fiação já oferece o material certo. O gargalo não parece ser
  a regra. Os candidatos são o subgrafo e o modelo neural: 4 descendentes,
  LC10a sem entradas "acima" para os DNa10, LIF com parâmetros uniformes e
  ganho sináptico único. Os próximos passos naturais são incluir mais
  descendentes de direção de voo e testar ganhos por tipo celular.
- Resultados por seed em
  `docs/achado-19-resultados-real-elevation-visual-1500serves.jsonl`.
  Reproduzir: `python scripts/add_retina_coords.py` e depois
  `python scripts/run_achado19.py --retina-axis elevation
  --homeostasis-scope visual --serves 1500`.

Reproduzir:
`python fetch_connectome.py --synthetic --out-dir /tmp/syn && python
scripts/run_achado19.py --data-dir /tmp/syn --out /tmp/achado19.jsonl`
(~15 min com 4 processos).

## Limitações conhecidas

- (Histórico, corrigido no Achado 9) O pool `descending` original tinha só
  4 neurônios (2 GF + 2 DNa02), e só a metade (GF) recebia qualquer sinapse do
  resto do circuito selecionado. A divisão "sobe"/"desce" por índice colocava
  as duas GF (que disparavam) de um lado e as duas DNa02 (mudas) do outro
  (Achado 6). Corrigido substituindo DNa02 por DNa10 (Achado 9) e trocando o
  split por índice por lateralidade real (`_R`/`_L`): os 4 neurônios
  descendentes atuais disparam e respondem com direção correta à posição da
  bola.
- LC10a contribui só ~2.7% do input total de DNa10 (achado-8); os outros
  ~97% vêm de vias fora do escopo deste projeto (PLP213, AOTU, IB, PS0xx),
  então DNa10 não é um canal "limpo" dedicado à visão, mesmo depois do
  redesign. (Refinado no Achado 13, com correção pós-hoc): um sinal
  sintético forte no mesmo local produz uma melhora fraca mas real (Fase 1;
  o número original sugeria "quase perfeito", mas isso vinha de uma seed com
  dinâmica travada/degenerada, e sem ela o efeito é bem mais modesto, ver
  `docs/achado-13-investigacao-seed2.md`). O achado mais robusto é da Fase 2:
  a regra de plasticidade de três fatores, como implementada, enfraquece
  sistematicamente qualquer sinapse nessa via ao longo de uma rodada típica
  (mais eventos de reforço negativo que positivo), inclusive uma já forte o
  bastante pra funcionar bem. É um resultado consistente em 5 de 6 seeds,
  não dependente do ajuste da Fase 1. Ver Achado 13.
- A posição retinotópica dos fotorreceptores (Achado 3) é uma aproximação via
  PCA de coordenadas de sinapse, não uma medida validada contra a anatomia
  real do olho, e pode não corresponder ao eixo dorso-ventral verdadeiro.
- Mapear lateralidade (esquerda/direita, o que DNa10/GF codificam de fato:
  controle de direção de caminhada) em "sobe"/"desce" de um paddle vertical
  (Achado 9) é uma escolha de engenharia, não um fato biológico.
- O efeito de plasticidade encontrado no Achado 10 (redução de magnitude do
  delta, consistente em 4/4 seeds, p=0.125 bicaudal) não é estatisticamente
  conclusivo com amostra tão pequena. A hipótese mecanicista proposta lá
  (atenuação por decaimento de peso, não reforço direcionado) foi
  corroborada de forma mais direta no Achado 13 (Fase 2): um canal sintético
  isolado, começando num peso forte o bastante pra funcionar bem (confirmado
  na Fase 1), decai de forma consistente nas 6 seeds testadas e termina com
  desempenho pior que um controle de peso fixo, mesmo padrão de decaimento,
  agora isolado do resto da rede real.
- Reward denso (Achado 11/12) não resolveu a ambiguidade: 6 seeds, p=0.68
  (teste de sinal e teste t pareado), não dá pra distinguir de ruído. A
  Fase 3 (recozimento denso→esparso) planejada não foi executada por decisão
  explícita de custo/benefício (~2h+ de compute em cima de sinal já
  ambíguo), não por limite técnico.
- A regra de plasticidade de três fatores implementada (`sim/network.py`)
  tem um viés estrutural de enfraquecer sinapses ao longo de uma rodada
  típica, porque eventos de reforço negativo (bola perdida) são mais
  frequentes que positivos (bola rebatida), confirmado isoladamente no
  Achado 13, Fase 2. **(Tentativa de correção no Achado 14, sem sucesso)**:
  duas opções (`freq_normalized`, `tonic_baseline`) foram implementadas e
  testadas no mesmo canal isolado; uma piora o problema de forma
  estatisticamente significativa (p=0.009, e essa piora não vem do teto de
  escala saturando como a hipótese inicial supunha, refutado por
  instrumentação direta), a outra não tem efeito líquido distinguível de
  ruído (p=0.98, com variância alta explicada por episódios reais de
  desempenho instável numa seed auditada, não por artefato de órbita
  travada). O mecanismo de atribuição de crédito continua sem correção
  funcional conhecida; normalizar pela frequência de eventos e usar linha
  de base não-zero, as duas ideias mais óbvias, já foram tentadas e
  descartadas, não são mais recomendação de trabalho futuro em aberto.
- Um baseline linear sem rede neural, mesmo sujeito a atraso de 1 frame e
  resolução sensorial igual à do subgrafo real (`game/baseline_policy.py`,
  `run_delayed_linear_baseline`), continua acertando 100% em todas as
  seeds testadas: essas duas restrições não equalizam a comparação porque
  o gargalo real do circuito neural é o readout de poucos neurônios do
  lado motor, não a resolução sensorial ou o atraso. Não existe ainda uma
  comparação "circuito real vs. baseline" que seja genuinamente informativa
  nesse sentido.
- **(Achado 15, reformula retroativamente os Achados 13 e 14)** O controle
  de peso fixo usado em todos os experimentos de plasticidade (`8.93`) fica
  a só 0.314 de média na varredura de peso, longe do peso-teto real (`40`,
  média 0.994) — uma razão de 4.48x. As conclusões sobre a regra de
  plasticidade em si continuam válidas (medidas corretamente contra o
  controle que foi de fato usado), mas "controle sem plasticidade" nos
  Achados 13-14 não era o melhor controle possível dessa arquitetura. Mais
  relevante: com `PLASTIC_DELTA_MAX=3.0`, a plasticidade partindo de 8.93
  só alcança `[5.93, 11.93]`, nunca perto do peso-teto de 40, mesmo no
  melhor caso hipotético. Se a plasticidade funcionaria partindo do
  peso-teto é uma pergunta em aberto — a Fase 3 do Achado 15 que
  responderia isso foi cancelada por restrição de tempo antes de produzir
  qualquer resultado, não é um achado negativo, é trabalho não feito.
- ~~O oponente (paddle direito) é uma IA que nunca erra; serve só pra manter a
  bola em jogo, não é um adversário real.~~ **Corrigido depois do Achado 18**
  (`game/pong.py`, commit `fac9325`): a IA agora reage com atraso
  (`AI_REACTION_FRAMES`), tem ruído gaussiano e uma probabilidade de "piscar"
  (`AI_MISS_PROB`), e fica mais difícil conforme o lado esquerdo marca ponto
  (currículo). **Todos os achados até aqui, incluindo o 18, foram medidos
  contra a IA antiga (invencível)** — não são diretamente comparáveis a
  qualquer trabalho futuro (torneio evolutivo, por exemplo) que use a IA
  nova. A auditoria pós-hoc da seed 2 do Achado 18 precisou recarregar a IA
  antiga especificamente pra reproduzir os números do dataset já coletado.
- **(Achado 17, correção pós-hoc)** Das duas seeds "sortudas" usadas pra
  argumentar que a plasticidade "perturba trajetória sortuda" (Achado 17),
  só a seed 1 é desempenho genuíno; a seed 2 era órbita travada (mesma
  assinatura estatística da seed 2 do Achado 13: correlação de -0.9999...
  entre janelas distantes no período estável). A conclusão estatística
  agregada da Fase 2 não muda, mas a leitura qualitativa vale só pra metade
  dos casos que pareciam sustentá-la.
- **(Achado 18)** A seed 2 travou em órbita degenerada pela terceira vez
  (Achados 13, 17 e 18), em 2 dos 3 modos de plasticidade testados no peso
  40 (correlação exata de 1,0 entre janelas distantes, faixa de `ball_y`
  <24% da tela) — reforça que essa seed específica deve ser tratada como
  suspeita por padrão em qualquer rodada futura com o mesmo gerador de jogo,
  não investigada do zero a cada vez. Além disso, todas as 18 seeds
  plásticas terminaram com peso dentro da região "saturada" da tabela do
  Achado 15 (≥38), mas o desempenho absoluto variou de 0,20 a 1,0 mesmo
  assim — peso final dentro do platô não garante desempenho alto quando o
  peso chegou lá via plasticidade (trajetória dinâmica), diferente de um
  peso fixo estático a vida toda.

## Citação

Berg et al., "A connectome of the adult male Drosophila central nervous
system", *Cell*, 2026 (MaleCNS v1.0). Dados acessados via neuPrint
(neuprint.janelia.org, dataset `male-cns:v1.0`).
