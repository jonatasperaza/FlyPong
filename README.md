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
resultados reais dos testes com dados reais do MaleCNS v1.0, ao longo de 14
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
não pela ordem cronológica dos 14 achados abaixo.

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
errada (teste t pareado t(5)=-4.13, p=0.009; teste de sinal p=0.031). A
normalização por frequência, do jeito que foi implementada, piora o
problema em vez de corrigi-lo — uma hipótese não confirmada é que a média
móvel (decaimento 0.999, ~1000 passos pra estabilizar) amplifica ruído no
início de cada rodada antes de convergir pra uma estimativa de frequência
confiável, mas isso não foi investigado a fundo.

`tonic_baseline` não falha tão claramente, mas também não funciona: 2 de 6
seeds melhoram bastante (seed 1 e seed 3, essa última com o peso final
ficando bem mais perto do valor inicial e a taxa de rebatida subindo pra
0.869), 3 pioram, 1 fica igual. A diferença média contra o controle é
essencialmente zero (t(5)=-0.02, p=0.98; teste de sinal 2 de 6, p=0.69) e o
peso continua decaindo na maioria das seeds, só que com desvio padrão bem
maior (0.39 contra 0.14 do `freq_normalized`). O resultado da seed 3 é
chamativo o bastante pra merecer a mesma desconfiança que a seed 2 do
Achado 13 recebeu antes de virar evidência: não foi auditado se é
rastreamento genuíno ou outro caso de órbita dinâmica travada (ver Achado
13, seção de investigação da seed 2, pro método). Fica registrado aqui como
não verificado, não como achado positivo.

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

A Fase 4 (comparação com um baseline trivial sem rede neural nenhuma,
`game/baseline_policy.py`: `action = sign(ball_y - paddle_y)`, mesma
métrica) foi rodada de qualquer forma, já que é barata e não depende do
resultado da Fase 3. Nas 6 seeds testadas, o baseline trivial atinge taxa de
rebatida 1.0 do início ao fim, sempre: por ter latência zero e acesso
perfeito à posição da bola (sem ruído de disparo, sem os 4 substeps de
atraso do LIF), ele nunca erra, o que o torna essencialmente imbatível por
construção. Isso não é reportado como uma derrota do circuito do conectoma:
a comparação relevante seria o quão perto o circuito chega desse teto, não
se ele vence, e como a plasticidade não produziu um circuito claramente
melhor que o baseline sem ela (Achados 10-14), essa comparação de
proximidade não foi refeita aqui.

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
  estatisticamente significativa (p=0.009), a outra não tem efeito líquido
  distinguível de ruído (p=0.98). O mecanismo de atribuição de crédito
  continua sem correção funcional conhecida; normalizar pela frequência de
  eventos e usar linha de base não-zero, as duas ideias mais óbvias, já
  foram tentadas e descartadas, não são mais recomendação de trabalho
  futuro em aberto.
- O oponente (paddle direito) é uma IA que nunca erra; serve só pra manter a
  bola em jogo, não é um adversário real.

## Citação

Berg et al., "A connectome of the adult male Drosophila central nervous
system", *Cell*, 2026 (MaleCNS v1.0). Dados acessados via neuPrint
(neuprint.janelia.org, dataset `male-cns:v1.0`).
