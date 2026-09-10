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
| `interneuron` | L1-L5, Mi1/Mi4/Mi9, Tm1/Tm2/Tm4/Tm9 | lâmina + medula: ponte entre fotorreceptor e T4/T5 (ver Achado 2 abaixo — adicionada depois de descobrir que faltava) |
| `motion` | T4, T5 (subtipos a-d) | detecção de movimento local |
| `object` | LC4, LPLC2 | detecção de objeto em expansão/loom |
| `descending` | DNa02, Giant Fiber (GF), DNp01 | vias motoras descendentes |
| `dopaminergic` | PAM*, PPL1* | reforço (PAM ~ recompensa, PPL1 ~ aversivo, cf. Aso et al. 2014) |

**Contagem real final** (dataset `male-cns:v1.0`): **44 714 neurônios,
1 147 005 sinapses** — `photoreceptor`: 1783, `interneuron`: 28 699, `motion`:
13 585, `object`: 311, `dopaminergic`: 332, `descending`: **4**. Grafo
sintético (dev/teste offline): 690 neurônios, 10 131 sinapses. Ver
`connectome_data/metadata.json` pra contagem exata da execução mais recente.

O pool `descending` real é bem menor do que o esperado (4 neurônios no total,
contra 20 no gerador sintético) — provavelmente porque DNa02, Giant Fiber e
DNp01 são vias com poucas cópias (tipicamente 1-2 por lado do corpo),
diferente de interneurônios do lobo óptico que se repetem por coluna
retinotópica. Isso deixa o readout motor com apenas 2 neurônios por grupo
("sobe"/"desce"), extremamente sensível a ruído de disparo — é a causa raiz
de boa parte dos problemas na seção de validação.

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
(N substeps, propagando `photoreceptor → interneuron → motion → object →
descending`) → `read_motor_action` → `PongGame.step(action)` → se houve
contato ou perda de bola, aplica um pulso de corrente nos neurônios
dopaminérgicos, cuja atividade resultante modula (regra de três fatores tipo
STDP, cf. Frémaux & Gerstner 2016) **apenas** os pesos das sinapses
`motion`/`object` → `descending` marcadas como plásticas — o resto do grafo é
fixo. A dopamina é modelada como sinal global (volume transmission), não
exige aresta anatômica dopaminérgico→alvo.

A divisão do pool `descending` em dois grupos de leitura ("sobe"/"desce") é
**uma escolha de modelagem, não um fato biológico**: é a metade de menor
índice vs. a de maior índice do conjunto selecionado (com só 4 neurônios reais
no total, isso vira 2 vs. 2). Essa divisão produz um viés estrutural de
excitabilidade entre os dois grupos sem relação com a posição da bola; o
código mede esse viés numa fase de calibração (bola parada no centro, 200
frames) e o subtrai do sinal motor (`game/motor_read.py`, parâmetro
`baseline_diff`).

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
plasticidade ligada e desligada, seed a seed**. Ou seja: mesmo uma mudança de
peso substancial (chegando ao limite configurado) não tem efeito mensurável
no comportamento. Isso não é mais o bug do achado 5 (já descartado por esse
mesmo experimento) — é evidência de que as 311 sinapses plásticas
(`object`→`descending`) não têm alavancagem suficiente sobre um readout de
apenas 4 neurônios pra alterar qual ação é tomada.

**Conclusão honesta:**
1. **Não há evidência de que a plasticidade dopaminérgica implementada
   produza aprendizado**, com dados sintéticos ou reais, mesmo depois de
   corrigir um bug real que invalidava comparações anteriores.
2. **O rastreamento sensório-motor básico (bola → ação) não funciona de forma
   confiável com dados reais**, mesmo depois de fechar o circuito anatômico
   (achado 2) e adicionar posição espacial real aos fotorreceptores (achado
   3) — o gargalo final parece ser o tamanho ínfimo do pool `descending` (4
   neurônios reais) combinado com incerteza sobre se o eixo de posição
   encontrado via PCA corresponde de fato a uma dimensão retinotópica
   utilizável.

Isso é reportado como resultado final, não escondido — mesma postura do
`doomfly` ao documentar quando o modelo não aprende de fato. Trabalho futuro
que não foi feito aqui: verificar a orientação real dos eixos espaciais do
dataset (ex. via ROIs conhecidas do lobo óptico), ou selecionar mais
neurônios descendentes (relaxando o filtro de tipos) pra ter um readout menos
ruidoso.

Reproduzir:

```bash
python fetch_connectome.py --token SEU_TOKEN   # ou --synthetic
python main.py --headless --frames 30000
```

## Limitações conhecidas

- O pool `descending` real tem só 4 neurônios (2 por grupo "sobe"/"desce"),
  então o readout motor é binário/muito ruidoso por construção.
- A posição retinotópica dos fotorreceptores (Achado 3) é uma aproximação via
  PCA de coordenadas de sinapse, não uma medida validada contra a anatomia
  real do olho — pode não corresponder ao eixo dorso-ventral verdadeiro.
- A divisão dos neurônios descendentes em grupos "sobe"/"desce" é arbitrária
  (metade/metade por índice), não baseada em vias motoras conhecidas
  associadas especificamente a subir/descer.
- Nenhuma evidência de aprendizado dopaminérgico foi observada nas condições
  testadas.
- O oponente (paddle direito) é uma IA que nunca erra; serve só pra manter a
  bola em jogo, não é um adversário real.

## Citação

Berg et al., "A connectome of the adult male Drosophila central nervous
system", *Cell*, 2026 (MaleCNS v1.0). Dados acessados via neuPrint
(neuprint.janelia.org, dataset `male-cns:v1.0`).
