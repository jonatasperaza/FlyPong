# FlyPong — opiniões do GPT-6 Terra

> Documento de revisão independente, escrito a partir de uma leitura estática
> do repositório em 16 de setembro de 2026. Não substitui os resultados
> experimentais registrados no `README.md`; serve para orientar as próximas
> decisões de engenharia e pesquisa.

## Resumo honesto

Este é um projeto de pesquisa computacional bem mais maduro do que um demo de
"rede neural jogando Pong". Ele usa dados reais do conectoma, explicita as
abstrações, roda controles, documenta resultados negativos e revisita
conclusões quando uma seed ou uma métrica parece boa demais. Essa postura é o
maior ativo do FlyPong.

O projeto **ainda não demonstrou que uma mosca simulada aprende Pong**. O que
existe é um simulador de circuito inspirado em um subgrafo do conectoma de
*Drosophila*, com uma regra de plasticidade de três fatores. Nos experimentos
registrados até aqui, essa regra não produziu melhoria robusta e reproduzível
de desempenho sobre os controles. Isso não é uma falha do projeto: é um
resultado científico útil, desde que seja comunicado assim.

O objetivo mais promissor não é provar rapidamente que "a mosca aprendeu".
É construir uma bancada aberta, reproduzível e bem instrumentada onde pessoas
possam testar hipóteses de percepção, memória, crédito temporal e controle em
circuitos derivados de conectomas.

## O que já está muito bom

### Rigor e honestidade

- O `README.md` registra bugs, correções, hipóteses refutadas e resultados
  ambíguos, em vez de selecionar somente os números favoráveis.
- O projeto compara condições com e sem plasticidade, usa múltiplas seeds e
  já identificou artefatos de trajetórias determinísticas.
- A separação do RNG de recompensa e do RNG da física do jogo foi uma correção
  importante: uma mudança no reforço não pode mudar secretamente a trajetória
  da bola quando a plasticidade está desligada.
- A auditoria anatômica é cuidadosa. Em particular, o projeto reconhece que
  conectividade existente não equivale automaticamente a uma via funcional
  apropriada para a tarefa.

### Estrutura de software

- Dados (`connectome_data`), simulação (`sim`), ambiente (`game`) e coleta
  (`runs`) estão separados de forma compreensível.
- Parquet e saídas estruturadas facilitam reanálise.
- O modo headless permite experimentos sem interface gráfica.
- Os baselines são uma boa iniciativa. O baseline linear revela que a tarefa
  pode ser simples demais para diferenciar uma política direta do circuito.

### O resultado negativo é informativo

O achado de que a regra atual tende a enfraquecer a via e não melhora o
controle é exatamente o tipo de informação que evita meses de ajuste cego.
Ele sugere que o problema é de atribuição de crédito, representação e desenho
da tarefa — não apenas de `learning rate`.

## Limites atuais: o que o modelo representa e o que ele não representa

É importante manter estas frases próximas de qualquer demonstração pública.

- O grafo vem de conectividade real, mas sinapses são convertidas em pesos
  positivos proporcionais à contagem. O simulador não modela tipo de
  neurotransmissor, sinal inibitório/excitatório, atraso por sinapse,
  compartimentos dendríticos ou plasticidade biologicamente calibrada.
- A retina é uma projeção 1D aproximada. A associação entre seu eixo e o eixo
  vertical do Pong é uma escolha de engenharia, não uma propriedade anatômica
  validada.
- Lateralidade de neurônios descendentes foi mapeada para subir/descer o
  paddle. Também é uma convenção de controle, não uma afirmação de que DNa10
  naturalmente controla esse movimento.
- A tarefa de Pong exige acompanhar posição, estimar a trajetória futura e
  agir no tempo certo. O circuito atual recebe uma observação extremamente
  reduzida e não tem uma representação explícita de velocidade, memória ou
  estado recorrente funcional para previsão.

Portanto, a formulação correta é: **"um agente de controle inspirado em um
subgrafo de conectoma"**, e não "uma mosca digital aprendendo a jogar".

## O que ajuda a chegar a aprendizado real no simulador

### 1. Transformar o experimento em um protocolo, não em uma sequência manual

Cada execução deve registrar automaticamente:

- seed do jogo, seed(s) do modelo e seed do reforço;
- commit Git, hash dos arquivos de dados e todos os argumentos de linha de
  comando;
- condição experimental, duração, taxa de rebatida total e séries temporais;
- pesos iniciais/finais, taxa de disparo por população e estatísticas de
  recompensa;
- versão do ambiente e dependências.

Também vale expor `--seed` na CLI. Hoje o jogo é iniciado com seed fixa em
`main.py`, o que é bom para depuração, mas ruim como interface principal de
experimentos.

**Critério de sucesso sugerido:** toda tabela do README deve poder ser
regenerada por um único comando ou arquivo de configuração versionado.

### 2. Definir uma métrica primária resistente a trajetórias determinísticas

Comparar o primeiro e o último quinto de uma execução é um diagnóstico útil,
mas não demonstra aprendizado por si só: as janelas são sobrepostas e a física
determinística pode criar períodos fáceis ou órbitas repetitivas.

A métrica principal deveria ser uma diferença pareada por seed:

`desempenho(plasticidade ligada) - desempenho(controle, mesmos eventos)`.

Idealmente, o ambiente precisa separar a aleatoriedade da trajetória da bola
da aleatoriedade do agente. Uma opção é gerar previamente uma lista de saques
e ângulos para cada seed, reutilizá-la em todas as condições, e reportar
intervalo de confiança por bootstrap além do teste estatístico escolhido.

### 3. Criar ablações que respondam perguntas causais

Antes de expandir o conectoma, rodar estas comparações baratas:

| Pergunta | Comparação mínima | O que ela separa |
| --- | --- | --- |
| A rede transmite sinal sensorial útil? | rede intacta vs. sinapses embaralhadas, mesma distribuição de pesos | topologia real vs. mera atividade |
| Há controle além de viés motor? | entradas embaralhadas no tempo vs. entrada normal | ação guiada por visão vs. viés estrutural |
| A regra aprende, e não só perturba? | pesos congelados vs. plásticos, mesmas seeds e currículo | efeito causal da atualização |
| A tarefa exige memória? | observação atual vs. pilha de observações/velocidade codificada | gargalo perceptivo-temporal |
| O resultado depende da seed? | conjunto de avaliação bloqueado, nunca usado para ajuste | generalização experimental |

Não é necessário que todo experimento seja biologicamente perfeito. É
necessário que cada um elimine uma explicação alternativa concreta.

### 4. Separar "teste de teto" de "hipótese biológica"

Os modos `synthetic_strong` e `synthetic_plastic` injetam nos neurônios
motores a direção correta baseada em `ball_y - centro_do_paddle`. Isso é
informação privilegiada: excelente para responder "esse readout conseguiria
controlar o paddle se recebesse o sinal certo?", mas não evidencia uma via
visual biologicamente plausível nem aprendizado sensorial.

Manter esses modos é correto, desde que sejam chamados de **oráculos de
capacidade** ou **ablação de canal**. Nunca usar seus resultados como prova de
aprendizado da mosca simulada.

### 5. Atacar o crédito temporal explicitamente

O reforço acontece depois da ação e o evento relevante pode estar muitos
frames depois da percepção que importou. Uma regra local simples baseada em
traços que decaem rapidamente dificilmente resolve isso sem uma representação
de estado.

Hipóteses mais produtivas que apenas mudar o `learning rate`:

- traços de elegibilidade por sinapse com escala de tempo calibrada à duração
  de uma troca de bola;
- modulação baseada em vantagem/erro de previsão, não somente recompensa
  bruta positiva ou negativa;
- uma população recorrente pequena e explícita para memória de velocidade e
  fase da trajetória;
- currículo: seguir uma bola lenta, depois variar ângulos e velocidades;
- recompensas densas cuidadosamente desenhadas como diagnóstico, seguidas de
  avaliação somente com recompensa esparsa;
- comparação com uma regra de referência simples (por exemplo, policy-gradient
  ou R-STDP em uma rede pequena sem conectoma) sob exatamente o mesmo
  ambiente.

O objetivo é descobrir se o obstáculo é a regra de aprendizado, a informação
de entrada, o readout ou a própria tarefa.

## O que provavelmente piora o projeto

### Ajuste excessivo em poucas seeds

Não escolher pesos, limiares, modos de plasticidade ou arquitetura olhando o
mesmo pequeno conjunto de seeds usado para avaliar o resultado final. Separe:

- seeds de desenvolvimento: livres para explorar;
- seeds de validação: usadas raramente;
- seeds de teste: bloqueadas até o protocolo estar congelado.

Sem essa separação, uma melhoria pode ser apenas ajuste à órbita específica da
bola.

### Confundir mais neurônios com mais capacidade de aprender

Adicionar o complexo central é uma hipótese científica interessante, mas
introduz dinâmica recorrente e atratores, uma categoria nova de problema. Não
deve ser feito apenas porque a rede atual não aprendeu. Primeiro prove, com
um protótipo mínimo, que uma memória recorrente melhora uma tarefa controlada
e que a dinâmica é estável.

### Reduzir desempenho sem preservar semântica

Há mudanças não commitadas que adicionam o caminho rápido `--fast` com poda
causal. Na implementação atual, `_causal_prune` converte as colunas `pre` e
`post` de `bodyId` para índices locais, mas o construtor depois tenta mapeá-las
novamente como se ainda fossem `bodyId`. Isso pode eliminar as arestas do grafo
podado. Resultados de `--fast` não devem ser usados até existir um teste que
prove:

1. as arestas do grafo podado continuam válidas;
2. as populações de entrada, saída e sinapses plásticas esperadas continuam
   presentes;
3. para uma configuração equivalente, o caminho rápido preserva o resultado
   ou documenta exatamente a diferença semântica.

Otimização só é um avanço quando acompanha testes de equivalência.

### Remover documentação científica durante refactors

As alterações locais de desempenho também reduziram bastante docstrings e
contexto de `main.py` e `sim/network.py`. Comentários extensos não são um
substituto para testes, mas neste projeto eles registram quais decisões foram
biológicas, quais foram computacionais e quais já falharam. Preservar esse
contexto reduz a chance de reintroduzir erros já investigados.

## Roteiro recomendado

### Fase A — tornar a base confiável

1. Corrigir ou desativar a poda causal até haver testes automatizados.
2. Adicionar testes para carga de dados, contagens por papel, conectividade de
   caminhos essenciais, determinismo por seed e comportamento com plasticidade
   zerada.
3. Criar um `experiment.py` ou arquivo de configuração que rode matrizes de
   condições e escreva resultados completos em JSONL/CSV.
4. Congelar um conjunto de seeds de avaliação e reproduzir uma tabela já
   existente de ponta a ponta.

### Fase B — entender o problema mínimo

1. Criar uma versão simplificada de Pong com bola lenta, ângulos discretos e
   sem oponente.
2. Provar que uma política de referência aprende nessa versão usando a mesma
   observação e o mesmo sinal de recompensa.
3. Provar que a rede conectômica transmite informação sobre posição e, depois,
   velocidade para o readout; medir isso diretamente com classificadores ou
   correlação decodificada, não apenas pelo placar.
4. Só então avaliar uma alteração de plasticidade ou memória recorrente.

### Fase C — construir um benchmark aberto

Quando a base estiver estável, o projeto pode virar algo muito interessante
para a comunidade:

- ambientes com dificuldade crescente e interfaces estáveis;
- conectomas/subgrafos versionados e checados;
- protocolos e seeds públicos;
- leaderboard que compare desempenho, eficiência e fidelidade das restrições;
- baselines não conectômicos claros;
- guias para acrescentar novas tarefas, sensores, readouts e regras de
  plasticidade sem adulterar o benchmark.

O nome da ambição poderia ser algo como "Connectome Learning Bench": não
promete consciência nem uma mosca literal, mas oferece uma infraestrutura
honesta para testar quando estruturas conectômicas ajudam agentes a aprender.

## Orientação para agentes de código (Claude Code e outros)

Ao receber uma tarefa neste repositório, siga estas regras:

1. Leia `README.md` e os documentos de achados relevantes antes de propor uma
   mudança. Não trate conclusões anteriores como detalhes descartáveis.
2. Nunca declare aprendizado com base em uma única seed, um delta interno ou
   uma curva visualmente bonita.
3. Não altere simultaneamente ambiente, observação, readout e regra de
   plasticidade; isso destrói a atribuição causal do experimento.
4. Para toda otimização, escreva antes um teste de invariância ou deixe claro
   que é uma nova condição experimental, não uma versão equivalente.
5. Preserve o README científico; resultados incompletos devem ser marcados
   como incompletos, não resumidos como conclusão.
6. Trate canais sintéticos como ablações/oráculos e os mantenha separados dos
   resultados do circuito real.
7. Prefira uma melhoria mensurável e reprodutível a uma alteração grande e
   biologicamente sedutora, porém impossível de interpretar.

## Conclusão

Vale muito a pena continuar, mas com a ambição correta: fazer o FlyPong se
tornar uma plataforma confiável para descobrir **quais ingredientes faltam
para que um circuito inspirado em conectoma aprenda uma tarefa de controle**.

Se algum dia o agente aprender Pong de forma robusta, com avaliação bloqueada
e controles fortes, o resultado terá credibilidade justamente porque este
projeto documentou com o mesmo cuidado todas as vezes em que ele ainda não
aprendeu.

## Nota de continuidade — resultados recentes e mudanças de 16/09/2026

Esta seção é deliberadamente datada. Ela registra o estado da investigação no
momento desta revisão para que o próximo agente não trate uma rodada parcial
como um resultado encerrado.

### Leitura de `runs/achado18_fase1.jsonl`

O arquivo foi concluído com os 24 resultados previstos (4 modos × 6 seeds).
A linha antes ausente, `freq_normalized`, seed 5, foi reexecutada pelo
`scripts/run_achado18.py` com 15.000 frames e levou 234 s; terminou em taxa
final 0,333 e peso 39,979. A condição de controle (peso sintético fixo 40)
terminou com taxa de rebatida 1,00 nas seis seeds.

| Condição | Resultados presentes | Média da taxa final | Diferença final média vs. controle pareado | Leitura responsável |
| --- | ---: | ---: | ---: | --- |
| controle | 6/6 | 1,000 | 0,000 | referência no teto; deixa pouco espaço para melhoria |
| `original` | 6/6 | 0,821 | -0,179 | não supera o controle em nenhuma seed; três empates e três quedas materiais |
| `freq_normalized` | 6/6 | 0,706 | -0,294 | não supera o controle; três empates e três quedas materiais |
| `tonic_baseline` | 6/6 | 0,553 | -0,447 | completo e desfavorável: quatro quedas materiais e dois empates |

No modo `original`, os pesos terminaram acima de 40 em todas as seis seeds
(aproximadamente 40,54 a 43,00), enquanto o desempenho final piorou nas seeds
42, 4 e 5. Esse contraste é um sinal mecanístico importante: **simplesmente
fortalecer o canal não é uma explicação suficiente para melhorar o controle**.

Análise exploratória de endpoint, pareada contra o controle: `original`
teve t(5)=-2,01, p=0,101; `freq_normalized`, t(5)=-2,06, p=0,095; e
`tonic_baseline`, t(5)=-3,06, p=0,028 (bicaudal). Esses p-valores **não são
a nova métrica primária pré-registrada**: o controle está no teto e endpoint
foi escolhido depois de ver os dados. Ainda assim, todos os sinais apontam na
direção desfavorável à plasticidade nessa configuração, em especial o modo
`tonic_baseline`.

### Correções implementadas nesta revisão

1. A poda causal em `sim/network.py` agora preserva `pre` e `post` como
   `bodyId`s no parquet. Antes, o caminho podado os convertia para índices
   locais e o carregador tentava mapeá-los novamente como IDs, podendo remover
   silenciosamente todas as arestas. O carregamento também falha cedo se uma
   aresta apontar para um ID ausente, em vez de descartá-la.
2. Foi adicionado `tests/test_causal_prune.py`, que cria um conectoma mínimo,
   poda o grafo e verifica que as duas arestas causais ainda chegam ao
   `ConnectomeNetwork`.
3. `main.py` aceita agora `--seed`, `--reward-mode` e `--mix-ratio`, para que
   a CLI exponha parâmetros que já existiam no código.
4. Cada execução headless passa a salvar um JSON ao lado do CSV, contendo
   seed, argumentos efetivos, configurações de plasticidade, tamanho do grafo
   e relatório. Esse arquivo é parte do artefato experimental e deve ser
   mantido junto de qualquer tabela publicada.

### Verificação executada

- `python tests/test_causal_prune.py -v`: passou. O teste cobre a invariância
  de IDs e a presença das arestas causais depois de recarregar o parquet
  podado.
- `python main.py --headless --fast --frames 1 --seed 42 --no-history`:
  passou. O grafo real podado carregou com 44.835 neurônios e avançou um
  frame. O resultado não é um experimento de aprendizado; é somente um smoke
  test do caminho de execução.

O conjunto original possui 44.989 neurônios. A poda removeu apenas 154
(~0,34%), portanto a aceleração de `--fast` vem quase toda de um substep por
frame e da desativação do histórico, não de redução material do conectoma.
Não apresentar a poda como uma otimização grande sem medir tempo e preservar
semântica da condição experimental.

### Próximo passo obrigatório

Antes de mudar novamente a regra de plasticidade, registre como conclusão
negativa desta configuração e defina a métrica primária do próximo protocolo.
`scripts/run_achado18.py` agora preserva a receita reconstruída e retoma o
JSONL sem sobrescrever células existentes; os 23 resultados anteriores,
contudo, foram gerados interativamente e não contêm metadados completos.
Resultados futuros devem nascer pelo executor, não ser reconstruídos depois.
