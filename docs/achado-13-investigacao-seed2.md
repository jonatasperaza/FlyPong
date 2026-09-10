# Investigação: seed 2 da Fase 1 do Achado 13 (taxa de rebatida = 1.000)

## Pergunta

Na Fase 1 do Achado 13 (`signal_mode="synthetic_strong"`, sem plasticidade),
a seed 2 terminou com taxa de rebatida final **1.000**, bem acima das outras
3 seeds (0.137–0.390). Antes de usar esse número como parte da justificativa
pra prosseguir pra Fase 2, esta nota confirma se é rastreamento genuíno ou um
artefato de dinâmica degenerada específico dessa seed.

## Passo 1-2: a seed 2 é "fácil" independente do sinal?

Rodando a mesma seed 2 com `signal_mode="real"` (circuito original, sem
reforço sintético), 15 000 frames, `LR=0`:

| signal_mode | tentativas | bounces | misses | taxa final |
|---|---|---|---|---|
| `real` | 77 | 11 | 66 | 0.100 |
| `synthetic_strong` | 123 | 109 | 14 | 1.000 |

Com sinal real, a seed 2 é uma das piores entre as 4 testadas (11/77 = 14%
de acerto bruto), o que bate exatamente com o número já registrado no
Achado 10/12 pra essa seed no modo controle. Não é uma seed estruturalmente
fácil: o salto de desempenho só aparece com o sinal sintético forte.

## Passo 2 (continuação): onde os erros aconteceram, e o que veio depois

Sequência completa dos 123 eventos (`runs/seed2_synthetic_strong_events.csv`):
os 14 misses ocorreram todos entre os índices 0 e 17 (os primeiros ~18
eventos, do frame 68 ao início da rodada). Depois disso, são 105 rebatidas
consecutivas sem nenhum erro até o frame 14 952 (fim da janela de 15 000
frames analisada).

A velocidade da bola atinge o teto (`BALL_SPEED_MAX=8.75`) no frame 4966,
cerca de 1/3 da rodada, e permanece lá pelo resto do tempo. 97 dos 123
eventos (79%) acontecem com a bola em velocidade máxima ou perto dela.

## Passo 2 (aprofundado): as 105 rebatidas são desafios independentes?

Comparando a posição Y da bola em duas janelas distantes no tempo, ambas já
depois do teto de velocidade ser atingido (frames 6000–7500 vs.
10000–11500), a correlação entre as duas janelas é de -0.9999, com a
posição Y restrita a uma faixa estreita (225–263, ~12% da altura de 320px da
tela) nos dois casos.

Esse padrão indica um ciclo dinâmico travado, não coincidência ou ruído. Uma
vez que o sistema (bola e paddle nosso, determinístico dado sinal fixo sem
plasticidade, mais a IA adversária, também determinística) acerta um número
suficiente de rebatidas seguidas, a trajetória da bola entra numa órbita
estreita e repetitiva que se auto-sustenta indefinidamente: nenhum
`reset_ball` acontece enquanto não há perda de ponto, e a mesma faixa
estreita de posições nunca desafia o paddle o bastante pra ele errar de
novo. As 105 "rebatidas" acabam sendo uma única situação fácil repetida, não
105 situações diferentes sendo resolvidas.

## Conclusão

O resultado não é nem "genuíno" no sentido de robustez ampla, nem
"totalmente degenerado" no sentido de que a seed já era fácil de qualquer
jeito. É um terceiro caso: a melhora inicial é real (o sinal sintético forte
ajudou a superar os 14 primeiros pontos difíceis, algo que o sinal real não
conseguiu nessa mesma seed), mas o resultado final de 1.000 superestima a
robustez daquilo, porque o sistema ficou preso numa órbita estreita e fácil
depois do sucesso inicial, não porque o circuito consegue rastrear a bola em
qualquer posição ou velocidade de forma confiável.

## Números recalculados da Fase 1 (com e sem a seed 2)

| | com seed 2 (original) | sem seed 2 |
|---|---|---|
| média real (fim) | 0.198 | 0.230 |
| média sintético forte (fim) | 0.465 | 0.287 |
| diferença | +0.267 | +0.057 |
| teste t pareado | t(3)=1.23, p=0.31 | t(2)=0.72, p=0.55 |

Sem a seed 2, o efeito da Fase 1 fica muito mais fraco (diferença de +0.057
em vez de +0.267) e continua não-significativo. A alegação original de
"desempenho quase perfeito" foi baseada num artefato de dinâmica travada de
uma única seed, não em rastreamento robusto; a correção fica registrada no
Achado 13 do README, com o número original mantido visível ao lado do
recalculado, não apagado.

Isso não invalida a Fase 2, que já era negativa e desfavorável à plasticidade
de forma consistente em 5 de 6 seeds, independente desse ajuste. Mas
enfraquece a força da evidência que motivou prosseguir da Fase 1 pra Fase 2:
o salto pra Fase 2 foi razoável mesmo com o efeito mais fraco (2 de 3 seeds
restantes ainda melhoram com sinal sintético), só não tão contundente quanto
o número original sugeria.
