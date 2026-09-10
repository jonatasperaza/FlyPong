# Guia prático: controlando algo com um conectoma real

Checklist extraído da experiência real de construir o FlyPong (Pong
controlado por um subgrafo do MaleCNS v1.0 via neuPrint). Não é um resumo do
projeto, é organizado por etapa do trabalho, pra usar como checklist ativo
enquanto você constrói o seu. Cada item linka pro achado correspondente em
[`README.md`](../README.md) pra quem quiser o detalhe completo (números,
código, auditoria bruta).

## 1. Antes de escrever qualquer código: como escolher os tipos celulares

Não escolha tipos celulares pelo nome que soa certo na literatura. `LC4` e
`LPLC2` são classicamente "detectores de objeto", mas o circuito real que
você vai simular pode não ter a aresta que você presume que existe entre eles
e o seu alvo motor. Audite a conectividade real antes de decidir o subgrafo,
não depois: puxe todos os parceiros pré/pós-sinápticos reais de um neurônio
candidato (`fetch_adjacencies` com o outro lado irrestrito), sem filtrar por
tipo, e veja o que aparece de fato. Repita isso pros seus candidatos
motores/de saída também, não só pros sensoriais. Ver Achados 6, 8 e 9: a
mudança de `DNa02` (desconectado) pra `DNa10` (conectado via `LC10a`) só foi
possível porque a lista completa de parceiros foi auditada em vez de
assumida.

## 2. Armadilhas de query no neuPrint

O `neuprint-python` traduz `regex=True` pro operador `=~` do Cypher/Neo4j,
que exige fullmatch: a string inteira precisa bater com o padrão, não é
busca de substring. Um padrão como `"T4"` só bate com a string exata `"T4"`,
nunca com `"T4a"`. Use `.*` no fim de padrões que precisam cobrir subtipos,
`"T4.*"`, não `"T4"` ou `"^T4"`. Confira sempre a contagem de neurônios
retornados antes de seguir: se um tipo que você sabe que existe volta com 0,
é quase sempre esse bug, não ausência real do tipo. Ver Achado 1.

## 3. Armadilhas de anatomia: camadas intermediárias escondidas

A maioria dos circuitos sensoriais reais tem camadas intermediárias que não
aparecem óbvias numa primeira leitura da literatura. Fotorreceptores, por
exemplo, não fazem sinapse direta em detectores de movimento de alto nível;
o sinal passa por lâmina e medula antes. Depois de escolher seu subgrafo,
audite arestas por par de papéis (`groupby(['pre_role','post_role'])` nas
arestas baixadas) antes de assumir que a cadeia está completa. Se um par de
papéis adjacentes na sua cadeia lógica tem zero arestas reais entre si, sua
simulação vai propagar sinal só até aquele ponto e nunca mais. Ver Achado 2
(fotorreceptor→T4/T5 exigiu adicionar L1-L5/Mi1/Mi4/Mi9/Tm1/Tm2/Tm4/Tm9 no
meio).

## 4. Dados espaciais que não existem prontos

Se seu projeto precisa de posição espacial de neurônios (retinotopia,
somatotopia, etc.), não assuma que `somaLocation` está preenchido. Nesse
dataset, só 11 de 1783 fotorreceptores tinham soma segmentado (0.6%). A
alternativa que funcionou foi o centróide das sinapses de saída de cada
neurônio (`fetch_synapses`, quase sempre presente), projetado no eixo de
maior variância via PCA (SVD nas coordenadas centralizadas). Isso não
garante que o eixo encontrado é o eixo anatômico que você espera
(dorso-ventral, por exemplo), só que existe algum gradiente espacial
coerente. Confira isso explicitamente (ex.: testando se a leitura motora
resultante tem direção consistente com a posição do estímulo) antes de
confiar nele pra qualquer coisa que dependa de direção. Ver Achado 3.

## 5. Bugs de escala entre dados sintéticos e reais

Se você usa um gerador de dados sintético pra testar o pipeline offline (sem
token, sem acesso à API), qualquer constante calibrada nesses dados
sintéticos (limiares de clipping, ganhos sinápticos, tetos de peso) pode
quebrar silenciosamente com dados reais, cuja escala pode ser ordens de
grandeza diferente. Um exemplo concreto: um `clip()` de peso plástico com
limites absolutos (ex. `[-1, 5]`) colapsa o peso real na primeira atualização
se o peso real de base já for maior que o teto, mesmo com a taxa de
aprendizado zerada, invalidando qualquer comparação com um controle "sem
plasticidade". Prefira clipar o delta em torno do valor original de cada
sinapse, não o valor absoluto. Teste sempre seus limites e clips contra a
escala real de peso antes de rodar experimentos longos; um teste barato de
poucos frames já expõe isso. Ver Achado 5.

## 6. Antes de acusar a plasticidade de não funcionar

Esta é provavelmente a lição mais cara e mais generalizável do projeto
inteiro. Se duas condições de peso idênticas (plástico ligado vs. desligado)
produzem resultados idênticos, não presuma que é um bug de plasticidade:
audite individualmente quantas sinapses relevantes cada neurônio de saída
específico recebe. No FlyPong, 2 dos 4 neurônios motores selecionados
recebiam zero sinapses de qualquer via sensorial selecionada, não porque a
plasticidade estava quebrada, mas porque metade do readout motor estava
anatomicamente desconectada do resto do circuito. Nenhuma quantidade de
ajuste de taxa de aprendizado resolveria isso. A auditoria (quantas sinapses
de tipo X chegam no neurônio Y, individualmente) custa poucos minutos e
evita perseguir o bug errado por horas. Ver Achado 6.

## 7. Acoplamento acidental de RNG entre mecanismos independentes

Sempre que você adiciona um novo sorteio estocástico a um sistema que já tem
outro gerador de números aleatórios rodando (física do jogo, ruído
sensorial, amostragem de estímulo), use um `random.Random` separado, não
reaproveite o mesmo stream. Reusar o mesmo RNG pra dois mecanismos
independentes acopla os dois: mudar um (ex. ligar um novo modo de reforço)
muda o outro (ex. a sequência de física da bola) mesmo sem nenhuma relação
causal real entre eles. O sintoma que expôs esse bug no FlyPong foi que a
condição controle (sem plasticidade possível) mostrava "melhora" em 100% das
seeds testadas, estatisticamente implausível pra uma condição sem
aprendizado nenhum. Teste explicitamente que o novo mecanismo desligado
reproduz exatamente os números antigos antes de confiar em qualquer
comparação. Ver Achado 11.

## 8. Rigor estatístico com poucas seeds

Com poucas seeds (4 a 6, o que é tipicamente tudo que cabe no orçamento de
tempo de um projeto assim), a direção do efeito sozinha não significa nada;
metade das vezes vai favorecer uma condição por puro acaso. Reporte sempre o
p-valor de um teste de sinal e/ou teste t pareado, mesmo quando o resultado
"parece" positivo à primeira vista (ex. "4 de 6 seeds favorecem a condição
X" soa convincente até você calcular que isso dá p≈0.69). Isso evitou pelo
menos duas conclusões prematuras neste projeto (ver Achados 10, 11 e 12),
onde padrões que pareciam interessantes num primeiro olhar não sobreviveram
ao teste estatístico formal.

## 9. Quando parar

"Decidir não continuar" é uma conclusão válida quando documentada
explicitamente, não é desistência. Quando o próximo experimento custaria
muito mais tempo de computação do que o valor esperado de informação que ele
traria (ex. um recozimento de 4 níveis × N seeds em cima de um sinal já
estatisticamente indistinguível de ruído no ponto mais favorável), é
legítimo parar e reportar isso como a decisão que foi, com o motivo. O que
não é legítimo é parar silenciosamente e deixar parecer que a investigação
terminou por outro motivo. Ver Achado 12.

## Resumo rápido (se você só tem 2 minutos)

1. Audite conectividade real antes de escolher tipos celulares (§1, §6).
2. Confira contagens de neurônios retornadas: regex do neuPrint é fullmatch
   (§2).
3. Audite arestas por par de papéis antes de assumir que a cadeia sensorial
   está completa (§3).
4. `somaLocation` provavelmente não está preenchido; use centróide de
   sinapse + PCA, e confira a direção resultante (§4).
5. Teste clips/limites contra a escala real de peso antes de rodar
   experimentos longos (§5).
6. RNGs de mecanismos independentes ficam em streams separados (§7).
7. Reporte p-valor, não só direção do efeito (§8).
8. Parar com justificativa documentada é um resultado válido (§9).
