# Achado 24 — protocolo pré-registrado

Escrito e commitado **antes** de qualquer execução nas seeds de teste com
esta configuração. A mudança foi escolhida só com a seed de validação 2001
e as seeds de treino 1, 3, 5 e 42 (ver README, Achado 24, parte de
diagnóstico). Nenhum hiperparâmetro foi alterado depois.

## Pergunta

O recozimento da homeostase no fim da fase de desenvolvimento torna o
aprendizado confiável entre seeds, no conectoma real?

## Diagnóstico que motivou a mudança (só validação)

- Com a homeostase a taxa fixa, a rede alterna entre saques calmos e saques
  em explosão (cerca de 1/3 dos frames em explosão, em todas as seeds). O
  estado em que ela sai do desenvolvimento depende dos últimos saques.
- Na rede congelada, a fração de explosão ordena as seeds pelo aprendizado:
  seed 1 com 0,00 aprende (83%); seeds 5, 42 e 3, com 0,44–0,72, não
  aprendem (20–28%).
- Na explosão, quase todos os LC preferem o mesmo lado da bola. Com pesos
  positivos, um dos lados do readout fica sem sinal.
- Recozimento de 100 saques (a taxa das duas homeostases cai linearmente até
  zero): explosão de 0,10–0,25 e LC dos dois lados nas 4 seeds. Treino na
  validação: 64,7%, 47,7%, 57,3% e 47,0% (4/4 acima do ponto de partida, de
  20–23%).

## Configuração

```
python scripts/run_achado19.py --data-dir connectome_v09_in21 --retina-axis elevation \
    --homeostasis-scope all_visual --serves 1500 --dev-anneal-serves 100 \
    --conditions normal inverted
```

- A mesma configuração A do Achado 22 (male-cns:v0.9, 4 descendentes,
  input_set achado21, homeostase all_visual), com `dev_anneal_serves=100`.
- Hiperparâmetros inalterados: eta 0,001, ruído 6, 300 saques de
  desenvolvimento (os últimos 100 em recozimento), 1.500 de treino, taxa
  alvo 0,3 e taxa homeostática 0,01.
- Seeds de treino: 42 e 1–5. Seeds de teste: 1001–1006, com 300 saques cada.
- Políticas: treinada, controle homeostase (o mesmo desenvolvimento, com
  recozimento, sem treino) e controle inato.

## Critérios (definidos agora)

1. **Principal (normal):** o IC95 da treinada fica acima do IC95 de cada
   controle em ≥ 5 de 6 seeds.
2. **Invertida:** o mesmo critério em ≥ 5 de 6 seeds.
3. **Ganho sobre o Achado 22 (A, sem recozimento):** em cada condição, a taxa
   somada da treinada (1.800 saques) fica acima da do Achado 22 (normal
   37,3% [35,1; 39,5]; invertida 44,9% [42,6; 47,2]), com IC95 de Wilson
   sem sobreposição.
4. **Confiabilidade (descritivo):** número de seeds em que a treinada passa
   de 40%, nas duas condições, contra o Achado 22 (normal 1/6, invertida
   1/6).

Qualquer resultado entra no README como Achado 24, inclusive se for
negativo.
