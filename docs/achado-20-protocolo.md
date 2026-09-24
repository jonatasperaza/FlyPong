# Achado 20 — protocolo pré-registrado

Escrito e commitado **antes** de qualquer execução com os 28 descendentes.

## Pergunta

O ganho modesto do Achado 19 no dado real (36,9% contra 26,3% na condição
invertida) é limitado pelo canal de saída de só 4 descendentes? Com 28
descendentes (14 por lado), a mesma regra aprende mais?

## Dados

- `male-cns:v0.9`, porque o servidor deixou de oferecer a v1.0. Neste
  subgrafo as versões são quase idênticas: 44.989 neurônios nas duas,
  1.157.841 contra 1.157.952 arestas.
- **A (controle da versão):** `fetch_connectome.py --dataset male-cns:v0.9
  --dn-set original`, com 4 descendentes.
- **B (experimento):** `--dn-set achado20`, com 28 descendentes. São os 12
  tipos com par bilateral e ≥ 1000 de peso vindo dos LC do subgrafo
  (`docs/dn_candidatos_male-cns-v0.9.txt`), mais GF e DNa10.

## Configuração (idêntica em A e B, sem nenhum ajuste novo)

```
python scripts/run_achado19.py --data-dir <A ou B> --retina-axis elevation \
    --homeostasis-scope visual --serves 1500
```

- Hiperparâmetros: eta = 0,001, ruído 6, 300 saques de desenvolvimento e
  1.500 de treino. São os do Achado 19, escolhidos só nas seeds de
  validação.
- Seeds de treino: 42 e 1–5. Seeds de teste: 1001–1006, com 300 saques
  cada.
- Políticas: treinada, controle homeostase e controle inato.
- Condições: normal e invertida.

## Critérios (definidos agora)

1. **Principal:** em B, na condição invertida, o IC95 da treinada fica acima
   do IC95 de cada controle em ≥ 5 de 6 seeds.
2. **Efeito do canal de saída:** a taxa de acerto somada da treinada em B
   fica acima da de A, com IC95 de Wilson (1.800 saques somados) sem
   sobreposição, na condição invertida.
3. **Secundário:** os critérios 1 e 2 também na condição normal.

Qualquer resultado entra no README como Achado 20, inclusive se for
negativo. Nenhum hiperparâmetro muda depois de ver as seeds de teste.
