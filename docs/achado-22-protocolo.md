# Achado 22 — protocolo pré-registrado

Escrito e commitado **antes** de qualquer execução nas seeds de teste com
esta configuração. O único uso anterior dela foi um piloto na seed de
validação 2001, com treinada e controle nas duas condições. Os
hiperparâmetros não foram alterados depois do piloto.

## Pergunta

Com a saturação do sistema visual removida pela homeostase em todas as
camadas visuais (`--homeostasis-scope all_visual`) e a entrada dos LC
completada (`--input-set achado21`), a regra aprende nas **duas** condições,
inclusive na normal?

## Configurações

- **A (principal):** `connectome_v09_in21`, com 4 descendentes e input_set
  achado21, em `male-cns:v0.9`.
- **B (controle da ampliação):** `connectome_v09_4dn`, com os 4 descendentes
  e o input_set original. Ela separa o efeito da homeostase do efeito dos
  tipos novos.

As duas rodam com:

```
python scripts/run_achado19.py --data-dir <A ou B> --retina-axis elevation \
    --homeostasis-scope all_visual --serves 1500
```

- Hiperparâmetros inalterados: eta 0,001, ruído 6, 300 saques de
  desenvolvimento, 1.500 de treino, taxa alvo 0,3 e taxa homeostática 0,01.
- Seeds de treino: 42 e 1–5. Seeds de teste: 1001–1006, com 300 saques cada.
- Políticas: treinada, controle homeostase e controle inato.

## Critérios (definidos agora)

1. **Principal:** em A, na condição **normal**, o IC95 da treinada fica acima
   do IC95 de cada controle em ≥ 5 de 6 seeds.
2. **Invertida:** o mesmo critério em A, em ≥ 5 de 6 seeds.
3. **Ganho sobre o Achado 20:** em A, em cada condição, a taxa somada da
   treinada (1.800 saques) fica acima da de 4 descendentes com homeostase só
   nos LC (Achado 20: 36,6% na invertida e 30,7% na normal), com IC95 de
   Wilson sem sobreposição.
4. **Secundário (decomposição):** comparação somada entre A e B em cada
   condição, com o mesmo teste de IC.

Qualquer resultado entra no README como Achado 22, inclusive se for
negativo.
