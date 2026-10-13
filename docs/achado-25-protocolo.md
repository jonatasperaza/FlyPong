# Achado 25 — protocolo pré-registrado: controle de conectoma embaralhado

Escrito e commitado **antes** de qualquer execução com o conectoma
embaralhado. Não houve piloto de aprendizado com ele (só um teste de que a
rede monta e roda).

## Pergunta

O aprendizado do Achado 24 depende da ligação específica do conectoma, ou
qualquer rede com a mesma arquitetura em camadas, os mesmos graus e os
mesmos pesos aprenderia igual?

## Controle

`connectome_v09_in21_shuf0`, gerado por
`python scripts/make_shuffled_connectome.py --src connectome_v09_in21 --out connectome_v09_in21_shuf0 --seed 0`:
- dentro de cada bloco papel pré → papel pós (por exemplo, fotorreceptor →
  interneurônio, interneurônio → LC, LC → descendente), os neurônios
  pós-sinápticos das arestas são permutados ao acaso. Cada aresta mantém o
  neurônio pré-sináptico e o peso;
- **preservado:** neurônios, tipos, papéis, posição na retina e lado L/R;
  peso total; força de saída de cada neurônio (idêntica); número de arestas
  de entrada de cada neurônio por bloco; arquitetura em camadas; 498
  sinapses plásticas LC → descendente;
- **destruído:** a identidade dos parceiros. Os fotorreceptores que
  convergem num mesmo neurônio passam a vir de toda a retina (desvio da
  elevação de 547 para 9.720), e 78% desses neurônios recebem dos dois olhos
  (0% no real);
- **efeitos colaterais:** a força de entrada de cada neurônio muda
  (correlação 0,86 com a real), 0,3% das arestas viram pares repetidos
  (somados) e surgem 147 autoconexões.

## Execução

```
python scripts/run_achado19.py --data-dir connectome_v09_in21_shuf0 --retina-axis elevation \
    --homeostasis-scope all_visual --serves 1500 --dev-anneal-serves 100 \
    --conditions normal inverted
```

Tudo idêntico ao Achado 24: hiperparâmetros, recozimento, seeds de treino
42 e 1–5, seeds de teste 1001–1006 com 300 saques, e as três políticas.

## Critérios (definidos agora)

Comparação com o Achado 24 (conectoma real), já executado: treinada somada
de 58,2% [55,9; 60,5] na normal e 58,2% [55,9; 60,5] na invertida.

1. **Principal (normal):** a ligação real importa se a taxa somada da
   treinada real fica acima da do embaralhado, com IC95 de Wilson sem
   sobreposição.
2. **Invertida:** o mesmo teste na condição invertida.
3. **Descritivo:** se o embaralhado aprende de todo (treinada contra os
   próprios controles, por seed, como no Achado 24), e em quantas seeds a
   treinada passa de 40%.

**Interpretação definida agora:**
- Critério 1 cumprido: evidência de que a ligação específica do conectoma
  contribui para o aprendizado.
- Critério 1 não cumprido: não há evidência de que a ligação específica
  importe. O aprendizado do Achado 24 continua real, mas é atribuído à
  arquitetura, aos graus e aos pesos, não à fiação. Isso é registrado como
  resultado, sem buscar outro embaralhamento que dê o resultado contrário.

**Limitação conhecida antes de rodar:** um único embaralhamento (seed 0).
Uma segunda seed de embaralhamento só será rodada para replicar, qualquer
que seja o resultado, e relatada junto.

Qualquer resultado entra no README como Achado 25.
