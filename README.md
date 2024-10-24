# Projeto 1: Algoritmo do Vizinho Mais Próximo
Este projeto implementa o algoritmo do Vizinho Mais Próximo para resolver o Problema do Caixeiro Viajante (TSP). Ele busca encontrar o caminho mais curto para visitar todas as cidades de uma vez, começando e terminando na mesma cidade.

Como Funciona
Carregamento do Problema: O problema é carregado a partir de um arquivo .tsp (como berlin52.tsp) usando a biblioteca tsplib95.

Algoritmo do Vizinho Mais Próximo: O algoritmo começa em uma cidade inicial e, em cada passo, escolhe a cidade mais próxima que ainda não foi visitada. Este processo se repete até que todas as cidades sejam visitadas, retornando à cidade inicial no final.

Cálculo de Distâncias: As distâncias totais do percurso são calculadas usando a matriz de distâncias fornecida pelo problema.

Execução e Resultados: O código exibe a cidade inicial escolhida, o percurso encontrado e a distância total do percurso, além do tempo de execução.

Dependências
tsplib95: Para carregar e manipular os dados do problema.
time: Para medir o tempo de execução.
## Como Executar
  1. Instale as dependências com ``` pip install tsplib95. ```

  2. Execute o script e forneça o nome de um arquivo ``` .tsp ``` ao ser solicitado

```
 python vizinho_mais_proximo_final.py
```

## Exemplo de Saída

```Melhor cidade inicial: 1 
Melhor percurso: 1 -> 3 -> 5 -> ... -> 1
Distância total: 7542
Tempo decorrido: 0.4321 segundos´´´



