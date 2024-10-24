import tsplib95
import time

def metodo_vizinho_mais_proximo(matriz_distancias, cidade_inicial):
    num_cidades = len(matriz_distancias)
    visitado = [False] * num_cidades
    percurso = []
    
    cidade_atual = cidade_inicial
    percurso.append(cidade_atual)
    visitado[cidade_atual] = True
    
    for _ in range(num_cidades - 1):
        proxima_cidade = None
        menor_distancia = float('inf')
        
        for i in range(num_cidades):
            if not visitado[i] and matriz_distancias[cidade_atual][i] < menor_distancia:
                menor_distancia = matriz_distancias[cidade_atual][i]
                proxima_cidade = i
        
        percurso.append(proxima_cidade)
        visitado[proxima_cidade] = True
        cidade_atual = proxima_cidade
    
    percurso.append(percurso[0])
    return percurso

def calcular_distancia_total(percurso, matriz_distancias):
    distancia_total = 0
    for i in range(len(percurso) - 1):
        distancia_total += matriz_distancias[percurso[i]][percurso[i + 1]]
    return distancia_total

def encontrar_melhor_cidade_inicial(problema):
    nodes = list(problema.get_nodes())
    # Ajuste da indexação
    matriz_distancias = [[problema.get_weight(i, j) for j in nodes] for i in nodes]
    
    melhor_percurso = None
    menor_distancia = float('inf')
    melhor_cidade_inicial = None

    for cidade_inicial in range(len(matriz_distancias)):
        percurso = metodo_vizinho_mais_proximo(matriz_distancias, cidade_inicial)
        distancia_total = calcular_distancia_total(percurso, matriz_distancias)
        
        if distancia_total < menor_distancia:
            menor_distancia = distancia_total
            melhor_percurso = percurso
            melhor_cidade_inicial = cidade_inicial

    return melhor_cidade_inicial, melhor_percurso, menor_distancia

if __name__ == "__main__":
    nome_arquivo = input("Digite o nome do arquivo TSP: ")
    problema = tsplib95.load(nome_arquivo)

    # Medir o tempo de execução
    inicio_tempo = time.time()

    melhor_cidade_inicial, melhor_percurso, menor_distancia = encontrar_melhor_cidade_inicial(problema)

    # Calcular o tempo total de execução
    tempo_total = time.time() - inicio_tempo

    print(f"Melhor cidade inicial: {melhor_cidade_inicial + 1}")
    print("Melhor percurso:", ' -> '.join(str(cidade + 1) for cidade in melhor_percurso))
    print(f"Distância total: {menor_distancia}")
    print(f"Tempo decorrido: {tempo_total:.4f} segundos")