import tsplib95
import random
import time
import matplotlib.pyplot as plt

# Carrega o problema berlin52 da TSPLIB
problem = tsplib95.load('berlin52.tsp')

# Função para calcular a distância total de um caminho, incluindo o retorno ao ponto inicial
def calculate_distance(route, problem):
    distance = 0
    for i in range(len(route) - 1):
        from_city = route[i]
        to_city = route[i + 1]
        distance += problem.get_weight(from_city, to_city)
    if route[0] != route[-1]:
        distance += problem.get_weight(route[-1], route[0])
    return distance

# Função para criar um indivíduo usando o Algoritmo do Vizinho Mais Próximo
def create_individual_nearest_neighbor(problem):
    nodes = list(problem.get_nodes())
    start = random.choice(nodes)
    individual = [start]
    nodes.remove(start)
    while nodes:
        nearest = min(nodes, key=lambda x: problem.get_weight(individual[-1], x))
        individual.append(nearest)
        nodes.remove(nearest)
    individual.append(start)
    return individual

# Função para criar um indivíduo aleatoriamente
def create_individual_random(problem):
    cities = list(problem.get_nodes())
    random.shuffle(cities)
    start = cities[0]
    cities.append(start)
    return cities

# Função para criar a população inicial
def create_population(problem, population_size):
    population = []
    for _ in range(population_size // 2):
        population.append(create_individual_random(problem))
    for _ in range(population_size // 2, population_size):
        population.append(create_individual_nearest_neighbor(problem))
    return population

# Função para seleção de pais usando torneio
def tournament_selection(population, problem, k=3):
    selected = random.sample(population, k)
    selected.sort(key=lambda x: calculate_distance(x, problem))
    return selected[0], selected[1]

# Função para cruzamento dos pais
def crossover(parent1, parent2):
    size = len(parent1) - 1
    start, end = sorted(random.sample(range(size), 2))
    child = [None] * size
    child[start:end + 1] = parent1[start:end + 1]

    pointer = end + 1
    for gene in parent2:
        if gene not in child:
            if pointer >= size:
                pointer = 0
            child[pointer] = gene
            pointer += 1

    child.append(child[0])
    return child

# Função de mutação
def mutate(individual, mutation_rate=0.01):
    size = len(individual) - 1
    for i in range(size):
        if random.random() < mutation_rate:
            j = random.randint(0, size - 1)
            individual[i], individual[j] = individual[j], individual[i]
    individual[-1] = individual[0]

# Função para aplicar a otimização 2-opt
def two_opt(route, problem):
    best = route[:]
    improved = True
    while improved:
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route) - 1):
                if j - i == 1:
                    continue
                new_route = route[:]
                new_route[i:j] = route[j-1:i-1:-1]
                if calculate_distance(new_route, problem) < calculate_distance(best, problem):
                    best = new_route
                    improved = True
        route = best[:]
    return best

# Função principal do algoritmo genético
def genetic_algorithm(problem, population_size, generations, mutation_rate, elite_size):
    population = create_population(problem, population_size)
    best_individual = min(population, key=lambda x: calculate_distance(x, problem))

    for generation in range(generations):
        new_population = []

        population.sort(key=lambda x: calculate_distance(x, problem))
        elites = population[:elite_size]
        new_population.extend(elites)

        for _ in range((population_size - elite_size) // 2):
            parent1, parent2 = tournament_selection(population, problem)
            child1 = crossover(parent1, parent2)
            child2 = crossover(parent2, parent1)
            mutate(child1, mutation_rate)
            mutate(child2, mutation_rate)
            new_population.append(child1)
            new_population.append(child2)

        population = new_population
        current_best = min(population, key=lambda x: calculate_distance(x, problem))
        if calculate_distance(current_best, problem) < calculate_distance(best_individual, problem):
            best_individual = current_best

    best_individual = two_opt(best_individual, problem)
    return best_individual

# Função para visualizar o caminho
def plot_route(route, problem):
    x_coords = [problem.node_coords[city][0] for city in route] + [problem.node_coords[route[0]][0]]
    y_coords = [problem.node_coords[city][1] for city in route] + [problem.node_coords[route[0]][1]]
    plt.plot(x_coords, y_coords, 'o-')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Melhor Caminho Encontrado')
    
    for city in problem.get_nodes():
        plt.text(problem.node_coords[city][0], problem.node_coords[city][1], city, fontsize=12, ha='right')
    
    plt.grid(True)
    plt.show()

# Executa o algoritmo genético 30 vezes
population_size = 276
generations = 100
mutation_rate = 0.02
elite_size = 50
num_executions = 30

distances = []
execution_times = []

# Variáveis para armazenar a menor distância, o melhor percurso e o menor tempo associado à essa distância
min_distance = float('inf')
best_route = None
executions_with_min_distance = []  # Lista para armazenar todas as execuções com a menor distância

start_total_time = time.time()  # Início do tempo total de execução

for i in range(num_executions):
    start_time = time.time()
    best_solution = genetic_algorithm(problem, population_size, generations, mutation_rate, elite_size)
    best_distance = calculate_distance(best_solution, problem)
    end_time = time.time()
    
    execution_time = end_time - start_time
    distances.append(best_distance)
    execution_times.append(execution_time)

    # Verifica se encontramos uma nova menor distância
    if best_distance < min_distance:
        min_distance = best_distance
        best_route = best_solution  # Armazena o melhor percurso
        executions_with_min_distance = [(best_distance, execution_time)]  # Reinicia a lista
    elif best_distance == min_distance:
        executions_with_min_distance.append((best_distance, execution_time))  # Adiciona execução com a mesma menor distância

    print(f"Execução {i+1}: Melhor distância = {best_distance}, Tempo de execução = {execution_time:.2f} segundos")

end_total_time = time.time()  # Fim do tempo total de execução

# Determina a execução mais rápida entre aquelas que obtiveram a menor distância
fastest_execution_time = min(executions_with_min_distance, key=lambda x: x[1])[1]

# Calcula a distância média
average_distance = sum(distances) / num_executions

# Exibe os resultados
print(f"\nDistância média: {average_distance}")
print(f"Menor distância: {min_distance}")
print(f"Tempo da execução mais rápida que obteve a menor distância: {fastest_execution_time:.2f} segundos")
print(f"Tempo total de execução: {end_total_time - start_total_time:.2f} segundos")

# Exibe o melhor percurso em texto
print(f"\nMelhor percurso encontrado: {best_route}")

# Mostra graficamente o melhor percurso encontrado
plot_route(best_route, problem)