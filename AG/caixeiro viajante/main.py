import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

# Variáveis default (valores padrão)
POPULATION_SIZE = 10  # Tamanho da população inicial
MUTATION_RATE = 0.05  # Taxa de mutação
CROSSOVER_RATE = 0.85  # Taxa de crossover
GENERATION_NUMBER = 100  # Número de gerações
ELITISM_LEN = 1  # Número de indivíduos elitistas
IS_TOURNAMENT = True  # Tipo de seleção (True para torneio, False para roleta)
TOURNAMENT_SIZE = 5  # Tamanho do torneio
cidades_excel = []  # Lista de cidades (inicialmente vazia)
CROSSOVER_TYPE = 'ox'  # Tipo de crossover ('pmx', 'ox' ou 'cx')

# Função para criar um indivíduo (uma possível solução)
def create_individual():
    df = pd.read_excel('distancias.xlsx')  # Lê o arquivo Excel com as distâncias entre cidades
    distancias = df.values[:, 1:].astype(float)  # Converte as distâncias para um array NumPy de floats, ignorando a primeira coluna (nomes)
    distancias_np = np.array(distancias)
    np.random.shuffle(distancias_np)  # Embaralha as distâncias para criar um indivíduo único
    return deepcopy(distancias_np)  # Retorna uma cópia profunda do array embaralhado para evitar alterações acidentais

# Função para criar uma população de indivíduos
def create_population(population_size):
    return [create_individual() for _ in range(population_size)]  # Cria uma lista de indivíduos (população) do tamanho especificado

# Função de avaliação (fitness) que calcula o "valor" de um indivíduo
def fitness(individual, print_params=False):
    km_percorrido = 0
    for i in range(len(individual) - 1):
        index_of = np.where(individual[i + 1] == 0)[0][0]  # Encontra o índice do próximo cidade na matriz de distâncias
        km_percorrido += individual[i][index_of]  # Soma a distância percorrida até o próximo cidade
    return 4000 - km_percorrido  # Retorna o fitness (quanto menor a distância, maior o fitness)

# Função de seleção por roleta
def select_roulette(population, fitnesses, num_parents):
    total_fitness = sum(fitnesses)  # Soma total dos valores de fitness
    probs = [f / total_fitness for f in fitnesses]  # Probabilidades de seleção baseadas no fitness
    parents = []
    for _ in range(num_parents):
        r = random.random()  # Número aleatório entre 0 e 1
        for i, individual in enumerate(population):
            r -= probs[i]
            if r <= 0:
                parents.append(deepcopy(individual))  # Seleciona o indivíduo se o valor aleatório for menor ou igual à probabilidade acumulada
                break
    return parents

# Função de seleção por torneio
def select_tournament(population, fitnesses, num_parents, tournament_size=TOURNAMENT_SIZE):
    parents = []
    for _ in range(num_parents):
        contenders = random.sample(list(zip(population, fitnesses)), tournament_size)  # Seleciona aleatoriamente um conjunto de indivíduos (contenders)
        winner = max(contenders, key=lambda x: x[1])[0]  # Escolhe o indivíduo com o maior fitness entre os contenders
        parents.append(deepcopy(winner))  # Adiciona o vencedor à lista de pais
    return parents

# Função de crossover PMX
def pmx_crossover(parent1, parent2, crossover_rate):
    should_cross = random.random()  # Decide se o crossover deve acontecer
    if should_cross < crossover_rate:
        # Realiza crossover PMX (Partially Mapped Crossover)
        child1 = np.concatenate((parent1[0: 7], parent2[7: 14], parent1[14: 20]))
        child2 = np.concatenate((parent2[0: 7], parent1[7: 14], parent2[14: 20]))
        remove_repeated(child1)  # Remove genes repetidos para manter a validade do indivíduo
        remove_repeated(child2)
        return deepcopy(child1), deepcopy(child2)
    else:
        return parent1, parent2

# Função de crossover CX
def cx_crossover(parent1, parent2, crossover_rate):
    should_cross = random.random()  # Decide se o crossover deve acontecer
    if should_cross < crossover_rate:
        cycles = [-1] * len(parent1)
        cycle_no = 1
        cyclestart = (i for i, v in enumerate(cycles) if v < 0)
        for pos in cyclestart:
            while cycles[pos] < 0:
                cycles[pos] = cycle_no
                mask = np.all(parent1 == parent2[pos], axis=1)
                pos = np.where(mask)[0][0]
            cycle_no += 1
        child1 = np.array([parent1[i] if n % 2 else parent2[i] for i, n in enumerate(cycles)])
        child2 = np.array([parent2[i] if n % 2 else parent1[i] for i, n in enumerate(cycles)])
        return deepcopy(child1), deepcopy(child2)
    else:
        return parent1, parent2

# Função de crossover OX
def ox_crossover(parent1, parent2, crossover_rate):
    should_cross = random.random()  # Decide se o crossover deve acontecer
    if should_cross < crossover_rate:
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))  # Seleciona dois pontos para o crossover
        child1, child2 = np.array([[-1.0] * size] * size), np.array([[-1.0] * size] * size)  # Inicializa os filhos com valores inválidos (-1)
        child1[start:end] = parent1[start:end]  # Copia a subsequência do pai1 para o filho1
        child2[start:end] = parent2[start:end]  # Copia a subsequência do pai2 para o filho2
        for p1, p2 in zip(np.concatenate((parent1[end:], parent1[:end]), axis=0),
                          np.concatenate((parent2[end:], parent2[:end]), axis=0)):
            mask1 = np.all(child1 == p1, axis=1)
            if len(np.where(mask1)[0]) == 0:
                child1[np.where(child1 == -1)[0][0]] = p1
            mask2 = np.all(child2 == p2, axis=1)
            if len(np.where(mask2)[0]) == 0:
                child2[np.where(child2 == -1)[0][0]] = p2
        return child1, child2
    else:
        return parent1, parent2

# Função para remover genes repetidos em um indivíduo após o crossover
def remove_repeated(child):
    abc = []
    contador = {i: 0 for i in range(20)}  # Inicializa um contador para contar a ocorrência de cada gene
    for array in child:
        abc.append(np.where(array == 0)[0][0])
        contador[np.where(array == 0)[0][0]] += 1  # Conta a ocorrência de cada gene
    for i in abc:
        if contador[i] > 1:  # Se um gene aparece mais de uma vez
            indice_escolhido = zero_position(contador)  # Encontra um gene que não aparece
            contador[i] -= 1
            child[i] = distancias[indice_escolhido]  # Substitui o gene repetido por um gene válido
            contador[indice_escolhido] += 1
    return child

# Função auxiliar para encontrar a posição zero no contador
def zero_position(contador):
    for k, v in contador.items():
        if v == 0:  # Retorna a posição de um gene que não aparece no indivíduo
            return k

# Função para mutação de um indivíduo
def mutate(individual, mutation_rate):
    individual_copy = deepcopy(individual)  # Faz uma cópia profunda do indivíduo para evitar alterações acidentais
    for _ in range(20):  # Itera sobre cada gene no indivíduo
        should_mutate = random.random()  # Decide se a mutação deve acontecer
        if should_mutate < mutation_rate:
            j_chosen1 = random.randint(0, 19)  # Seleciona aleatoriamente dois genes para troca
            j_chosen2 = random.randint(0, 19)
            individual_copy[[j_chosen1, j_chosen2]] = individual_copy[[j_chosen2, j_chosen1]]  # Troca os genes
    return deepcopy(individual_copy)  # Retorna o indivíduo mutado

# Função para substituir a população antiga por novos indivíduos
def replace_population(population, new_individuals):
    population.sort(key=fitness)  # Ordena a população pelo fitness
    population[:len(new_individuals)] = deepcopy(new_individuals)  # Substitui os piores indivíduos pelos novos
    return deepcopy(population)  # Retorna a nova população

# Função para calcular a distância total percorrida a partir do fitness
def calculate_distance(fitness):
    return (-1 * fitness + 4000)  # Converte o fitness de volta para a distância total percorrida

# Função principal do algoritmo genético
def genetic_algorithm(population_size, num_generations, tournament, tournament_size, crossover_value, elitism, mutation, crossover_type):
    population = create_population(population_size)  # Cria a população inicial
    df = pd.read_excel('distancias.xlsx')  # Lê o arquivo Excel com as distâncias entre cidades
    global distancias
    distancias = df.values[:, 1:].astype(float)  # Converte as distâncias para um array NumPy de floats, ignorando a primeira coluna (nomes)
    distancias = np.array(distancias)  # Converte a matriz de distâncias para um array NumPy
    best_gen = 0  # Inicializa a geração da melhor solução
    best_fitness = 0  # Inicializa o melhor fitness

    # Listas para armazenar estatísticas de cada geração
    x = []
    best_array = []
    avg_array = []
    worst_array = []

    for gen in range(num_generations):  # Itera sobre cada geração
        fitnesses = [fitness(individual) for individual in population]  # Calcula o fitness de cada indivíduo na população
        if tournament:
            parents = select_tournament(population, fitnesses, population_size // 2, tournament_size)  # Seleciona os pais usando seleção por torneio
        else:
            parents = select_roulette(population, fitnesses, population_size // 2)  # Seleciona os pais usando seleção por roleta
        
        children = []
        for i in range(0, len(parents) - 1, 2):
            if crossover_type == 'pmx':
                child1, child2 = pmx_crossover(parents[i], parents[i + 1], crossover_value)  # Realiza o crossover PMX
            elif crossover_type == 'ox':
                child1, child2 = ox_crossover(parents[i], parents[i + 1], crossover_value)  # Realiza o crossover OX
            else:
                child1, child2 = cx_crossover(parents[i], parents[i + 1], crossover_value)  # Realiza o crossover CX
            children.append(child1)
            children.append(child2)
        
        if elitism > 0:
            population.sort(key=fitness, reverse=True)  # Ordena a população pelo fitness em ordem decrescente
            elites = deepcopy(population[:elitism])  # Seleciona os melhores indivíduos para elitismo
        
        mutated_children = [mutate(child, mutation) for child in children]  # Realiza a mutação nos filhos
        population = replace_population(population, mutated_children)  # Substitui a população antiga pelos novos indivíduos
        
        if elitism > 0:
            population.sort(key=fitness)  # Ordena a população pelo fitness
            population[:elitism] = deepcopy(elites)  # Substitui os piores indivíduos pelos elitistas
        
        statistic_fitness = [fitness(individual) for individual in population]  # Calcula o fitness de cada indivíduo na população
        if elitism > 0:
            fitness(elites[0], True)  # Imprime os parâmetros do melhor indivíduo elitista
        
        if max(statistic_fitness) > best_fitness:  # Atualiza o melhor fitness e a geração da melhor solução
            best_gen = gen
            best_fitness = max(statistic_fitness)
        
        best_array.append(max(statistic_fitness))  # Adiciona o melhor fitness da geração à lista
        avg_array.append(sum(statistic_fitness) / len(statistic_fitness))  # Adiciona o fitness médio da geração à lista
        worst_array.append(min(statistic_fitness))  # Adiciona o pior fitness da geração à lista
        x.append(gen)  # Adiciona a geração atual à lista
        
        # Imprime as estatísticas da geração atual
        print('Geração: ', gen + 1, 'Fitness: ', max(statistic_fitness), 'Melhor Geração: ', best_gen + 1,
              'Distância total: ', calculate_distance(max(statistic_fitness)), 'km')

    # Plota a evolução do fitness ao longo das gerações
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(x, best_array, label='Melhor', linewidth=3)
    ax1.plot(x, avg_array, label='Média', linestyle='dashed', linewidth=3)
    ax1.plot(x, worst_array, label='Pior', linestyle='dotted', linewidth=3)
    ax1.set_xlabel("Gerações")
    ax1.set_ylabel("Fitness")
    ax1.legend(loc='upper left')
    ax1.set_title("Evolução do Fitness")
    plt.grid(True)
    plt.show()

# Função principal para configurar e iniciar o algoritmo genético
def main():
    global POPULATION_SIZE, MUTATION_RATE, CROSSOVER_RATE, GENERATION_NUMBER, ELITISM_LEN, IS_TOURNAMENT, TOURNAMENT_SIZE, CROSSOVER_TYPE

    while True:
        # Coleta de parâmetros do usuário
        print("Configuração do Algoritmo Genético:")
        POPULATION_SIZE = int(input("Tamanho da população (padrão 10): ") or POPULATION_SIZE)
        MUTATION_RATE = float(input("Taxa de mutação (padrão 0.05): ") or MUTATION_RATE)
        CROSSOVER_RATE = float(input("Taxa de crossover (padrão 0.85): ") or CROSSOVER_RATE)
        GENERATION_NUMBER = int(input("Número de gerações (padrão 100): ") or GENERATION_NUMBER)
        ELITISM_LEN = int(input("Quantidade de elitismo (padrão 1): ") or ELITISM_LEN)
        IS_TOURNAMENT = input("Usar seleção por torneio? (padrão sim): ").strip().lower() in ['sim', 's', '']
        if IS_TOURNAMENT:
            TOURNAMENT_SIZE = int(input("Tamanho do torneio (padrão 5): ") or TOURNAMENT_SIZE)
        CROSSOVER_TYPE = input("Tipo de crossover (pmx/ox/cx, padrão ox): ").strip().lower() or CROSSOVER_TYPE

        # Exibe a configuração do algoritmo genético
        print("\nIniciando o algoritmo genético com as seguintes configurações:")
        print(f"Tamanho da população: {POPULATION_SIZE}")
        print(f"Taxa de mutação: {MUTATION_RATE}")
        print(f"Taxa de crossover: {CROSSOVER_RATE}")
        print(f"Número de gerações: {GENERATION_NUMBER}")
        print(f"Elitismo: {ELITISM_LEN}")
        print(f"Seleção por torneio: {'Sim' if IS_TOURNAMENT else 'Não'}")
        if IS_TOURNAMENT:
            print(f"Tamanho do torneio: {TOURNAMENT_SIZE}")
        print(f"Tipo de crossover: {CROSSOVER_TYPE}\n")

        # Inicia o algoritmo genético com os parâmetros fornecidos
        genetic_algorithm(POPULATION_SIZE, GENERATION_NUMBER, IS_TOURNAMENT, TOURNAMENT_SIZE, CROSSOVER_RATE, ELITISM_LEN, MUTATION_RATE, CROSSOVER_TYPE)

        # Pergunta ao usuário se deseja executar novamente com outras configurações
        continuar = input("Deseja executar novamente com outras configurações? (sim/não): ").strip().lower()
        if continuar not in ['sim', 's']:
            break

# Executa a função principal se o script for executado diretamente
if __name__ == "__main__":
    main()

