import tkinter as tk
from tkinter import ttk
import random
import matplotlib.pyplot as plt
from copy import deepcopy

DAYS = 5  # Número de dias da semana (segunda a sexta)
PERIODS = 12  # Número de períodos por dia (manhã e tarde)
CLASS = 2  # Número de turmas (cromossomo)
SUBJECTS = 15  # Número total de disciplinas

# Definição dos nomes das disciplinas e professores
subject_names = {
    -1: "JANELA",  # Slot vazio, usado para períodos livres
    1: "Algoritmos",
    2: "PI1",
    3: "GA",
    4: "EE",
    5: "IE",
    6: "CAL1",
    7: "ILM",
    8: "FIS1",
    9: "ED",
    10: "CIR1",
    11: "CAL2",
    12: "PROB",
    13: "ALGE",
    14: "PI2",
    15: "QT",
}

# Definição das disciplinas e suas cargas horárias por turma
subjects_by_class = {
    1: [
        {"Id": 1, "Professor": "Ernani Borges", "Nome": "Algoritmos", "Aulas": 7},
        {"Id": 2, "Professor": "Bruno", "Nome": "PI1", "Aulas": 1},
        {"Id": 3, "Professor": "Jorge", "Nome": "GA", "Aulas": 4},
        {"Id": 4, "Professor": "Aline", "Nome": "EE", "Aulas": 3},
        {"Id": 5, "Professor": "Johann", "Nome": "IE", "Aulas": 2},
        {"Id": 6, "Professor": "Leandro", "Nome": "CAL1", "Aulas": 5},
        {"Id": 7, "Professor": "Thomaz", "Nome": "ILM", "Aulas": 2},
    ],
    2: [
        {"Id": 8, "Professor": "Anderson", "Nome": "FIS1", "Aulas": 4},
        {"Id": 9, "Professor": "Daniela", "Nome": "ED", "Aulas": 5},
        {"Id": 10, "Professor": "Andreia", "Nome": "CIR1", "Aulas": 4},
        {"Id": 11, "Professor": "Leandro", "Nome": "CAL2", "Aulas": 4},
        {"Id": 12, "Professor": "Jorge", "Nome": "PROB", "Aulas": 4},
        {"Id": 13, "Professor": "Jorge", "Nome": "ALGE", "Aulas": 4},
        {"Id": 14, "Professor": "Gustavo", "Nome": "PI2", "Aulas": 1},
        {"Id": 15, "Professor": "Marcia", "Nome": "QT", "Aulas": 2},
    ],
}

# Função para traduzir o cronograma de IDs para nomes das disciplinas
def translate_schedule(schedule):
    translated_schedule = []
    for x in range(CLASS):  # Iterar sobre cada turma
        translated_schedule.append([])
        for y in range(PERIODS * DAYS):  # Iterar sobre cada período de cada dia
            translated_schedule[x].append(schedule[x][y]["Nome"])  # Adicionar o nome da disciplina no cronograma traduzido
    return translated_schedule

# Função para criar um indivíduo (um possível cronograma)
def create_individual():
    individual = []

    for i in range(CLASS):
        group = [None] * (PERIODS * DAYS)  # Inicializar a matriz do cronograma com None
        if i == 0:  # Para a turma 1
            # Slots disponíveis são os períodos da tarde (períodos 6 a 11)
            available_slots = [x for x in range(PERIODS * DAYS) if (x % PERIODS) >= 6]
        else:  # Para a turma 2
            # Slots disponíveis são os períodos da manhã (períodos 0 a 5)
            available_slots = [x for x in range(PERIODS * DAYS) if (x % PERIODS) < 6]
        subjects = subjects_by_class[i + 1]  # Disciplinas da turma atual
        total_slots_needed = sum(subject["Aulas"] for subject in subjects)  # Total de slots necessários

        if total_slots_needed > len(available_slots):
            raise ValueError(f"Total de aulas ({total_slots_needed}) excede o número de slots disponíveis ({len(available_slots)}) para a turma {i + 1}.")

        for subject in subjects:  # Para cada disciplina
            if subject["Aulas"] > len(available_slots):
                raise ValueError(f"A disciplina {subject['Nome']} requer mais slots ({subject['Aulas']}) do que os disponíveis ({len(available_slots)}).")

            # Escolher slots aleatórios disponíveis para a disciplina
            subject_slots = random.sample(available_slots, subject["Aulas"])
            for slot in subject_slots:
                group[slot] = subject  # Alocar a disciplina no slot
                available_slots.remove(slot)  # Remover o slot da lista de disponíveis

        # Preencher quaisquer slots restantes com "JANELA" (períodos livres)
        for slot in available_slots:
            group[slot] = {"Id": -1, "Professor": None, "Nome": "JANELA"}

        # Verificação adicional para garantir que não haja slots 'None'
        for idx in range(len(group)):
            if group[idx] is None:
                group[idx] = {"Id": -1, "Professor": None, "Nome": "JANELA"}

        individual.append(group)  # Adicionar o cronograma da turma ao indivíduo

    return individual

# Função para criar a população inicial
def create_population(population_size):
    return [create_individual() for _ in range(population_size)]  # Criar uma lista de indivíduos

# Função para calcular a aptidão (fitness) de um indivíduo
def fitness(individual):
    score = 0  # Inicializar pontuação
    cp = 0  # Contador de conflitos de professores (mais de um professor no mesmo período)
    ocr2 = 0  # Contador de aulas duplas consecutivas
    ocr3 = 0  # Contador de aulas triplas consecutivas
    ocr4 = 0  # Contador de aulas quádruplas consecutivas
    janela_intermediaria = 0  # Penalidade para janelas intermediárias
    days_used = set()  # Conjunto de dias utilizados

    # Verificar aulas consecutivas
    for i in range(DAYS):
        for z in range(CLASS):
            if len(individual[z]) > i * PERIODS + PERIODS:
                v = individual[z][i * PERIODS:(i + 1) * PERIODS]
                comparador = v[0] if v and v[0] is not None else None
                if comparador:
                    count = 1
                    for j in range(1, PERIODS):
                        if len(v) > j and v[j] is not None and comparador['Nome'] == v[j]['Nome']:
                            count += 1
                        else:
                            if count == 2:
                                ocr2 += 1
                            elif count == 3:
                                ocr3 += 1
                            elif count == 4:
                                ocr4 += 1
                            count = 1
                            comparador = v[j] if len(v) > j and v[j] is not None else None
                    if count == 2:
                        ocr2 += 1
                    elif count == 3:
                        ocr3 += 1
                    elif count == 4:
                        ocr4 += 1

    # Verificar conflitos de professores e janelas intermediárias
    for i in range(PERIODS * DAYS):
        counter = []
        for j in range(CLASS):
            if len(individual[j]) > i and individual[j][i] is not None:
                counter.append(individual[j][i]['Professor'])
                # Penalizar janelas intermediárias (períodos livres entre aulas)
                if (i % PERIODS != 0 and i % PERIODS != PERIODS - 1) and individual[j][i]['Nome'] == "JANELA":
                    janela_intermediaria += 1
        counter_set = set(counter)
        cp += (len(counter) - len(counter_set))  # Contar conflitos de professores

    # Verificar os dias utilizados
    for i in range(CLASS):
        for j in range(PERIODS * DAYS):
            if j < len(individual[i]) and individual[i][j] is not None and individual[i][j]['Nome'] != "JANELA":
                days_used.add(j // PERIODS)

    # Calcular a pontuação final
    return 7900 + 10 * ocr2 - 50 * ocr3 - 100 * ocr4 - 500 * cp - 200 * janela_intermediaria + 200 * len(days_used)

# Função para selecionar os pais para a próxima geração (roleta)
def select(population, fitnesses, num_parents):
    total_fitness = sum(fitnesses)  # Somar a aptidão total da população
    probs = [f / total_fitness for f in fitnesses]  # Calcular as probabilidades de seleção
    parents = []
    for _ in range(num_parents):
        r = random.random()  # Gerar um número aleatório
        for i, individual in enumerate(population):
            r -= probs[i]
            if r <= 0:
                parents.append(individual)  # Selecionar o indivíduo como pai
                break
    return parents

# Função de crossover para combinar dois pais e gerar filhos
def crossover(parent1, parent2, crossover_rate):
    crossover_point = random.randint(0, CLASS - 1)  # Escolher um ponto de crossover aleatório
    should_cross = random.random()
    if should_cross < crossover_rate:
        child1 = parent1[:crossover_point] + parent2[crossover_point:]  # Criar o primeiro filho
        child2 = parent2[:crossover_point] + parent1[crossover_point:]  # Criar o segundo filho
        return deepcopy(child1), deepcopy(child2)  # Retornar os filhos
    else:
        return parent1, parent2  # Se não houver crossover, retornar os pais inalterados

# Função de mutação para introduzir variações aleatórias nos indivíduos
def mutate(individual, mutation_rate):
    individual_copy = deepcopy(individual)  # Fazer uma cópia do indivíduo
    for i in range(CLASS):
        for _ in range(PERIODS * DAYS):
            if random.random() < mutation_rate:
                # Escolher dois slots aleatórios e trocá-los
                j_chosen1 = random.randint(0, len(individual_copy[i]) - 1)
                j_chosen2 = random.randint(0, len(individual_copy[i]) - 1)
                individual_copy[i][j_chosen1], individual_copy[i][j_chosen2] = individual_copy[i][j_chosen2], individual_copy[i][j_chosen1]
    return individual_copy

# Função para substituir a população antiga pela nova
def replace_population(old_population, new_population):
    return new_population  # Simplesmente substituir a população antiga

# Função principal do Algoritmo Genético
def genetic_algorithm(population_size, num_generations, mutation_rate, crossover_rate, elitism):
    population = create_population(population_size)  # Criação da população inicial
    max_fitness_scores = []  # Lista para armazenar as pontuações máximas de aptidão
    plt.figure()  # Inicializar uma nova figura para o gráfico
    for gen in range(num_generations):
        fitnesses = [fitness(individual) for individual in population]  # Calcular a aptidão de cada indivíduo
        parents = select(population, fitnesses, population_size // 2)  # Selecionar os pais
        children = []
        for i in range(0, len(parents), 2):
            child1, child2 = crossover(parents[i], parents[i + 1], crossover_rate)  # Gerar filhos
            children.append(child1)
            children.append(child2)
        mutated_children = [mutate(child, mutation_rate) for child in children]  # Mutar os filhos
        population.sort(key=fitness, reverse=True)  # Ordenar a população pela aptidão
        elites = deepcopy(population[:elitism])  # Preservar os melhores indivíduos (elitismo)
        population = replace_population(population, mutated_children)  # Substituir a população antiga pela nova
        population.sort(key=fitness, reverse=False)
        population[:elitism] = deepcopy(elites)  # Reinserir os elites
        best_individual = max(population, key=fitness)  # Encontrar o melhor indivíduo
        max_fitness_scores.append(fitness(best_individual))  # Registrar a melhor aptidão
        plt.plot(max_fitness_scores)  # Plotar a aptidão máxima
        plt.pause(0.05)  # Atualizar o gráfico

    plt.show()  # Mostrar o gráfico final
    best_individual = max(population, key=fitness)  # Retornar o melhor indivíduo
    return best_individual

# Parâmetros do Algoritmo Genético
population_size = 100  # Tamanho da população
num_generations = 1000  # Número de gerações
mutation_rate = 0.05  # Taxa de mutação
crossover_rate = 0.7  # Taxa de crossover
elitism = 2  # Número de indivíduos preservados como elite

# Executar o algoritmo
best_schedule = genetic_algorithm(population_size, num_generations, mutation_rate, crossover_rate, elitism)

# Exibir o melhor cronograma
print("Melhor Cronograma:")
for i, class_schedule in enumerate(best_schedule):
    print(f"\nTurma {i + 1}:")
    for j, slot in enumerate(class_schedule):
        day = j // PERIODS  # Calcular o dia
        period = j % PERIODS  # Calcular o período
        print(f"Dia {day + 1}, Período {period + 1}: {slot['Nome']} com {slot['Professor']}")  # Imprimir a disciplina e o professor
