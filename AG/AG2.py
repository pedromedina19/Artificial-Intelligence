import numpy as np
import random

# Definindo a função de aptidão baseada na função fornecida
def fitness_function(x, y):
    return 15 + x * np.cos(2 * np.pi * x) + y * np.cos(14 * np.pi * y)

# Inicialização da população
def initialize_population(pop_size, bounds):
    population = []
    for _ in range(pop_size):
        individual = [random.uniform(bounds[0][0], bounds[0][1]), random.uniform(bounds[1][0], bounds[1][1])]
        population.append(individual)
    return population

# Avaliação da população
def evaluate_population(population):
    scores = []
    for individual in population:
        scores.append(fitness_function(individual[0], individual[1]))
    return scores

# Seleção por torneio
def tournament_selection(population, scores, tournament_size):
    tournament = random.sample(list(zip(population, scores)), tournament_size)
    tournament.sort(key=lambda x: x[1], reverse=True)
    return tournament[0][0], tournament[1][0]

# Seleção por roleta
def roulette_selection(population, scores):
    max_score = sum(scores)
    pick = random.uniform(0, max_score)
    current = 0
    for individual, score in zip(population, scores):
        current += score
        if current > pick:
            return individual

# Crossover de um ponto
def one_point_crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

# Crossover de dois pontos
def two_point_crossover(parent1, parent2):
    point1 = random.randint(1, len(parent1) - 1)
    point2 = random.randint(1, len(parent1) - 1)
    if point1 > point2:
        point1, point2 = point2, point1

    child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
    child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]

    return child1, child2

# Mutação
def mutate(individual, bounds, mutation_rate):
    for idx in range(len(individual)):
        if random.random() < mutation_rate:
            individual[idx] = random.uniform(bounds[idx][0], bounds[idx][1])
    return individual

# Elitismo
def elitism(population, scores, elite_size):
    elite_indices = np.argsort(scores)[-elite_size:]
    return [population[i] for i in elite_indices]

# Algoritmo Genético
def genetic_algorithm(pop_size, bounds, crossover_rate, mutation_rate, generations, selection_method, tournament_size, elite_size, crossover_type):
    population = initialize_population(pop_size, bounds)
    
    for generation in range(generations):
        scores = evaluate_population(population)
        next_generation = elitism(population, scores, elite_size)
        
        while len(next_generation) < pop_size:
            if selection_method == 'torneio':
                parent1, parent2 = tournament_selection(population, scores, tournament_size)
            elif selection_method == 'roleta':
                parent1 = roulette_selection(population, scores)
                parent2 = roulette_selection(population, scores)
            else:
                raise ValueError("Método de seleção não suportado")

            if random.random() < crossover_rate:
                if crossover_type == 'one_point':
                    child1, child2 = one_point_crossover(parent1, parent2)
                elif crossover_type == 'two_point':
                    child1, child2 = two_point_crossover(parent1, parent2)
                else:
                    raise ValueError("Tipo de crossover não suportado")
            else:
                child1, child2 = parent1, parent2

            child1 = mutate(child1, bounds, mutation_rate)
            child2 = mutate(child2, bounds, mutation_rate)
            
            next_generation.extend([child1, child2])

        population = next_generation[:pop_size]
    
    best_index = np.argmax(evaluate_population(population))
    best_individual = population[best_index]
    return best_individual

# Função principal para obter os parâmetros e executar o algoritmo genético
def main():
    pop_size = int(input("Tamanho da população: "))
    chromosome_length = 2  # Tamanho fixo do cromossomo para este problema
    bounds = [[float(input(f"Limite inferior para o gene {i+1}: ")), float(input(f"Limite superior para o gene {i+1}: "))] for i in range(chromosome_length)]
    crossover_rate = float(input("Probabilidade de cruzamento: "))
    mutation_rate = float(input("Probabilidade de mutação: "))
    generations = int(input("Quantidade de gerações: "))
    selection_method = input("Método de seleção (roleta/torneio): ").lower()
    
    if selection_method == 'torneio':
        tournament_size = int(input("Tamanho do torneio: "))
    else:
        tournament_size = None
    
    elite_size = int(input("Tamanho do elitismo: "))
    crossover_type = input("Tipo de crossover (one_point/two_point): ").lower()
    
    # Executar o Algoritmo Genético
    best_solution = genetic_algorithm(pop_size, bounds, crossover_rate, mutation_rate, generations, selection_method, tournament_size, elite_size, crossover_type)
    print("Melhor solução encontrada:", best_solution)
    print("Aptidão da melhor solução:", fitness_function(best_solution[0], best_solution[1]))

if __name__ == "__main__":
    main()
