from itertools import compress
import random
import time
import matplotlib.pyplot as plt

from data import get_big


def initial_population(individual_size, population_size) -> list:
    """Generate random staring population"""
    return [[random.choice([True, False]) for _ in range(individual_size)] for _ in range(population_size)]


def fitness(items, knapsack_max_capacity, individual) -> int:
    """Calculate the fitness of the individual solution"""
    total_weight = sum(compress(items['Weight'], individual))
    if total_weight > knapsack_max_capacity:
        return 0
    return sum(compress(items['Value'], individual))


def population_best(items, knapsack_max_capacity, population) -> tuple:
    """Find best fitting solution from the population"""
    best_individual = None
    best_individual_fitness = -1
    for individual in population:
        individual_fitness = fitness(items, knapsack_max_capacity, individual)
        if individual_fitness > best_individual_fitness:
            best_individual = individual
            best_individual_fitness = individual_fitness
    return best_individual, best_individual_fitness


def roulette_selection(items, knapsack_max_capacity, population) -> list:

    fitness_vector = [fitness(items, knapsack_max_capacity, individual) for individual in population]
    total_fitness = sum(fitness_vector)
    probabilities = [fit / total_fitness for fit in fitness_vector]
    cumulative_probabilities = [sum(probabilities[:i + 1]) for i in range(len(probabilities))]

    selected = []
    for _ in range(len(population)):
        r = random.random()
        for i, cum_prob in enumerate(cumulative_probabilities):
            if r <= cum_prob:
                selected.append(population[i])
                break

    return selected


def crossover(population) -> list:
    children = []

    for i in range(0, len(population), 2):

        p1 = population[i]
        p2 = population[i + 1]

        split_point = len(p1) // 2

        child1 = p1[:split_point] + p2[split_point:]
        child2 = p2[:split_point] + p1[split_point:]

        children.append(child1)
        children.append(child2)

    return children


def mutate(population, mutation_rate):
    for individual in population:
        for i in range(len(individual)):
            r = random.random()
            if r < mutation_rate:
                individual[i] = random.choice((0, 1))


def next_generation(items, knapsack_max_capacity, population) -> list:
    parents = roulette_selection(items, knapsack_max_capacity, population)
    children = crossover(parents)
    mutate(children, 0.1)
    return children


def main():
    items, knapsack_max_capacity = get_big()
    print(items)

    population_size = 100
    generations = 200

    start_time = time.time()
    best_solution = None
    best_fitness = 0
    population_history = []
    best_history = []
    population = initial_population(len(items), population_size)
    for _ in range(generations):
        population_history.append(population)
        population = next_generation(items, knapsack_max_capacity, population)
        best_individual, best_individual_fitness = population_best(items, knapsack_max_capacity, population)
        if best_individual_fitness > best_fitness:
            best_solution = best_individual
            best_fitness = best_individual_fitness
        best_history.append(best_fitness)

    end_time = time.time()
    total_time = end_time - start_time
    print('Best solution:', list(compress(items['Name'], best_solution)))
    print('Best solution value:', best_fitness)
    print('Time: ', total_time)

    # plot generations
    x = []
    y = []
    top_best = 10
    for i, population in enumerate(population_history):
        plotted_individuals = min(len(population), top_best)
        x.extend([i] * plotted_individuals)
        population_fitness = [fitness(items, knapsack_max_capacity, individual) for individual in population]
        population_fitness.sort(reverse=True)
        y.extend(population_fitness[:plotted_individuals])
    plt.scatter(x, y, marker='.')
    plt.plot(best_history, 'r')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.show()


if __name__ == '__main__':
    main()
