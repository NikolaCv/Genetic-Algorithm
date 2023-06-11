import copy
import random
import numpy as np


def generate_first_generation(N, N_cities, start_city):
    population = []
    # Each individual is a random permutation of cities excluding the start city, since start city is fixed
    for i in range(N):
        # Generate a random permutation of cities excluding the start city
        population.append(list(np.random.permutation([c for c in range(0, N_cities) if c != start_city])))

    return population


def fitness(population, N_cities, distances, start_city):
    population_fitness = []
    for i in range(0, len(population)):
        sumF = 0

        # Calculate the distance traveled from each city to the next
        for j in range(0, N_cities - 2):
            sumF += distances[population[i][j]][population[i][j + 1]]

        # Add the distance from the last city to the start city and from the start city to the first city
        sumF += distances[population[i][N_cities - 2]][start_city]
        sumF += distances[population[i][0]][start_city]
        population_fitness.append(sumF)

    return population_fitness


def elitism(population, N_elitism):
    selected_population = []

    # Select the best individuals (based on fitness) as the elite for the next generation
    # population is always sorted best to worst
    for i in range(0, N_elitism):
        selected_population.append(population[i])

    return selected_population


def selection(population, N_elitism, N):
    selected_population = elitism(population, N_elitism)

    new_population_fitness = []

    # Assign a fitness value based on rank for rank-based selection
    for i in range(0, N):
        new_population_fitness.append(N - i)

    # Add individuals based on rang selection
    for i in range(N_elitism, int(N / 2)):
        ind = selection_rang(new_population_fitness, N)
        selected_population.append(population[ind])

    return selected_population


def selection_rang(population_fitness, N):
    fitness_sum = sum(population_fitness)

    # Generate a random number within the range of fitness sum
    stop_sum = random.uniform(0, fitness_sum)

    curr_sum = 0
    # Calculate the cumulative sum of fitness values
    # When it becomes larger than stop_sum, current individual is selected
    for i in range(0, N):
        curr_sum += population_fitness[i]
        if (curr_sum >= stop_sum):
            return i


def cross_over(selected_population, N):
    new_population = []

    # Copy best half of the population to the next generation
    for j in range(0, int(N / 2)):
        new_population.append(copy.deepcopy(selected_population[j]))

    # For each pair of individuals create 2 new individuals by exchanging their genes
    for i in range(0, int(len(selected_population) / 2)):
        child1 = exchange_array(selected_population[2 * i], selected_population[2 * i + 1])
        child2 = exchange_array(selected_population[2 * i + 1], selected_population[2 * i])
        new_population.append(child1)
        new_population.append(child2)

    return new_population


def exchange_array(individual1, individual2):
    a = random.randint(0, len(individual1) - 1)
    b = random.randint(0, len(individual1) - 1)

    if a > b:
        a, b = b, a

    new_individual = []
    # Copy genes between a and b to the new individual
    for i in range(a, b):
        new_individual.append(individual2[i])

    # Copy the remaining missing genes
    for i in range(0, len(individual1)):
        if individual1[i] not in new_individual:
            new_individual.append(individual1[i])

    return new_individual


def mutation(individual, mutation_rate):
    x = random.uniform(0, 1)

    # Exchange genes on positions a and b
    if (x < mutation_rate):
        a = random.randint(0, len(individual) - 1)
        b = random.randint(0, len(individual) - 1)
        individual[a], individual[b] = individual[b], individual[a]

    return individual


def genetic_algorithm(N, N_cities, N_populations, N_generations, start_city, N_elitism, mutation_rate, distances,
                      info_every_K_epochs=10):
    max_distance = N_cities * np.max(distances) * 2

    best_solution = {
        "Distance": max_distance,
        "Order": []
    }

    for k in range(1, N_populations + 1):
        # Initialize population and its fitness
        population = generate_first_generation(N, N_cities, start_city)
        population_fitness = fitness(population, N_cities, distances, start_city)

        # Sort by fitness, lowest is the best
        population = sort_population(population, population_fitness)

        for i in range(1, N_generations + 1):
            # Select and do cross over, creating a new population
            selected_population = selection(population, N_elitism, N)
            population = cross_over(selected_population, N)

            # Mutate new population
            for j in range(N_elitism, N):
                population[j] = mutation(population[j], mutation_rate)

            # Calculate fitness for the new population
            population_fitness = fitness(population, N_cities, distances, start_city)

            # Sort by fitness, lowest is the best
            population = sort_population(population, population_fitness)

            if not i % info_every_K_epochs:
                print(f'Population {k}, Gen: {i}\nBest current: {population_fitness[0]}, Best overall: {best_solution}')

        if best_solution["Distance"] > population_fitness[0]:
            best_solution["Distance"] = population_fitness[0]
            best_solution["Order"] = copy.deepcopy(population[0])

    return best_solution


def sort_population(population, population_fitness):
    population = [x for _, x in sorted(zip(population_fitness, population))]
    population_fitness.sort()

    return population


if __name__ == '__main__':
    N = 2000
    elitism_percent = 0.01
    N_elitism = int(N * elitism_percent)
    N_cities = 20
    N_generations = 2500
    N_populations = 50
    start_city = 7
    mutation_rate = 0.2

    np.random.seed(42)

    # Create matrix of distrances where distance[i][j] corresponds to distance between city i and j in that order
    # This means that distance i -> j != distance j -> i
    distances = np.random.randint(2, 100, size=(N_cities, N_cities))

    # Create best solution to be distance of 20, in order 1 -> 2 -> 3 ... -> 20 -> 1
    for i in range(N_cities - 1):
        distances[i][i + 1] = 1
        distances[i][i] = 0
    distances[N_cities - 1][0] = 1
    distances[N_cities - 1][N_cities - 1] = 0

    best_solution = genetic_algorithm(N, N_cities, N_populations, N_generations, start_city, N_elitism, mutation_rate,
                                      distances, info_every_K_epochs=50)

    print(best_solution)
