import numpy as np
import random
import matplotlib.pyplot as plt

# Define the function to optimize
def func(x, y):
    return (1 - x) ** 2 * np.exp(-x ** 2 - (y + 1) ** 2) - (x - x ** 3 - y ** 3) * np.exp(-x ** 2 - y ** 2)

# Genetic Algorithm parameters
pop_size = 8
cross_prob = 0.7
mut_prob = 0.01
generations = 200

# Step 1: Initialize population
def initialize_population(size):
    return [(random.uniform(-2, 2), random.uniform(-2, 2)) for _ in range(size)]

# Step 2: Define fitness function
def calculate_fitness(population):
    return [func(x, y) for x, y in population]

# Step 5: Select parents for mating
def select_parents(population, fitness):
    return random.choices(population, weights=fitness, k=len(population))

# Step 6: Create offspring by crossover and mutation
def crossover_and_mutate(parents):
    offspring = []
    for i in range(0, len(parents), 2):
        parent1, parent2 = parents[i], parents[i + 1]
        if random.random() < cross_prob:
            cross_point = random.randint(0, 1)
            child1 = (parent1[0], parent2[1])
            child2 = (parent2[0], parent1[1])
        else:
            child1, child2 = parent1, parent2
        # Mutation
        child1 = (child1[0] + random.gauss(0, mut_prob), child1[1] + random.gauss(0, mut_prob))
        child2 = (child2[0] + random.gauss(0, mut_prob), child2[1] + random.gauss(0, mut_prob))
        offspring.extend([child1, child2])
    return offspring

# Step 9: Replace the initial population with the new population
def replace_population(old_population, offspring):
    return offspring

# Genetic Algorithm
population = initialize_population(pop_size)
best_solutions = []
for gen in range(generations):
    # Step 4: Calculate fitness of each individual chromosome
    fitness = calculate_fitness(population)

    # Step 5: Select parents for mating
    parents = select_parents(population, fitness)

    # Step 6: Create offspring by crossover and mutation
    offspring = crossover_and_mutate(parents)

    # Step 9: Replace the initial population with the new population
    population = replace_population(population, offspring)

    # Store the best solution of this generation
    best_index = np.argmax(calculate_fitness(population))
    best_solutions.append((gen, population[best_index], max(fitness)))

# Find the overall best solution
overall_best_index = np.argmax([func(x, y) for _, (x, y), _ in best_solutions])
overall_best_gen, overall_best_x, overall_best_y = best_solutions[overall_best_index][0], best_solutions[overall_best_index][1][0], best_solutions[overall_best_index][1][1]
overall_best_value = func(overall_best_x, overall_best_y)

# Information about the best generation
best_generation = overall_best_gen
best_generation_x = overall_best_x
best_generation_y = overall_best_y
best_generation_fitness = overall_best_value

print("The maximum point is at x = {:.2f}, y = {:.2f}".format(overall_best_x, overall_best_y))
print("The maximum value of the function is {:.2f}".format(overall_best_value))
print("Best generation: {}, x = {:.3f}, y = {:.3f}, fitness = {:.3f}".format(best_generation,
best_generation_x, best_generation_y, best_generation_fitness))

# Plotting the evolution of the best solution
generations = [gen for gen, _, _ in best_solutions]
x_values = [x for _, (x, _) , _ in best_solutions]
y_values = [y for _, (_, y), _ in best_solutions]

plt.figure(figsize=(10, 6))
plt.plot(generations, x_values, label='x values')
plt.plot(generations, y_values, label='y values')
plt.xlabel('Generation')
plt.ylabel('Value')
plt.title('Evolution of Best Solution')

# Annotate the best solution at the top of the graph
plt.annotate('Best: f({:.2f}, {:.2f}) = {:.2f}'.format(overall_best_x, overall_best_y,
overall_best_value),
xy=(0.5, 0.95), xycoords='axes fraction',
xytext=(0.5, 1.05), textcoords='axes fraction',
horizontalalignment='center', verticalalignment='center')

# Annotate the best generation on the graph
plt.annotate('Best Generation: {}'.format(best_generation),
xy=(best_generation, overall_best_value), xycoords='data',
xytext=(-20, 20), textcoords='offset points',
arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

plt.legend()
plt.grid(True)
plt.show()
