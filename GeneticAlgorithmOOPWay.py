import numpy as np
import random
import matplotlib.pyplot as plt

class GeneticAlgorithm:
    def __init__(self, pop_size, cross_prob, mut_prob, generations):
        self.pop_size = pop_size
        self.cross_prob = cross_prob
        self.mut_prob = mut_prob
        self.generations = generations

    def func(self, x, y):
        return (1 - x) ** 2 * np.exp(-x ** 2 - (y + 1) ** 2) - (x - x ** 3 - y ** 3) * np.exp(-x ** 2 - y ** 2)

    def initialize_population(self):
        return [(random.uniform(-2, 2), random.uniform(-2, 2)) for _ in range(self.pop_size)]

    def calculate_fitness(self, population):
        return [self.func(x, y) for x, y in population]

    def select_parents(self, population, fitness):
        return random.choices(population, weights=fitness, k=len(population))

    def crossover_and_mutate(self, parents):
        offspring = []
        for i in range(0, len(parents), 2):
            parent1, parent2 = parents[i], parents[i + 1]
            if random.random() < self.cross_prob:
                cross_point = random.randint(0, 1)
                child1 = (parent1[0], parent2[1])
                child2 = (parent2[0], parent1[1])
            else:
                child1, child2 = parent1, parent2
            # Mutation
            child1 = (child1[0] + random.gauss(0, self.mut_prob), child1[1] + random.gauss(0, self.mut_prob))
            child2 = (child2[0] + random.gauss(0, self.mut_prob), child2[1] + random.gauss(0, self.mut_prob))
            offspring.extend([child1, child2])
        return offspring

    def replace_population(self, old_population, offspring):
        return offspring

    def run_genetic_algorithm(self):
        population = self.initialize_population()
        best_solutions = []
        for gen in range(self.generations):
            fitness = self.calculate_fitness(population)
            parents = self.select_parents(population, fitness)
            offspring = self.crossover_and_mutate(parents)
            population = self.replace_population(population, offspring)
            best_index = np.argmax(self.calculate_fitness(population))
            best_solutions.append((gen, population[best_index], max(fitness)))

        # Find the overall best solution
        overall_best_index = np.argmax([self.func(x, y) for _, (x, y), _ in best_solutions])
        overall_best_gen, overall_best_x, overall_best_y = best_solutions[overall_best_index][0], \
                                                            best_solutions[overall_best_index][1][0], \
                                                            best_solutions[overall_best_index][1][1]
        overall_best_value = self.func(overall_best_x, overall_best_y)

        # Information about the best generation
        best_generation = overall_best_gen
        best_generation_x = overall_best_x
        best_generation_y = overall_best_y
        best_generation_fitness = overall_best_value

        print("The maximum point is at x = {:.2f}, y = {:.2f}".format(overall_best_x, overall_best_y))
        print("The maximum value of the function is {:.2f}".format(overall_best_value))
        print("Best generation: {}, x = {:.3f}, y = {:.3f}, fitness = {:.3f}".format(best_generation,
                                                                                        best_generation_x,
                                                                                        best_generation_y,
                                                                                        best_generation_fitness))
        
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
                    xytext=(0.5, 1.09), textcoords='axes fraction',
                    horizontalalignment='center', verticalalignment='center')
        
   

        # Annotate the best generation on the graph
        plt.annotate('Best Generation: {}'.format(best_generation),
                    xy=(best_generation, overall_best_value), xycoords='data',
                    xytext=(-20, 20), textcoords='offset points',
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    pop_size = 8
    cross_prob = 0.7
    mut_prob = 0.01
    generations = 200

    ga = GeneticAlgorithm(pop_size, cross_prob, mut_prob, generations)
    ga.run_genetic_algorithm()
