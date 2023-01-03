import numpy as np


class GeneticAlgorithm:

    def __init__(self, chromosomeShape,
                 errorFunction,
                 elitism=1,
                 populationSize=5,
                 mutationProbability=0.1,
                 mutationScale=0.1,
                 numIterations=10000
                 ):
        self.populationSize = populationSize
        self.p = mutationProbability
        self.numIter = numIterations
        self.f = errorFunction
        self.keep = elitism
        self.k = mutationScale

        self.i = 0

        self.population = []
        for _ in range(populationSize):
            chromosome = np.random.randn(chromosomeShape) * 0.01
            fitness = self.calculateFitness(chromosome)
            self.population.append((chromosome, fitness))
        self.population = sorted(self.population, key=lambda t: -t[1])

    def step(self):

        self.i += 1
        new_population = self.bestN(self.keep)
        while len(new_population) < self.populationSize:
            first_parent, second_parent = self.selectParents()
            child = self.crossover(first_parent, second_parent)
            chromosome = np.array(self.mutate(child))
            fitness = self.calculateFitness(chromosome)
            new_population.append((chromosome, fitness))
        self.population = sorted(new_population, key=lambda t: -t[1])
        should_stop = self.i > self.numIter
        return should_stop, self.i, self.best()[0]

    def calculateFitness(self, chromosome):
        chromosomeError = self.f(chromosome)
        return 1 / chromosomeError

    def bestN(self, n):
        return self.population[0:n]

    def best(self):
        return self.population[0]

    def selectParents(self):
        fitness_sum = 0
        for value in self.population:
            fitness_sum += value[1]

        first_parent = self.roulette_wheel_selection(fitness_sum)
        second_parent = self.roulette_wheel_selection(fitness_sum)

        return first_parent, second_parent

    def roulette_wheel_selection(self, fitness_sum):
        pointer = np.random.uniform(0, fitness_sum)
        current_fitness = 0
        for chromosome, fitness in self.population:
            current_fitness += fitness
            if current_fitness > pointer:
                return chromosome

    def crossover(self, p1, p2):
        return (p1 + p2) / 2

    def mutate(self, chromosome):
        for index in range(len(chromosome)):
            if np.random.random() < self.p:
                chromosome[index] += np.random.normal(0, self.k)
        return chromosome
