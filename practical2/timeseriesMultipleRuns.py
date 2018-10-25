
import random
import numpy as np
import matplotlib.pyplot as plt

from deap import base
from deap import creator
from deap import tools

trainingData = np.genfromtxt ('training.csv', delimiter=",")

# We want to minimize because then the fitness can be the difference
# between our predicted value and the actual value
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Attribute generator 
toolbox.register("attr_float", random.random)

# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual, 
    toolbox.attr_float, 4)

# define the population to be a list of individuals
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Provides a prediction for a given row from the first 4 datapoints
def prediction(individual,row):
	dotproduct = sum(i * r for i,r in zip(individual, row) )
	return( dotproduct / sum(individual) )

# The goal - fitness function to be mimimized
def evalMeanSquDiff(individual):
	totalSquDiff = sum( (prediction(individual, row) - row[4])**2 for row in trainingData )
	return( totalSquDiff / len(trainingData), )

toolbox.register("evaluate", evalMeanSquDiff)

# crossover operator
toolbox.register("mate", tools.cxOnePoint)

# register a mutation operator using the Guassian distribution
toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.2, indpb=0.005)

toolbox.register("select", tools.selTournament, tournsize=10)


def main():
	random.seed(64)

	NGEN, NRUN, CXPB = 20, 20, 0.5

	meanFitness = [None] * NGEN

    # create an initial population of 300 individuals (where
    # each individual is a list of integers)
	pop = toolbox.population(n=300)

	for r in range(NRUN):

		print("Start of evolution: Run %i " % r)

	    # Evaluate the entire population
		fitnesses = list(map(toolbox.evaluate, pop))
		for ind, fit in zip(pop, fitnesses):
			ind.fitness.values = fit

		for g in range(NGEN):
			print("-- Generation %i --" % g)

			# Select the next generation individuals
			offspring = toolbox.select(pop, len(pop))
			# Clone the selected individuals
			offspring = list(map(toolbox.clone, offspring))

	        # Apply crossover on the offspring
			for child1, child2 in zip(offspring[::2], offspring[1::2]):
				# cross two individuals with probability CXPB
				if random.random() < CXPB:
					toolbox.mate(child1, child2)
					# fitness values of the children
					# must be recalculated later
					del child1.fitness.values
					del child2.fitness.values

	    	# Apply mutation on the offspring
			for mutant in offspring:
				toolbox.mutate(mutant)
				del mutant.fitness.values

			# Evaluate the individuals with an invalid fitness
			invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
			fitnesses = map(toolbox.evaluate, invalid_ind)
			for ind, fit in zip(invalid_ind, fitnesses):
				ind.fitness.values = fit

			print("  Evaluated %i individuals" % len(invalid_ind))

			# The population is entirely replaced by the offspring
			pop[:] = offspring

			# Gather all the fitnesses in one list and print the stats
			fits = [ind.fitness.values[0] for ind in pop]

			length = len(pop)
			mean = sum(fits) / length
			sum2 = sum(x*x for x in fits)
			std = abs(sum2 / length - mean**2)**0.5

			print("  Min %s" % min(fits))
			print("  Max %s" % max(fits))
			print("  Avg %s" % mean)
			print("  Std %s" % std)

		print("-- End of evolution --")

		best_ind = tools.selBest(pop, 1)[0]
		print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

		meanFitness[r] = mean

	x = range(1,NGEN+1)
	#plt.plot(x, meanFitness)
	#plt.ylabel('Fitness')
	#plt.xlabel('Generation')
	#plt.show()
    
	plt.boxplot(meanFitness)
	plt.show()
    
if __name__ == "__main__":
	main()

