#Exam No. Y3838392

import curses
import random
import operator
import copy
import numpy
from functools import partial
import pandas as pd
import time

from deap import base
from deap import creator
from deap import tools
from deap import gp

S_RIGHT, S_LEFT, S_UP, S_DOWN = 0, 1, 2, 3
XSIZE, YSIZE = 14, 14
NFOOD = 1
MAX_SCORE = 133

# Calls one of the given function depending on the result of the condition function
def if_then_else(condition, out1, out2):
    out1() if condition() else out2()


# Snake agent class
class SnakePlayer(list):
    global S_RIGHT, S_LEFT, S_UP, S_DOWN
    global XSIZE, YSIZE

    def __init__(self):
        self.direction = S_RIGHT
        self.body = [[4, 10], [4, 9], [4, 8], [4, 7], [4, 6], [4, 5], [4, 4], [4, 3], [4, 2], [4, 1], [4, 0]]
        self.score = 0
        self.ahead = []
        self.food = []

    def _reset(self):
        self.direction = S_RIGHT
        self.body[:] = [[4, 10], [4, 9], [4, 8], [4, 7], [4, 6], [4, 5], [4, 4], [4, 3], [4, 2], [4, 1], [4, 0]]
        self.score = 0
        self.ahead = []
        self.food = []

    def getScore(self):
        return self.score

    def getAheadLocation(self):
        self.ahead = [self.body[0][0] + (self.direction == S_DOWN and 1) + (self.direction == S_UP and -1),
                      self.body[0][1] + (self.direction == S_LEFT and -1) + (self.direction == S_RIGHT and 1)]

    def updatePosition(self):
        self.getAheadLocation()
        self.body.insert(0, self.ahead)

    def snakeHasCollided(self):
        return self.is_obstacle_location(self.body[0])


    # Methods used by senses
    def is_obstacle_location(self, location):
        return not (location[0] > 0 and location[0] < (YSIZE - 1) and location[1] > 0 and location[1] < (XSIZE - 1)
                and location not in self.body[1:])

    def is_out_of_bounds_location(self, location):
        return not (location[0] > 0 and location[0] < (YSIZE - 1) and location[1] > 0 and  location[1] < (XSIZE - 1))

    def is_tail_location(self, location):
        return location not in self.body[1:]

    # Snake terminals
    def changeDirectionUp(self):
        self.direction = S_UP

    def changeDirectionRight(self):
        self.direction = S_RIGHT

    def changeDirectionDown(self):
        self.direction = S_DOWN

    def changeDirectionLeft(self):
        self.direction = S_LEFT

    # Sense food in every direction
    def sense_food_up(self):
        return self.food[0][0] < self.body[0][0]

    def sense_food_down(self):
        return self.food[0][0] > self.body[0][0]

    def sense_food_left(self):
        return self.food[0][1] < self.body[0][1]

    def sense_food_right(self):
        return self.food[0][1] > self.body[0][1]

    # Sense wall/tail 1 unit away from the head in every direction
    def sense_obstacle_up_1(self):
        up = [self.body[0][0] - 1, self.body[0][1]]
        return self.is_obstacle_location(up)

    def sense_obstacle_down_1(self):
        down = [self.body[0][0] + 1, self.body[0][1]]
        return self.is_obstacle_location(down)

    def sense_obstacle_left_1(self):
        left = [self.body[0][0], self.body[0][1] - 1]
        return self.is_obstacle_location(left)

    def sense_obstacle_right_1(self):
        right = [self.body[0][0], self.body[0][1] + 1]
        return self.is_obstacle_location(right)

    # Sense wall/tail 2 unit away from the head in every direction
    def sense_obstacle_up_2(self):
        up = [self.body[0][0] - 2, self.body[0][1]]
        return self.is_obstacle_location(up)

    def sense_obstacle_down_2(self):
        down = [self.body[0][0] + 2, self.body[0][1]]
        return self.is_obstacle_location(down)

    def sense_obstacle_left_2(self):
        left = [self.body[0][0], self.body[0][1] - 2]
        return self.is_obstacle_location(left)

    def sense_obstacle_right_2(self):
        right = [self.body[0][0], self.body[0][1] + 2]
        return self.is_obstacle_location(right)

    # Sense wall 1 unit away from the head in every direction
    def sense_out_of_bounds_up_1(self):
        up = [self.body[0][0] - 1, self.body[0][1]]
        return self.is_out_of_bounds_location(up)

    def sense_out_of_bounds_down_1(self):
        down = [self.body[0][0] + 1, self.body[0][1]]
        return self.is_out_of_bounds_location(down)

    def sense_out_of_bounds_left_1(self):
        left = [self.body[0][0], self.body[0][1] - 1]
        return self.is_out_of_bounds_location(left)

    def sense_out_of_bounds_right_1(self):
        right = [self.body[0][0], self.body[0][1] + 1]
        return self.is_out_of_bounds_location(right)

    # Sense wall 2 unit away from the head in every direction
    def sense_out_of_bounds_up_2(self):
        up = [self.body[0][0] - 2, self.body[0][1]]
        return self.is_out_of_bounds_location(up)

    def sense_out_of_bounds_down_2(self):
        down = [self.body[0][0] + 2, self.body[0][1]]
        return self.is_out_of_bounds_location(down)

    def sense_out_of_bounds_left_2(self):
        left = [self.body[0][0], self.body[0][1] - 2]
        return self.is_out_of_bounds_location(left)

    def sense_out_of_bounds_right_2(self):
        right = [self.body[0][0], self.body[0][1] + 2]
        return self.is_out_of_bounds_location(right)


    # Sense tail 1 unit away from the head in every direction
    def sense_tail_up_1(self):
        up = [self.body[0][0] - 1, self.body[0][1]]
        return self.is_tail_location(up)

    def sense_tail_down_1(self):
        down = [self.body[0][0] + 1, self.body[0][1]]
        return self.is_tail_location(down)

    def sense_tail_left_1(self):
        left = [self.body[0][0], self.body[0][1] - 1]
        return self.is_tail_location(left)

    def sense_tail_right_1(self):
        right = [self.body[0][0], self.body[0][1] + 1]
        return self.is_tail_location(right)

    # Sense tail 2 unit away from the head in every direction
    def sense_tail_up_2(self):
        up = [self.body[0][0] - 2, self.body[0][1]]
        return self.is_tail_location(up)

    def sense_tail_down_2(self):
        down = [self.body[0][0] + 2, self.body[0][1]]
        return self.is_tail_location(down)

    def sense_tail_left_2(self):
        left = [self.body[0][0], self.body[0][1] - 2]
        return self.is_tail_location(left)

    def sense_tail_right_2(self):
        right = [self.body[0][0], self.body[0][1] + 2]
        return self.is_tail_location(right)


    #If then else functions for PrimitiveSet
    def if_food_up(self, out1, out2):
        return partial(if_then_else, self.sense_food_up, out1, out2)

    def if_food_down(self, out1, out2):
        return partial(if_then_else, self.sense_food_down, out1, out2)

    def if_food_left(self, out1, out2):
        return partial(if_then_else, self.sense_food_left, out1, out2)

    def if_food_right(self, out1, out2):
        return partial(if_then_else, self.sense_food_right, out1, out2)


    def if_obstacle_up_1(self, out1, out2):
        return partial(if_then_else, self.sense_obstacle_up_1, out1, out2)

    def if_obstacle_down_1(self, out1, out2):
        return partial(if_then_else, self.sense_obstacle_down_1, out1, out2)

    def if_obstacle_left_1(self, out1, out2):
        return partial(if_then_else, self.sense_obstacle_left_1, out1, out2)

    def if_obstacle_right_1(self, out1, out2):
        return partial(if_then_else, self.sense_obstacle_right_1, out1, out2)

    def if_obstacle_up_2(self, out1, out2):
        return partial(if_then_else, self.sense_obstacle_up_2, out1, out2)

    def if_obstacle_down_2(self, out1, out2):
        return partial(if_then_else, self.sense_obstacle_down_2, out1, out2)

    def if_obstacle_left_2(self, out1, out2):
        return partial(if_then_else, self.sense_obstacle_left_2, out1, out2)

    def if_obstacle_right_2(self, out1, out2):
        return partial(if_then_else, self.sense_obstacle_right_2, out1, out2)


    def if_out_of_bounds_up_1(self, out1, out2):
        return partial(if_then_else, self.sense_out_of_bounds_up_1, out1, out2)

    def if_out_of_bounds_down_1(self, out1, out2):
        return partial(if_then_else, self.sense_out_of_bounds_down_1, out1, out2)

    def if_out_of_bounds_left_1(self, out1, out2):
        return partial(if_then_else, self.sense_out_of_bounds_left_1, out1, out2)

    def if_out_of_bounds_right_1(self, out1, out2):
        return partial(if_then_else, self.sense_out_of_bounds_right_1, out1, out2)

    def if_out_of_bounds_up_2(self, out1, out2):
        return partial(if_then_else, self.sense_out_of_bounds_up_2, out1, out2)

    def if_out_of_bounds_down_2(self, out1, out2):
        return partial(if_then_else, self.sense_out_of_bounds_down_2, out1, out2)

    def if_out_of_bounds_left_2(self, out1, out2):
        return partial(if_then_else, self.sense_out_of_bounds_left_2, out1, out2)

    def if_out_of_bounds_right_2(self, out1, out2):
        return partial(if_then_else, self.sense_out_of_bounds_right_2, out1, out2)


    def if_tail_up_1(self, out1, out2):
        return partial(if_then_else, self.sense_tail_up_1, out1, out2)

    def if_tail_down_1(self, out1, out2):
        return partial(if_then_else, self.sense_tail_down_1, out1, out2)

    def if_tail_left_1(self, out1, out2):
        return partial(if_then_else, self.sense_tail_left_1, out1, out2)

    def if_tail_right_1(self, out1, out2):
        return partial(if_then_else, self.sense_tail_right_1, out1, out2)

    def if_tail_up_2(self, out1, out2):
        return partial(if_then_else, self.sense_tail_up_2, out1, out2)

    def if_tail_down_2(self, out1, out2):
        return partial(if_then_else, self.sense_tail_down_2, out1, out2)

    def if_tail_left_2(self, out1, out2):
        return partial(if_then_else, self.sense_tail_left_2, out1, out2)

    def if_tail_right_2(self, out1, out2):
        return partial(if_then_else, self.sense_tail_right_2, out1, out2)


# This function places a food item in the environment
def placeFood(snake):
    food = []
    while len(food) < NFOOD:
        potentialfood = [random.randint(1, (YSIZE - 2)), random.randint(1, (XSIZE - 2))]
        if not (potentialfood in snake.body) and not (potentialfood in food):
            food.append(potentialfood)
    snake.food = food  # let the snake know where the food is
    return (food)


snake = SnakePlayer()


# Function displaying a strategy run in terminal
def displayStrategyRun(strategy, g):
    global snake
    global pset

    curses.initscr()
    win = curses.newwin(YSIZE, XSIZE, 0, 0)
    win.keypad(1)
    curses.noecho()
    curses.curs_set(0)
    win.border(0)
    win.nodelay(1)
    win.timeout(120)

    snake._reset()
    food = placeFood(snake)

    for f in food:
        win.addch(f[0], f[1], '@')

    timer = 0
    collided = False

    # Check if max score reached
    while not collided and not timer == ((2 * XSIZE) * YSIZE) and (snake.score < MAX_SCORE):

        # Set up the display
        win.border(0)
        win.addstr(0, 2, ' Score: ' + str(snake.score) + ' ')
        win.addstr(YSIZE - 1, 3, ' Gen: ' + str(g) + ' ')


        win.getch()

        # Run strategy
        strategy()

        snake.updatePosition()

        if snake.body[0] in food:
            snake.score += 1
            for f in food: win.addch(f[0], f[1], ' ')
            if (snake.score < MAX_SCORE - 1):
                food = placeFood(snake)
            for f in food: win.addch(f[0], f[1], '@')
            timer = 0
        else:
            last = snake.body.pop()
            win.addch(last[0], last[1], ' ')
            timer += 1  # timesteps since last eaten
        win.addch(snake.body[0][0], snake.body[0][1], 'o')

        collided = snake.snakeHasCollided()
        hitBounds = (timer == ((2 * XSIZE) * YSIZE))

    curses.endwin()

    return snake.score,


# This function runs a strategy run for evaluation returning the snake score
def runGame(strategy):
    global snake

    snake._reset()
    food = placeFood(snake)
    timer = 0

    totalScore = 0

    # Check if max score reached
    while not snake.snakeHasCollided() and not timer == XSIZE * YSIZE and (snake.score < MAX_SCORE):

        strategy()

        snake.updatePosition()

        if snake.body[0] in food:
            snake.score += 1
            if (snake.score < MAX_SCORE - 1):
                food = placeFood(snake)
            timer = 0
        else:
            snake.body.pop()
            timer += 1  # timesteps since last eaten

    return snake.score,

# Function evaluation for an individual returning the avarage score after 5 runs
def evalArtificialSnake(individual):
    strategy = gp.compile(individual, pset)
    totalScore = 0

    for run in range(5):
        totalScore += runGame(strategy)[0]

    return totalScore/5,


snake = SnakePlayer()

pset = gp.PrimitiveSet("MAIN", 0)

# List of all possible primitives. Uncomment to apply different combination of senses.

pset.addPrimitive(snake.if_food_up, 2)
pset.addPrimitive(snake.if_food_down, 2)
pset.addPrimitive(snake.if_food_left, 2)
pset.addPrimitive(snake.if_food_right, 2)

pset.addPrimitive(snake.if_obstacle_up_1, 2)
pset.addPrimitive(snake.if_obstacle_down_1, 2)
pset.addPrimitive(snake.if_obstacle_left_1, 2)
pset.addPrimitive(snake.if_obstacle_right_1, 2)
#pset.addPrimitive(snake.if_obstacle_up_2, 2)
#pset.addPrimitive(snake.if_obstacle_down_2, 2)
#pset.addPrimitive(snake.if_obstacle_left_2, 2)
#pset.addPrimitive(snake.if_obstacle_right_2, 2)

#pset.addPrimitive(snake.if_tail_up_1, 2)
#pset.addPrimitive(snake.if_tail_down_1, 2)
#pset.addPrimitive(snake.if_tail_left_1, 2)
#pset.addPrimitive(snake.if_tail_right_1, 2)
#pset.addPrimitive(snake.if_tail_up_2, 2)
#pset.addPrimitive(snake.if_tail_down_2, 2)
#pset.addPrimitive(snake.if_tail_left_2, 2)
#pset.addPrimitive(snake.if_tail_right_2, 2)

#pset.addPrimitive(snake.if_out_of_bounds_up_1, 2)
#pset.addPrimitive(snake.if_out_of_bounds_down_1, 2)
#pset.addPrimitive(snake.if_out_of_bounds_left_1, 2)
#pset.addPrimitive(snake.if_out_of_bounds_right_1, 2)
pset.addPrimitive(snake.if_out_of_bounds_up_2, 2)
pset.addPrimitive(snake.if_out_of_bounds_down_2, 2)
pset.addPrimitive(snake.if_out_of_bounds_left_2, 2)
pset.addPrimitive(snake.if_out_of_bounds_right_2, 2)

pset.addTerminal(snake.changeDirectionUp)
pset.addTerminal(snake.changeDirectionRight)
pset.addTerminal(snake.changeDirectionDown)
pset.addTerminal(snake.changeDirectionLeft)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

# Setup DEAP toolbox
toolbox = base.Toolbox()

toolbox.register("expr_init", gp.genFull, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr_init)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evalArtificialSnake)
toolbox.register("select", tools.selDoubleTournament, fitness_size=10, parsimony_size= 2, fitness_first=True)
##toolbox.register("select", tools.selTournament, tournsize=8)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=5)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

stats = tools.Statistics(key=lambda ind: ind.fitness.values)

stats.register("avg", numpy.mean)
stats.register("std", numpy.std)
stats.register("min", numpy.min)
stats.register("max", numpy.max)

logbook = tools.Logbook()
logbook.header = "gen", "evals", "std", "min", "avg", "max"

def main():
    global snake
    global pset

    start, end = 0,0
    # Specify the random seed
    SEED = 0

    # Set parameters for evolution
    POP_SIZE, NGEN, CXPB, MUTPB = 300, 1000, 0.4, 1.0

    random.seed(SEED)
    pop = toolbox.population(n=POP_SIZE)
    bestIndividual = 0

    start = time.time()

    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    for g in range(NGEN):
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2, in zip (offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

        del child1.fitness.values
        del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit


        pop[:] = offspring

        fits = [ind.fitness.values[0] for ind in pop]

        record = stats.compile(pop)

        logbook.record(gen=g, evals=len(invalid_ind), **record)
        print(logbook.stream)

        bestBestInCurrentPopulation = tools.selBest(pop, 1)[0]

        # Store the best individual yet
        if bestIndividual == 0 or bestBestInCurrentPopulation.fitness.values[0] >= bestIndividual.fitness.values[0]:
            bestIndividual = bestBestInCurrentPopulation


    end = time.time()

    # Store logbook in a csv file
    df_log = pd.DataFrame(logbook)
    df_log.to_csv('seed_' + str(SEED) + '_pop_' + str(POP_SIZE) + '_ngen_' + str(NGEN) + '_cxpb_' + str(CXPB) + '_mutpb_' + str(MUTPB) + '.csv', index=False)

    # Print the best individual
    print(bestIndividual)

    # Print the time taken to execute the algorithm
    print(end - start)

    bestStrategy = gp.compile(bestIndividual, pset)
    displayStrategyRun(bestStrategy, g)

if __name__ == "__main__":
    main()
