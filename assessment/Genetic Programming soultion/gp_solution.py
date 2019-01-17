# This code defines the agent (as in the playable version) in a way that can be called and executed from an evolutionary algorithm.
# The code is partial and will not execute. You need to add to the code to create an evolutionary algorithm that evolves and executes a snake agent.
import curses
import random
import operator
import copy
import numpy
from functools import partial

# from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

S_RIGHT, S_LEFT, S_UP, S_DOWN = 0, 1, 2, 3
XSIZE, YSIZE = 14, 14
NFOOD = 1  # NOTE: YOU MAY NEED TO ADD A CHECK THAT THERE ARE ENOUGH SPACES LEFT FOR THE FOOD (IF THE TAIL IS VERY LONG)


def if_then_else(condition, out1, out2):
    out1() if condition() else out2()


# This class can be used to create a basic player object (snake agent)
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

    def getAheadLocation(self):
        self.ahead = [self.body[0][0] + (self.direction == S_DOWN and 1) + (self.direction == S_UP and -1),
                      self.body[0][1] + (self.direction == S_LEFT and -1) + (self.direction == S_RIGHT and 1)]

    def updatePosition(self):
        self.getAheadLocation()
        self.body.insert(0, self.ahead)

    ## You are free to define more sensing options to the snake
    def changeDirectionUp(self):
        self.direction = S_UP

    def changeDirectionRight(self):
        self.direction = S_RIGHT

    def changeDirectionDown(self):
        self.direction = S_DOWN

    def changeDirectionLeft(self):
        self.direction = S_LEFT

    def snakeHasCollided(self):
        return self.is_obstacle_location(self.body[0])

    def is_obstacle_location(self, location):
        return not (location[0] > 0 and location[0] < (YSIZE - 1) and location[1] > 0 and location[1] < (
                XSIZE - 1) and location not in self.body[1:])

    def is_out_of_bounds_location(self, location):
        return not (location[0] > 0 and location[0] < (YSIZE - 1) and location[1] > 0 and location[1])

    def is_tail_location(self, location):
        return not (XSIZE - 1) and location not in self.body[1:]

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


    def sense_wall_ahead(self):
        self.getAheadLocation()
        return (self.ahead[0] == 0 or self.ahead[0] == (YSIZE - 1) or self.ahead[1] == 0 or self.ahead[1] == (
                XSIZE - 1))

    def sense_tail_ahead(self):
        self.getAheadLocation()
        return self.ahead in self.body

    def sense_food_ahead(self):
        self.getAheadLocation()
        return self.ahead in self.food

    def sense_food_up(self):
        return self.food[0][0] < self.body[0][0]

    def sense_food_down(self):
        return self.food[0][0] > self.body[0][0]

    def sense_food_left(self):
        return self.food[0][1] < self.body[0][1]

    def sense_food_right(self):
        return self.food[0][1] > self.body[0][1]

    # PrimitiveSet
    def if_wall_ahead(self, out1, out2):
        return partial(if_then_else, self.sense_wall_ahead, out1, out2)

    def if_tail_ahead(self, out1, out2):
        return partial(if_then_else, self.sense_tail_ahead, out1, out2)

    def if_food_ahead(self, out1, out2):
        return partial(if_then_else, self.sense_food_ahead, out1, out2)

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


    def if_food_up(self, out1, out2):
        return partial(if_then_else, self.sense_food_up, out1, out2)

    def if_food_down(self, out1, out2):
        return partial(if_then_else, self.sense_food_down, out1, out2)

    def if_food_left(self, out1, out2):
        return partial(if_then_else, self.sense_food_left, out1, out2)

    def if_food_right(self, out1, out2):
        return partial(if_then_else, self.sense_food_right, out1, out2)


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


# This outline function is the same as runGame (see below). However,
# it displays the game graphically and thus runs slower
# This function is designed for you to be able to view and assess
# your strategies, rather than use during the course of evolution
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
    while not collided and not timer == ((2 * XSIZE) * YSIZE):

        # Set up the display
        win.border(0)
        win.addstr(0, 2, ' Score: ' + str(snake.score) + ' ')
        win.addstr(YSIZE - 1, 3, ' Gen: ' + str(g) + ' ')




        win.getch()

        ## EXECUTE THE SNAKE'S BEHAVIOUR HERE ##
        strategy()

        snake.updatePosition()

        if snake.body[0] in food:
            snake.score += 1
            for f in food: win.addch(f[0], f[1], ' ')
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


# This outline function provides partial code for running the game with an evolved agent
# There is no graphical output, and it runs rapidly, making it ideal for
# you need to modify it for running your agents through the game for evaluation
# which will depend on what type of EA you have used, etc.
# Feel free to make any necessary modifications to this section.
def runGame(strategy):
    global snake

    snake._reset()
    food = placeFood(snake)
    timer = 0

    totalScore = 0

    while not snake.snakeHasCollided() and not timer == XSIZE * YSIZE:

        strategy()

        snake.updatePosition()

        if snake.body[0] in food:
            snake.score += 1
            totalScore += 50
            food = placeFood(snake)
            timer = 0
        else:
            snake.body.pop()
            timer += 1  # timesteps since last eaten
            totalScore += 1


    return snake.score,


snake = SnakePlayer()

pset = gp.PrimitiveSet("MAIN", 0)
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

pset.addPrimitive(snake.if_food_up, 2)
pset.addPrimitive(snake.if_food_down, 2)
pset.addPrimitive(snake.if_food_left, 2)
pset.addPrimitive(snake.if_food_right, 2)
pset.addTerminal(snake.changeDirectionUp)
pset.addTerminal(snake.changeDirectionRight)
pset.addTerminal(snake.changeDirectionDown)
pset.addTerminal(snake.changeDirectionLeft)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Attribute generator
toolbox.register("expr_init", gp.genFull, pset=pset, min_=1, max_=2)

# Structure initializers
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr_init)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def evalArtificialSnake(individual):
    strategy = gp.compile(individual, pset)
    totalScore = 0

    for run in range(5):
        totalScore += runGame(strategy)[0]

    return totalScore/5,


toolbox.register("evaluate", evalArtificialSnake)
toolbox.register("select", tools.selDoubleTournament, fitness_size=10, parsimony_size= 2, fitness_first=True)
#toolbox.register("select", tools.selTournament, tournsize=8)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=5)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

def main():
    random.seed(0)
    global snake
    global pset

    pop = toolbox.population(n=300)

    NGEN, CXPB = 1000, 0.35

    print("Start of evolution")

    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    for g in range(NGEN):
        print("-- Generation %i --" % g)

        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2, in zip (offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

        del child1.fitness.values
        del child2.fitness.values

        for mutant in offspring:
            toolbox.mutate(mutant)
            del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        print("evaluated %i individuals" % len(invalid_ind))

        pop[:] = offspring

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

        if (g % 100 == 0):
            bestIndividual = tools.selBest(pop, 1)[0]
            bestStrategy = gp.compile(bestIndividual, pset)
            displayStrategyRun(bestStrategy, g)



main()
print("done")
