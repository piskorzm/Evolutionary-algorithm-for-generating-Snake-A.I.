# This code defines the agent (as in the playable version) in a way that can be called and executed from an evolutionary algorithm. The code is partial and will not execute. You need to add to the code to create an evolutionary algorithm that evolves and executes a snake agent.
import curses
import random
import operator
import neat
import os
from functools import partial

RIGHT,LEFT,UP,DOWN = 0,1,2,3
XSIZE,YSIZE = 14,14
NFOOD = 1 # NOTE: YOU MAY NEED TO ADD A CHECK THAT THERE ARE ENOUGH SPACES LEFT FOR THE FOOD (IF THE TAIL IS VERY LONG)
SCORE = 0
GENERATION, MAX_FITNESS, BEST_GENOME = 0,0,0

# This class can be used to create a basic player object (snake agent)
class SnakePlayer(list):
    global RIGHT,LEFT,UP,DOWN
    global XSIZE, YSIZE

    def __init__(self):
        self.direction = RIGHT
        self.body = [ [4,10], [4,9], [4,8], [4,7], [4,6], [4,5], [4,4], [4,3], [4,2], [4,1],[4,0] ]
        self.score = 0
        self.ahead = []
        self.food = []

    def _reset(self):
        self.direction = RIGHT
        self.body[:] = [ [4,10], [4,9], [4,8], [4,7], [4,6], [4,5], [4,4], [4,3], [4,2], [4,1],[4,0] ]
        self.score = 0
        self.ahead = []
        self.food = []

    def getAheadLocation(self):
        self.ahead = [ self.body[0][0] + (self.direction == DOWN and 1) + (self.direction == UP and -1), self.body[0][1] + (self.direction == LEFT and -1) + (self.direction == RIGHT and 1)]

    def updatePosition(self):
        self.getAheadLocation()
        self.body.insert(0, self.ahead )

    def snakeHasCollided(self):
        self.hit = False
        if self.body[0][0] == 0 or self.body[0][0] == (YSIZE-1) or self.body[0][1] == 0 or self.body[0][1] == (XSIZE-1): self.hit = True
        if self.body[0] in self.body[1:]: self.hit = True
        return( self.hit )

    def sense_wall_ahead(self):
        self.getAheadLocation()
        return( self.ahead[0] == 0 or self.ahead[0] == (YSIZE-1) or self.ahead[1] == 0 or self.ahead[1] == (XSIZE-1) )

    def sense_food_ahead(self):
        self.getAheadLocation()
        return self.ahead in self.food

    def sense_tail_ahead(self):
        self.getAheadLocation()
        return self.ahead in self.body

    def getDistance(self, direction, location):
        head = self.body[0]
        distance = 0

        if (direction == UP and head[0] > location[0]):
            distance = head[0] - location[0]
        elif (direction == DOWN and head[0] < location[0]):
            distance = location[0] - head[0]
        elif (direction == LEFT and head[1] > location[1]):
            distance = head[0] - location[0]
        elif (direction == UP and head[1] < location[1]):
            distance = location[1] - head[1]

        return distance

    def getFoodDistance(self, direction):
        return self.getDistance(direction, self.food[0])

    def getClosestTailDistance(self, direction):
        if (direction == UP or direction == DOWN):
            distance = YSIZE
        elif (direction == LEFT or direction == RIGHT):
            distance = XSIZE

        tailInDirection = False

        for location in self.body:
            distanceToLocation = self.getDistance(direction, location)
            if (distanceToLocation != 0 and distanceToLocation < distance):
                tailInDirection = True
                distance = distanceToLocation

        if not tailInDirection:
            distance = 0

        return distance

    def getWallDistance(self, direction):
        head = self.body[0]
        distance = 0

        if (direction == UP):
            distance = head[0]
        elif (direction == DOWN):
            distance = YSIZE - head[0]
        elif (direction == LEFT):
            distance = head[1]
        elif (direction == RIGHT):
            distance = XSIZE - head[1]

        return distance

    def getInputsForNet(self):
        return (self.getFoodDistance(UP), self.getFoodDistance(DOWN), self.getFoodDistance(LEFT), self.getFoodDistance(RIGHT),
                self.getWallDistance(UP), self.getWallDistance(DOWN), self.getWallDistance(LEFT), self.getWallDistance(RIGHT),
                self.getClosestTailDistance(UP), self.getClosestTailDistance(DOWN), self.getClosestTailDistance(LEFT), self.getClosestTailDistance(RIGHT))

# This function places a food item in the environment
def placeFood(snake):
    food = []
    while len(food) < NFOOD:
        potentialfood = [random.randint(1, (YSIZE-2)), random.randint(1, (XSIZE-2))]
        if not (potentialfood in snake.body) and not (potentialfood in food):
            food.append(potentialfood)
    snake.food = food  # let the snake know where the food is
    return( food )


snake = SnakePlayer()


# This outline function is the same as runGame (see below). However,
# it displays the game graphically and thus runs slower
# This function is designed for you to be able to view and assess
# your strategies, rather than use during the course of evolution
def displayStrategyRun(net):
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
    while not collided and not timer == ((2*XSIZE) * YSIZE):

        # Set up the display
        win.border(0)
        win.addstr(0, 2, 'Score : ' + str(snake.score) + ' ')
        win.getch()

        output = net.activate(snake.getInputsForNet())
        snake.direction = output.index(max(output))

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
            timer += 1 # timesteps since last eaten
        win.addch(snake.body[0][0], snake.body[0][1], 'o')

        collided = snake.snakeHasCollided()
        hitBounds = (timer == ((2*XSIZE) * YSIZE))

    curses.endwin()
    raw_input("Press to continue...")

    return snake.score


# This outline function provides partial code for running the game with an evolved agent
# There is no graphical output, and it runs rapidly, making it ideal for
# you need to modify it for running your agents through the game for evaluation
# which will depend on what type of EA you have used, etc.
# Feel free to make any necessary modifications to this section.
def runGame(net):
    global snake

    totalScore = 0
    numberOfRuns = 5

    for i in range(numberOfRuns):
        snake._reset()
        food = placeFood(snake)
        timer = 0

        while not snake.snakeHasCollided() and not timer == XSIZE * YSIZE:

            output = net.activate(snake.getInputsForNet())
            snake.direction = output.index(max(output))

            snake.updatePosition()

            if snake.body[0] in food:
                snake.score += 100
                food = placeFood(snake)
                timer = 0
            else:
                snake.body.pop()
                snake.score += 1
                timer += 1  # timesteps since last eaten

        totalScore += snake.score

    return totalScore/numberOfRuns

def eval_genomes(genomes, config):

    i = 0
    global SCORE
    global GENERATION, MAX_FITNESS, BEST_GENOME


    GENERATION += 1
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        genome.fitness = runGame(net)
        print("Gen : %d Genome # : %d  Fitness : %f Max Fitness : %f"%(GENERATION,i,genome.fitness, MAX_FITNESS))
        if genome.fitness >= MAX_FITNESS:
            MAX_FITNESS = genome.fitness
            BEST_GENOME = genome
        SCORE = 0
        i+=1


def main():
    global snake
    global pset

    reandom_seed(1)

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         'config')

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    winner = pop.run(eval_genomes, 1000)

    net = neat.nn.FeedForwardNetwork.create(winner, config)

    print(winner)

    displayStrategyRun(net)



main()
