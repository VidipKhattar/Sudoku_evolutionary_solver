from random import *
from copy import deepcopy
from time import time as t


def evolution_func(board):
    """
    Function to conduct the evolution algorithm.

    Parameters:
    :param board -- sudoku board that is given to be solved

    """

    # start timer to check algorithm performance
    time = t()
    # count of how many times the same fitness appears in a row
    min_fitness_repeat_count = 0

    # hold min fitness recorded (best fitness)
    curr_min_fitness = 999

    total_best_indiv_pop = 0
    total_best_fitness = 999

    # forms population
    population = create_population(board)

    # evaluate the starting population
    fitness_population = evaluate_population(population, board)
    evolutions_done = 0

    for gen in range(NUMBER_GENERATION):
        # mating pool contains the top half of population with the best fitness
        top_population = select_population(population, fitness_population)

        # creates children population by crossing over parents for evolution
        children_population = crossover_population(top_population)

        # mutates the children and makes that the new population to be evolved
        population = mutate_population(children_population)

        # evaluates fitness of new population with evolved children
        fitness_population = evaluate_population(population, board)

        # retrieves best individual puzzle from the population along with its fitness score
        best_indiv_pop, curr_best_fitness = best_population(population, fitness_population)

        # checks if the best current best fitness of population is better than the total best fit
        if curr_best_fitness < total_best_fitness:
            total_best_indiv_pop = best_indiv_pop
            total_best_fitness = curr_best_fitness

        # if fitness is 0, means puzzle is solved and ends
        if curr_best_fitness == 0:
            break

        # if best fitness is same as last generations best fitness, add one to repeat fitness
        # counter
        if curr_best_fitness == curr_min_fitness:
            min_fitness_repeat_count += 1
        else:
            min_fitness_repeat_count = 0

        curr_min_fitness = curr_best_fitness

        # if stuck at certain fitness level, start over with new initial population
        if min_fitness_repeat_count > 30:
            min_fitness_repeat_count = 0
            population = create_population(board)

        # print out current generation's best fitness to keep track of search
        # print("Curr best fitness:", curr_best_fitness, "Current evolution:", evolutions_done)
        evolutions_done += 1

    total_fitness = 0
    # used for average fitness of last evolved population
    for i in fitness_population:
        total_fitness += i

    fitness_average = total_fitness / POPULATION_SIZE
    final_time = t() - time
    # return performance results
    return total_best_indiv_pop, total_best_fitness, evolutions_done, fitness_average, final_time


def best_population(population, fitness_population):
    """

    :param population: population needed to be searched over
    :param fitness_population: fitness of each table of population
    :return: population and fitness population tuple with the lowest (best) fitness score
    """
    return sorted(zip(population, fitness_population), key=lambda ind_fit: ind_fit[1])[0]


def create_population(board):
    """
    creates population from the given board
    population_size is the number of boards created in the population

    :param board: board from text file to be used to create board populations
    :return: array of individual board as a population list
    """
    return [create_ind(board) for _ in range(POPULATION_SIZE)]


def create_ind(initial_board):
    """
    creates individuals board to be added to the population
    :param initial_board: initial board to compared too
    :return: randomised board built of the initial board
    """
    random_board = []
    # loop through each block in board, fill empty spaces randomly
    for block in initial_board:
        random_block = populate_block(block)
        random_board.append(random_block)
    return random_board


def populate_block(block):
    """
    adds random unique number values to empty cells in the block
    :param block: block from intial board to add random values in its empty spaces
    :return: randomised block in a list form
    """
    block_values = []
    new_block = []
    # creates empty block
    for i in range(3):
        new_block.append([0, 0, 0])

    # creates list of values from a 3 by 3 block
    for row in block:
        for cell in row:
            block_values.append(cell)

    # choice list contains values of available number to be randomly put in the block
    # since only one number can appear once in the block
    populate_values = list({1, 2, 3, 4, 5, 6, 7, 8, 9} - set(block_values))
    for row in range(3):
        for cell in range(3):
            # if cell val = 0 then populate that cell with unique number value
            if block[row][cell] == 0:
                new_cell = choice(populate_values)
                populate_values.remove(new_cell)
                new_block[row][cell] = new_cell
            else:
                new_block[row][cell] = 0
    return list(new_block)


def evaluate_population(population, board):
    """
    evaluates the fitness of each table in the given population according to the given board

    :param population: population needed to be searched over
    :param board:
    :return: array containing fitness of each individual board in the population
    """
    return [fitness_ind(add_boards(individual, board)) for individual in population]


def fitness_ind(board):
    """

    :param board: board to find the fitness of
    :return: int value of fitness based on repetition of numbers in rows and columns
    """

    # only checks row and columns since already confirmed that block all have unique numbers
    rows = []
    cols = []
    for i in range(9):
        rows.append([])
        cols.append([])
    for block_num in range(9):
        for row_count in range(3):
            for cell_count in range(3):
                board_cell = board[block_num][row_count][cell_count]
                rows[row_count + 3 * int(block_num / 3)] = rows[row_count + 3 * int(
                    block_num / 3)] + [board_cell]
                cols[(block_num % 3) * 3 + cell_count] = cols[(block_num % 3) * 3 + cell_count] + [
                    board_cell]
    return fitness_row_col(rows, 1) + fitness_row_col(cols, 1)


def fitness_row_col(row_col, conflict_val):
    """
    # evaluate either rows or columns of a board
    :param row_col: row or column to be evaluate
    :param conflict_val: conflict value to be added to fitness
    :return: int value of number of conflicts in the row or column
    """

    row_col_fitness = 0
    for val in row_col:
        row = list(val)
        row.sort()
        for x in range(len(row) - 1):
            if row[x] == row[x + 1]:
                row_col_fitness += conflict_val
    return row_col_fitness


def select_population(population, fitness_population):
    """
    Selects top percentage of boards in the population with the best fitness
    :param population: population needed to be searched over
    :param fitness_population: array of fitness of each board in the element
    :return: list of top percentage fitness populations
    """
    # sort the population by fitness
    sorted_population = sorted(zip(population, fitness_population), key=lambda pop_fit: pop_fit[1])

    # select the top 25% (given by truncation rate) of the population
    return [individual for individual, fitness in
            sorted_population[:int(POPULATION_SIZE * TRUNCATION_RATE)]]


def crossover_population(population):
    """
    randomly chooses two parents from the fittest population and cross them to repopulate the
    population
    :param population: population needed to be searched over
    :return: list of children
    board population
    """
    return [crossover_ind(choice(population), choice(population)) for _ in
            range(POPULATION_SIZE)]


def crossover_ind(parent1, parent2):
    """
    crosses over and breeds parent boards

    :param parent1: parent 1 board from the previous gens population
    :param parent2: parent 2 board from the previous gens population
    :return: board in the a list containing random blocks from both parents
    """
    return [choice(block) for block in zip(parent1, parent2)]


def mutate_population(population):
    """
    mutates the children population to evolve them
    :param population: population to be mutates
    :return: list of boards in the population which are individually mutated
    """
    return [mutate_ind(individual) for individual in population]


def mutate_ind(individual):
    """
    mutates each individual board in the population
    :param individual:
    :return: board block in a list form
    """
    return [mutate_block(block) for block in individual]


def mutate_block(block):  # completely mix up a single block
    if random() > MUTATION_RATE:
        return deepcopy(block)  # if not mutate, return unaffected block
    new_block = deepcopy(block)

    coords = []
    for i in range(len(new_block)):
        for j in range(len(new_block[0])):
            if (new_block[i][j] != 0):
                coords.append((i, j))  # all the possible indices that can be changed

    coord1 = choice(coords)  # choose two
    coords.remove(coord1)
    coord2 = choice(coords)
    new_block[coord1[0]][coord1[1]], new_block[coord2[0]][coord2[1]] = new_block[coord2[0]][
                                                                           coord2[1]], \
                                                                       new_block[coord1[0]][coord1[
                                                                           1]]  # swap two numbers in a block

    return new_block


def add_boards(given_board, pop_board):
    """
    adds two boards in the form of matrices together
    :param given_board: individual board from the population
    :param pop_board: initial board
    :return: tuple of the population and given board
    """
    return [add_blocks(b) for b in zip(given_board, pop_board)]


def add_blocks(block):
    """
    adds two blocks together from the added boards
    :param block: blocks in the tuple of matrices
    :return: list of added blocks from the combined boards
    """
    return [add_rows(row) for row in zip(block[0], block[1])]


def add_rows(row):
    """
    adds 2 list together represented a row in a block
    :param row: row in a block
    :return: list of summed rows
    """
    return [sum(x) for x in zip(row[0], row[1])]


# I/O operations

def read_grid(name):
    """

    :param name: name of file with board values
    :return: list of strings containing contents of grid values
    """
    grid_file = open(name, 'r')
    string_board = grid_file.readlines()
    for c in range(0, len(string_board) - 1):
        string_board[c] = string_board[c][:len(string_board[c]) - 1]
    grid_file.close()
    return string_board


def create_board(contents):
    """
    forms board used to  perform algorithm
    :param contents: readable contents from board file
    :return: board in the form of  a 3-d list
    """
    board = []
    # starts one line later to avoid border line
    for i in range(0, len(contents) - 1, 3 + 1):
        # combine first 3 lines together
        horizontal_blocks = list(zip(contents[i], contents[i + 1], contents[i + 2]))
        for j in range(0, len(horizontal_blocks) - 1, 4):
            # create a block from the given row
            block = create_block(
                list(zip(horizontal_blocks[j], horizontal_blocks[j + 1], horizontal_blocks[j + 2])))
            board.append(block)
    return board


def create_block(block):
    """
    :param block: block in the board being formed
    :return: int value of the board
    """
    block_list = []
    for i in range(3):
        row_list = []
        for j in range(3):
            if block[i][j] == ".":
                row_list.append(0)
            else:
                row_list.append(int(block[i][j]))
        block_list.append(row_list)
    return block_list


def print_board(board):
    """
    prints board in a readable way when performing operations
    :param board: board being printed
    """
    # loop through row of blocks
    for i in range(0, len(board) - 1, 3):
        # print block
        print_row(list(zip(board[i], board[i + 1], board[i + 2])))


def print_row(row):
    """
    prints row of a single block
    :param row: row  of block being outputted
    """
    for line in row:
        for i in range(3):
            print(line[i], end=' ')
        print()
    print()


NUMBER_GENERATION = 2000
TRUNCATION_RATE = 0.20
MUTATION_RATE = 0.16
POPULATION_SIZE = 10

# random.seed(20)
# random.seed(40)
# random.seed(60)
# random.seed(80)
# random.seed(100)

# create given boards
board1 = create_board(read_grid("Grid1.txt"))
board2 = create_board(read_grid("Grid2.txt"))
board3 = create_board(read_grid("Grid3.txt"))



for i in range(4):
    if i == 0:
        POPULATION_SIZE = 10
    elif i == 1:
        POPULATION_SIZE = 100
    elif i == 2:
        POPULATION_SIZE = 1000
    elif i == 3:
        POPULATION_SIZE = 10000
    print("population sizes is now:", POPULATION_SIZE)
    print()
    evolve1 = evolution_func(board1)
    print("Board 1 has been evolved")

    evolve2 = evolution_func(board2)
    print("Board 2 has been evolved")

    evolve3 = evolution_func(board3)
    print("Board 3 has been evolved")

    print("Board 1 results:")
    print()
    print_board(add_boards(evolve1[0], board1))
    print("Best board fitness:", evolve1[1], "Num generations:", evolve1[2], "Time taken:", evolve1[4])
    print()

    print("Board 2 results:")
    print()
    print_board(add_boards(evolve2[0], board2))
    print("Best board fitness:", evolve2[1], "Num generations:", evolve2[2], "Time taken:", evolve2[4])
    print()

    print("Board 3 results:")
    print()
    print_board(add_boards(evolve3[0], board3))
    print("Best board fitness:", evolve3[1], "Num generations:", evolve3[2], "Time taken:", evolve3[4])
    print()


