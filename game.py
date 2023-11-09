from utils import generate_board, check_big_board
from game_display import print_board
from players import random_player
from players import human_player
from copy import deepcopy

from time import time
from tqdm import tqdm
from os import system

import numpy as np

def play(all_board = None, global_board = None, curr_player = 0, i = None, j = None, player_0 = random_player, player_1 = random_player, verbose = False):
    players = [[0, 1, 0], [0, 0, 1]]
    result = None
    if all_board is None:
        all_board, global_board = generate_board()
        curr_player = 0
        i, j, = None, None
    else:
        all_board = deepcopy(all_board)
        global_board = deepcopy(global_board)
    
    while result is None:
        if verbose: system('clear')
        if curr_player == 0:
            if verbose: print('X player turn')
            i, j, k, l = player_0(all_board, global_board, i, j)
        else:
            if verbose: print('O player turn')
            i, j, k, l = player_1(all_board, global_board, i, j)
        
        all_board[i][j][k][l] = players[curr_player]
        i, j = k, l
        curr_player += 1
        curr_player %= 2
        all_board, global_board, result = check_big_board(all_board, global_board)

    if verbose: system('clear')
    if verbose: print_board(all_board, global_board)
    return result

def play_step(all_board, global_board, next_position, player = 0):
    if len(next_position) == 2:
        i, j = next_position
        i, j, k, l = random_player(all_board, global_board, i, j)
    else:
        i, j, k, l = next_position

    if player == 0: all_board[i][j][k][l] = [0, 1, 0]
    else: all_board[i][j][k][l] = [0, 0, 1]
    i, j = k, l
    all_board, global_board, result = check_big_board(all_board, global_board)

    reward = 1
    if result is not None:
        if result == 'draw': reward = 50
        elif result == 'X won' and player == 0: reward = 100
        elif result == 'O won' and player == 1: reward = 100
        else: reward = -50

    return all_board, global_board, reward, i, j

if __name__ == '__main__':
    SIMS = 10000
    t0 = time()
    all_board, global_board, i, j = None, None, None, None
    curr_player = 0
    # all_board, global_board = generate_board()
    # # i, j = None, None
    # i, j = 0, 0
    # global_board[i][j] = [0, 1, 0]
    # i, j = 1, 0
    # global_board[i][j] = [0, 1, 0]
    # i, j = 1, 1
    # global_board[i][j] = [0, 1, 0]
    # i, j = 2, 1
    # global_board[i][j] = [0, 1, 0]
    # i, j = 2, 2
    # all_board[i][j][2][2] = [0, 1, 0]
    # curr_player = 0
    # print_board(all_board, global_board)
    results = [play(all_board = all_board, global_board = global_board, curr_player = curr_player, i = i, j = j) for _ in tqdm(range(SIMS))]
    results, counts = np.unique(results, return_counts = True)
    expected_reward = np.dot(counts, np.array([-1, 2, 1])) / SIMS
    tf = time()
    print(results, counts, expected_reward, f'{tf - t0:.2f} seconds')
    