from utils import get_avaliabe_spaces, boards_to_array, get_possible_moves, move_to_idx
from game_display import print_board
from random import choice

import numpy as np

def human_player(all_board, global_board, i, j):
    print_board(all_board, global_board)
    k, l = None, None
    while True:
        try:
            while i is None or global_board[i][j] != [1, 0, 0]:
                i, j = map(int, input('Which subgrid do you wanna pick? Input as two integers with an space between then, like (0-2) (0-2)\n').split())

            break
        except ValueError:
            print('Invalid values')
        except IndexError:
            print('Invalid values')
            i, j = None, None
            
    while True:
        try:
            while k is None or all_board[i][j][k][l] != [1, 0, 0]:
                k, l = map(int, input(f'Which space from grid ({i}, {j}) do you wanna pick? Input as two integers with an space between then, like (0-2) (0-2)\n').split())

            break
        except ValueError:
            print('Invalid values')
        except IndexError:
            print('Invalid values')
            i, j = None, None
    
    return i, j, k, l

def random_player(all_board, global_board, i, j):
    if i is None or global_board[i][j] != [1, 0, 0]:
        avaliable_subgrids = get_avaliabe_spaces(global_board)
        i, j = choice(avaliable_subgrids)
    
    board = all_board[i][j]
    avaliable_spaces = get_avaliabe_spaces(board)
    k, l = choice(avaliable_spaces)
    return i, j, k, l

def trained_player(all_board, global_board, i, j, agent, eps):
    state = boards_to_array(all_board, global_board)
    possible_moves = get_possible_moves(all_board, global_board, i, j)

    probs = agent(state)
    probs = probs.numpy().flatten()
    probs = probs * possible_moves
    probs = probs / sum(probs)
    if np.random.random() < eps: move = np.random.choice(81, p=probs)
    else: move = np.argmax(probs)
    i, j, k, l = move_to_idx(move)
    return i, j, k, l
