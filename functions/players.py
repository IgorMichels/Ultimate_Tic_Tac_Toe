from utils import get_avaliabe_spaces, boards_to_array, get_possible_moves, move_to_idx
from torch.distributions import Categorical
from game_display import print_board
from random import choice

import torch
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

def trained_player(all_board, global_board, next_action_basis, actor, eps=0, training=False):
    i, j = next_action_basis
    possible_moves = get_possible_moves(all_board, global_board, i, j)
    
    state = boards_to_array(all_board, global_board)
    state = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)

    probs = actor(state)
    probs = torch.nan_to_num(probs)
    probs = probs * torch.Tensor(possible_moves)
    s0 = torch.sum(probs)
    if s0.item() == 0:
        probs = (1 - probs) * torch.Tensor(possible_moves)
        s = torch.sum(probs)
        probs = probs / s
    else:
        probs = probs / s0
    
    m = Categorical(probs)
    if np.random.random() < eps or s0.item() == 0: action = m.sample()
    else: action = torch.argmax(probs)
    
    i, j, k, l = move_to_idx(action.item())

    if training: return (i, j, k, l), m.log_prob(action)
    else: return i, j, k, l

def evaluate_board(all_board, global_board, critic):
    state = boards_to_array(all_board, global_board)
    return critic(state)
