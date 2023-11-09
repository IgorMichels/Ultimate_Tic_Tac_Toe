from utils import get_avaliabe_spaces
from game_display import print_board
from random import choice

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
    if i is None:
        avaliable_subgrids = get_avaliabe_spaces(global_board)
        i, j = choice(avaliable_subgrids)
    
    board = all_board[i][j]
    avaliable_spaces = get_avaliabe_spaces(board)
    k, l = choice(avaliable_spaces)
    return i, j, k, l
