import numpy as np

SYMBOL_MAP = {0 : '   ', 1 : '\033[1;31m X \033[0m', 2 : '\033[1;34m O \033[0m'}
BIG_SYMBOL_MAP = {0 : ['           ', '           ', '           ', '           ', '           '],
                  1 : ['\033[1;31m   X   X   \033[0m',
                       '\033[1;31m    X X    \033[0m',
                       '\033[1;31m     X     \033[0m',
                       '\033[1;31m    X X    \033[0m',
                       '\033[1;31m   X   X   \033[0m'],
                  2 : ['\033[1;34m   OOOOO   \033[0m',
                       '\033[1;34m  O     O  \033[0m',
                       '\033[1;34m  O     O  \033[0m',
                       '\033[1;34m  O     O  \033[0m',
                       '\033[1;34m   OOOOO   \033[0m']}

def print_board(board, global_board, header=''):
    if header != '': print(header)
    horizontal_div = '\033[1;37m' + '=' * 11 + '++' + '=' * 11 + '++' + '=' * 11
    for i, big_row in enumerate(board):
        lines = ['\033[1;37m', '\033[1;37m', '\033[1;37m', '\033[1;37m', '\033[1;37m']
        for j, sub_board in enumerate(big_row):
            winner = np.argmax(global_board[i][j])
            if winner or not max(global_board[i][j]):
                for k in range(5):
                    lines[k] += BIG_SYMBOL_MAP[winner][k]
            else:
                for k, row in enumerate(sub_board):
                    for l, col in enumerate(row):
                        symbol = np.argmax(col)
                        if l == 0: lines[2 * k] += SYMBOL_MAP[symbol]
                        else: lines[2 * k] += '\033[1;37m|\033[0m' + SYMBOL_MAP[symbol]

                lines[1] += '\033[1;37m---+---+---\033[0m'
                lines[3] += '\033[1;37m---+---+---\033[0m'

            if j != 2:
                for k in range(5): lines[k] += '\033[1;37m||\033[0m'
        
        for k in range(5): lines[k] += '\033[0m'
        print('\n'.join(lines))
        if i != 2: print(horizontal_div)
