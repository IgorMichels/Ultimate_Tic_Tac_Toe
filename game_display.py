import numpy as np

SYMBOL_MAP = {0 : '   ', 1 : ' X ', 2 : ' O '}
BIG_SYMBOL_MAP = {0 : ['           ', '           ', '           ', '           ', '           '],
                  1 : ['   X   X   ', '    X X    ', '     X     ', '    X X    ', '   X   X   '],
                  2 : ['   OOOOO   ', '  O     O  ', '  O     O  ', '  O     O  ', '   OOOOO   ']}

def print_board(board, global_board):
    horizontal_div = '=' * 11 + '++' + '=' * 11 + '++' + '=' * 11
    for i, big_row in enumerate(board):
        lines = ['', '', '', '', '']
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
                        else: lines[2 * k] += '|' + SYMBOL_MAP[symbol]

                lines[1] += '---+---+---'
                lines[3] += '---+---+---'

            if j != 2:
                for k in range(5): lines[k] += '||'
        
        print('\n'.join(lines))
        if i != 2: print(horizontal_div)

if __name__ == '__main__':
    from utils import generate_board

    all_board, global_board = generate_board()
    all_board = np.array(all_board)
    global_board = np.array(global_board)
    all_board[0][0][0][0] = [0, 0, 1]
    global_board[1][1] = [0, 0, 1]
    global_board[2][1] = [0, 1, 0]
    global_board[2][2] = [0, 0, 0]
    print_board(all_board, global_board)