import numpy as np
import tensorflow as tf

def generate_board():
    all_board = [
        [
            [
                [[1, 0, 0] for _ in range(3)]
            for _ in range(3)]
        for _ in range(3)]
    for _ in range(3)]

    global_board = [[[1, 0, 0] for _ in range(3)] for _ in range(3)]
    return all_board, global_board

def check_board(board):
    for i in range(3):
        if all(board[i][0] == board[i][j] for j in range(3)) and board[i][0] != [1, 0, 0]: return board[i][0]
        if all(board[0][i] == board[j][i] for j in range(3)) and board[0][i] != [1, 0, 0]: return board[0][i]
    
    if all(board[0][0] == board[i][i] for i in range(3)) and board[0][0] != [1, 0, 0]: return board[0][0]
    if all(board[2][0] == board[2 - i][i] for i in range(3)) and board[2][0] != [1, 0, 0]: return board[2][0]

    non_empty = 0
    for i in range(3):
        for j in range(3):
            non_empty += board[i][j] != [1, 0, 0]

    if non_empty == 9: return [0, 0, 0]

def check_big_board(all_board, global_board):
    for i in range(len(all_board)):
        for j in range(len(all_board[i])):
            board = all_board[i][j]
            result = check_board(board)
            if result is not None:
                global_board[i][j] = result

    result = check_board(global_board)
    if result is not None:
        if result == [0, 0, 0]: result = 'draw'
        elif result == [0, 1, 0]: result = 'X won'
        elif result == [0, 0, 1]: result = 'O won'
    return all_board, global_board, result

def get_avaliabe_spaces(board):
    avaliable_spaces = list()
    for k, row in enumerate(board):
        for l, col in enumerate(row):
            if col == [1, 0, 0]:
                avaliable_spaces.append([k, l])
        
    return avaliable_spaces

def boards_to_array(all_board, global_board):
    board = np.hstack([np.argmax(all_board, axis=-1).flatten(),
                       np.argmax(global_board, axis=-1).flatten()])
    # board = tf.convert_to_tensor(board, dtype=tf.float32)
    # board = tf.expand_dims(board, axis=0)
    return board

def move_to_idx(move):
    i = move // 27
    move = move % 27
    j = move // 9
    move = move % 9
    k = move // 3
    move = move % 3
    l = move % 3

    return i, j, k, l

def idx_to_move(i, j, k, l):
    move = 3 * k + l
    move += 9 * j
    move += 27 * i
    return move

def get_possible_moves(all_board, global_board, i, j):
    possible_moves = np.zeros(81)
    possible_subgrids = get_avaliabe_spaces(global_board)
    if i is not None and [i, j] in possible_subgrids:
        spaces = get_avaliabe_spaces(all_board[i][j])
        if len(spaces) > 0:
            for k, l in spaces: possible_moves[idx_to_move(i, j, k, l)] = 1
            return possible_moves
    
    for i, j in possible_subgrids:
        spaces = get_avaliabe_spaces(all_board[i][j])
        if len(spaces) > 0:
            for k, l in spaces: possible_moves[idx_to_move(i, j, k, l)] = 1
    
    return possible_moves
