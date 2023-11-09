import numpy as np

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
    return np.hstack([np.argmax(all_board, axis=-1).flatten(),
                      np.argmax(global_board, axis=-1).flatten()])

if __name__ == '__main__':
    all_board, global_board = generate_board()
    global_board[0][0] = [0, 1, 0]
    global_board[0][1] = [0, 1, 0]
    global_board[0][2] = [0, 1, 0]
    all_board, global_board, result = check_big_board(all_board, global_board)
    print(result)
    print(boards_to_array(all_board, global_board).shape)
    