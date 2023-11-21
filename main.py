import sys
import torch

from time import sleep
from functions.game import play
from functions.players import human_player
from functions.players import random_player
from functions.players import trained_player

policy_X = torch.jit.load('models/policy_X_0.pt')
policy_O = torch.jit.load('models/policy_O_0.pt')

def rl_player_X(all_board, global_board, i, j):
    next_action_basis = (i, j)
    return trained_player(all_board, global_board, next_action_basis, policy_X)

def rl_player_O(all_board, global_board, i, j):
    next_action_basis = (i, j)
    return trained_player(all_board, global_board, next_action_basis, policy_O)

if __name__ == '__main__':
    x_won, draw, o_won = 0, 0, 0
    player_0 = random_player
    player_0_type = 'random'
    player_1 = random_player
    player_1_type = 'random'
    verbose = 0.1
    games = 10

    for arg in sys.argv:
        if '-g' in arg:
            arg = arg.split('=')[1]
            games = int(arg)

        if '-v' in arg:
            arg = arg.split('=')[1]
            verbose = float(arg)

        if '-p0' in arg:
            arg = arg.split('=')[1]
            if arg == 'random':
                player_0 = random_player
                player_0_type = 'random'
            elif arg == 'rl':
                player_0 = rl_player_X
                player_0_type = 'trained'
            elif arg == 'h':
                player_0 = human_player
                player_0_type = 'human'

        if '-p1' in arg:
            arg = arg.split('=')[1]
            if arg == 'random':
                player_1 = random_player
                player_1_type = 'random'
            elif arg == 'rl':
                player_1 = rl_player_X
                player_1_type = 'trained'
            elif arg == 'h':
                player_1 = human_player
                player_1_type = 'human'
    
    for _ in range(games):
        header = f'Stats:\nX ({player_0_type}) won: {x_won}\ndraws: {draw}\nO ({player_1_type}) won: {o_won}'
        result = play(player_0=player_0, player_1=player_1, verbose=verbose, header=header)
        if result == 'X won': x_won += 1
        elif result == 'O won': o_won += 1
        else: draw += 1
        sleep(1)