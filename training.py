import os
import sys
import torch
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

from tqdm import tqdm

from functions.game import play
from functions.players import trained_player
from functions.policy_train import train_netwrok
from functions.policy_train import PolicyNetwork
from functions.policy_train import StateValueNetwork

LEARNING_RATE = .005
GAMES = 2_000
EPOCHS = 250

PLAYER = 'X'
for arg in sys.argv:
    if '-p' in arg: PLAYER = arg.split('=')[-1].upper()

WON_PERCENTAGE = .85 if PLAYER == 'X' else .75
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

policy_X = PolicyNetwork(90, 81).to(DEVICE)
stateval_X = StateValueNetwork(90).to(DEVICE)

policy_opt_X = optim.SGD(policy_X.parameters(), lr=LEARNING_RATE)
stateval_opt_X = optim.SGD(stateval_X.parameters(), lr=LEARNING_RATE)

policy_O = PolicyNetwork(90, 81).to(DEVICE)
stateval_O = StateValueNetwork(90).to(DEVICE)

policy_opt_O = optim.SGD(policy_O.parameters(), lr=LEARNING_RATE)
stateval_opt_O = optim.SGD(stateval_O.parameters(), lr=LEARNING_RATE)

def rl_player_X(all_board, global_board, i, j, **kwargs):
    next_action_basis = (i, j)
    return trained_player(all_board, global_board, next_action_basis, policy_X)

def rl_player_O(all_board, global_board, i, j, **kwargs):
    next_action_basis = (i, j)
    return trained_player(all_board, global_board, next_action_basis, policy_O)

if __name__ == '__main__':
    os.system('clear')
    x_scores = list()
    o_scores = list()
    iterations = list()
    iteration = 0
    best_x_score = 0
    best_o_score = 0
    while iteration < EPOCHS:
        print(f'Iteration: {iteration + 1}')
        if PLAYER == 'X':
            train_netwrok(policy_X, stateval_X, policy_opt_X, stateval_opt_X, 0, rl_player_O, WON_PERCENTAGE)
        else:
            train_netwrok(policy_O, stateval_O, policy_opt_O, stateval_opt_O, 1, rl_player_X, WON_PERCENTAGE)

        if PLAYER == 'X':
            results = [play(player_0=rl_player_X) for _ in tqdm(range(GAMES))]
            x_score = np.mean(np.array(results) == 'X won')
            x_scores.append(x_score)
            print(f'Actual X win percentage: {x_score:.2%}')
            print(f'Best X win percentage: {best_x_score:.2%}')
            if x_score > best_x_score:
                model_scripted = torch.jit.script(policy_X)
                model_scripted.save(f'models/policy_{PLAYER}.pt')
                best_x_score = x_score
        else:
            results = [play(player_1=rl_player_O) for _ in tqdm(range(GAMES))]
            o_score = np.mean(np.array(results) == 'O won')
            o_scores.append(o_score)
            print(f'Actual O win percentage: {o_score:.2%}')
            print(f'Best O win percentage: {best_o_score:.2%}')
            if o_score > best_o_score:
                model_scripted = torch.jit.script(policy_O)
                model_scripted.save(f'models/policy_{PLAYER}.pt')
                best_o_score = o_score
        
        iteration += 1
        iterations.append(iteration)
        if PLAYER == 'X':
            plt.plot(iterations, x_scores)
            plt.plot([min(iterations), max(iterations)], [.6, .6], color='red', linestyle='--')
            plt.title('Iterations vs. X win rate')
        else:
            plt.plot(iterations, o_scores)
            plt.plot([min(iterations), max(iterations)], [.5, .5], color='red', linestyle='--')
            plt.title('Iterations vs. O win rate')
        
        plt.xlabel('Iteration')
        plt.ylabel('Win rate')

        plt.savefig(f'images/win_rate_{PLAYER}.png')
        plt.close()
        print()
