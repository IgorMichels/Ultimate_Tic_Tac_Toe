import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from functions.game import play
from functions.players import random_player
from functions.players import trained_player

policy_X = torch.jit.load('models/policy_X.pt')
policy_O = torch.jit.load('models/policy_O.pt')

def rl_player_X(all_board, global_board, i, j, **kwargs):
    next_action_basis = (i, j)
    return trained_player(all_board, global_board, next_action_basis, policy_X)

def rl_player_O(all_board, global_board, i, j, **kwargs):
    next_action_basis = (i, j)
    return trained_player(all_board, global_board, next_action_basis, policy_O)

if __name__ == '__main__':
    GAMES = 100_000
    possible_results = ['X won', 'draw', 'O won']
    results = pd.DataFrame(columns=['X', 'O', 'X win rate', 'draw rate', 'O win rate'])
    scenarios = [['random', 'random'], ['trained', 'random'], ['random', 'trained'], ['trained', 'trained']]

    for p0, p1 in scenarios:
        if p0 == 'random': player_0 = random_player
        else: player_0 = rl_player_X

        if p1 == 'random': player_1 = random_player
        else: player_1 = rl_player_O

        games = GAMES if (p0 == 'random' or p1 == 'random') else 1
        scenario_results = [play(player_0=player_0, player_1=player_1) for _ in tqdm(range(games))]
        scenario_results = [p0, p1] + [scenario_results.count(r) / games for r in possible_results]
        results.loc[len(results)] = scenario_results

    bar_width = 0.25
    index = np.arange(len(results))
    fig, ax = plt.subplots(figsize=(10, 6))

    rects1 = ax.bar(index - bar_width, results['X win rate'], bar_width, label='X Win Rate')
    rects2 = ax.bar(index, results['draw rate'], bar_width, label='Draw Rate')
    rects3 = ax.bar(index + bar_width, results['O win rate'], bar_width, label='O Win Rate')

    ax.set_xlabel('Game')
    ax.set_ylabel('Result (%)')
    ax.set_title('Distribution of observed results')
    ax.set_xticks(index)
    ax.set_xticklabels([f'{x.capitalize()} vs. {o.capitalize()}' for x, o in zip(results['X'], results['O'])])
    ax.legend()

    plt.savefig('images/results_distribution_s.png')
