import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from collections import deque

from functions.game import play_step
from functions.utils import generate_board
from functions.utils import boards_to_array
from functions.players import trained_player

RECENT_SCORES_LEN = 200
DISCOUNT_FACTOR = .8
NUM_EPISODES = 1_000
SOLVED_SCORE = 120
MAX_STEPS = 41
EPS = .2

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class PolicyNetwork(nn.Module):
    def __init__(self, observation_space, action_space):
        super(PolicyNetwork, self).__init__()
        self.layer1 = nn.Linear(observation_space, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.relu2 = nn.ReLU()
        self.output_layer = nn.Linear(32, action_space)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.output_layer(x)
        x = self.softmax(x)
        
        return x

class StateValueNetwork(nn.Module):
    def __init__(self, observation_space):
        super(StateValueNetwork, self).__init__()
        self.layer1 = nn.Linear(observation_space, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.relu2 = nn.ReLU()
        self.output_layer = nn.Linear(32, 1)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.output_layer(x)
        
        return x

def train_netwrok(policy_network, stateval_network, policy_optimizer, stateval_optimizer, player, rl_player, won_percentage):
    recent_scores = deque(maxlen=RECENT_SCORES_LEN)
    rival = (player + 1) % 2
    for _ in tqdm(range(NUM_EPISODES)):
        if len(recent_scores) == RECENT_SCORES_LEN and np.mean(np.array(recent_scores) >= SOLVED_SCORE) >= won_percentage: break
        all_board, global_board = generate_board()
        next_action_basis = (None, None)
        if player == 1:
            i, j = next_action_basis
            next_action = rl_player(all_board, global_board, i, j)
            all_board, global_board, _, next_action_basis = play_step(all_board, global_board, next_action, player=rival)

        state = boards_to_array(all_board, global_board)
        done = False
        score = 0
        I = 1
        
        for step in range(MAX_STEPS):
            action, lp = trained_player(all_board, global_board, next_action_basis, policy_network, eps=EPS, training=True)
            
            all_board, global_board, reward, next_action_basis = play_step(all_board, global_board, action, player=player)
            new_state = boards_to_array(all_board, global_board)
            score += reward
            
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
            state_val = stateval_network(state_tensor)
            
            new_state_tensor = torch.from_numpy(new_state).float().unsqueeze(0).to(DEVICE)        
            new_state_val = stateval_network(new_state_tensor)
            
            done = reward != 1
            if not done:
                i, j = next_action_basis
                next_action = rl_player(all_board, global_board, i, j)
                all_board, global_board, other_reward, next_action_basis = play_step(all_board, global_board, next_action, player=rival)
                
                new_state = boards_to_array(all_board, global_board)
                if other_reward == 100: reward = -50
                else: reward = other_reward
                score += reward
                done = reward != 1
            
            if done: new_state_val = torch.tensor([0]).float().unsqueeze(0).to(DEVICE)
            val_loss = F.mse_loss(reward + DISCOUNT_FACTOR * new_state_val, state_val)
            val_loss *= I
            
            advantage = reward + DISCOUNT_FACTOR * new_state_val.item() - state_val.item()
            policy_loss = -lp * advantage
            policy_loss *= I
            
            policy_optimizer.zero_grad()
            policy_loss.backward(retain_graph=True)
            policy_optimizer.step()
            
            stateval_optimizer.zero_grad()
            val_loss.backward()
            stateval_optimizer.step()
            
            if done: break
            state = new_state
            I *= DISCOUNT_FACTOR
        
        recent_scores.append(score)
