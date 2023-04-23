import sys
import copy
import random
from collections import deque

import numpy as np
import pandas as pd
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import Categorical
from torch.optim import Adam
sys.path.append('..')
from graph_test_new import get_data, MyHetero
import wandb
from torchviz import make_dot

class PolicyNet(torch.nn.Module):
    def __init__(self, data, hidden_dim=128):
        super().__init__()
        self.action_size = data['word'].x.size()[0]
        self.emb_dim = data['word'].x.size()[1] + data['spot'].x.size()[1]
        self.hidden_dim = hidden_dim
        self.l1 = nn.Linear(self.action_size, self.hidden_dim)
        self.l2 = nn.Linear(self.hidden_dim, self.action_size)

    def forward(self, x):
        x = self.l2(F.relu(self.l1(x)))
        #x = self.l2(x)
        x = F.softmax(x, dim=1)
        return x

class ValueNet(torch.nn.Module):
    def __init__(self, data, hidden_dim=128):
        super().__init__()
        self.action_size = data['word'].x.size()[0]
        self.hidden_dim = hidden_dim
        self.l1  = nn.Linear(self.action_size, self.hidden_dim)
        self.l2 = nn.Linear(self.hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x

class Agent:
    def __init__(self, data, device):
        self.gamma = 0.98
        self.lr_pi = 2e-4
        self.lr_v = 5e-4
        self.action_size = data['word'].x.size()[1]

        self.pi =  PolicyNet(data).to(device)
        self.v = ValueNet(data).to(device)
        self.optimizer_pi = Adam(self.pi.parameters(), self.lr_pi)
        self.optimizer_v = Adam(self.v.parameters(), self.lr_v)
        self.data = data
        self.device = device

        self.count_dist = np.load('/home/yamanishi/project/trip_recommend/model/RL/count.npy')

    def get_action(self, state, episode):
        probs = self.pi(state.unsqueeze(0))
        probs = probs[0]
        m = Categorical(probs)
        action = m.sample().item()
        return action, probs[action]

    def update(self, state, action_prob, reward, next_state, done):
        target = reward + self.gamma * self.v(next_state.unsqueeze(0))
        target.detach()
        v = self.v(state.unsqueeze(0))
        loss_fn = nn.MSELoss()
        loss_v = loss_fn(v, target)

        delta = target-v
        loss_pi = -torch.log(action_prob)*delta.item()
        self.optimizer_v.zero_grad()
        self.optimizer_pi.zero_grad()
        loss_v.backward()
        loss_pi.backward()
        self.optimizer_v.step()
        self.optimizer_pi.step()

class POIEnv:
    def __init__(self, data, device):
        self.device = device
        self.original_data = data
        self.data = data
        self.model = MyHetero(self.data.x_dict, self.data.edge_index_dict, hidden_channels=128, out_channels=1, out_dim=512,multi=False).to(device)
        self.model.load_state_dict(torch.load('../../data/model/model_new.pth'))
        self.model.requires_grad=False

        self.spot_size = self.data['spot'].x.size()[0]

        node_idx = torch.randint(low=0, high=self.spot_size, size=(1,))[0]
        self.node_idx = node_idx
        self.state = torch.zeros(self.data['word'].x.size()[0]).to(self.device)
        index=self.data.edge_index_dict['spot','relate','word'][1][self.data.edge_index_dict['spot','relate','word'][0]==node_idx]
        self.state[index]=1
        self.word_num = 15

        self.current_edge = self.data.edge_index_dict['spot','relate','word']
        self.current_rev_edge = self.data.edge_index_dict['word','revrelate','spot']
        with torch.no_grad():
            self.current_reward = self.model(self.data.x_dict, self.data.edge_index_dict)[node_idx]


    def reset(self):
        self.data = self.original_data
        node_idx = torch.randint(low=0, high=self.spot_size, size=(1,))[0]
        self.node_idx = node_idx
        self.state = torch.zeros(self.data['word'].x.size()[0]).to(self.device)
        index=self.data.edge_index_dict['spot','relate','word'][1][self.data.edge_index_dict['spot','relate','word'][0]==node_idx]
        self.state[index]=1
        self.word_num = 15

        self.current_edge = self.data.edge_index_dict['spot','relate','word']
        self.current_rev_edge = self.data.edge_index_dict['word','revrelate','spot']

        self.current_reward = self.model(self.data.x_dict, self.data.edge_index_dict)[self.node_idx]
        return self.state

    @torch.no_grad()
    def step(self, action):
        sw = torch.tensor([[self.node_idx], [action]]).to(self.device)
        ws = torch.tensor([[action], [self.node_idx]]).to(self.device)
        self.current_edge = torch.cat([self.current_edge, sw], axis=1)
        self.current_rev_edge = torch.cat([self.current_rev_edge, ws], axis=1)
        self.data['word','revrelate','spot'].edge_index = self.current_rev_edge
        self.data['spot','relate','word'].edge_index = self.current_edge
        self.state[action]=1
        next_state = self.state
        new_current_reward = self.model(self.data.x_dict, self.data.edge_index_dict)[self.node_idx]
        reward = new_current_reward - self.current_reward
        self.current_reward = new_current_reward

        self.word_num+=1
        if self.word_num==20:
            done=True
        else:
            done=False

        info=None

        return next_state, reward, done, info

    

        
    
def simulation(device):
    episodes = 1000000
    sync_interval=20

    data_dir = '/home/yamanishi/project/trip_recommend/data/jalan/spot/'
    df = pd.read_csv(os.path.join(data_dir,'experience_light.csv'))
    data = get_data(df).to(device)
    env = POIEnv(data, device) #TODO:実装
    agent = Agent(data, device)
    reward_history = []
    step=0
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            probs = agent.pi(state.unsqueeze(0))
            probs = probs[0]
            m = Categorical(probs)
            action = m.sample().item()
            action_prob = probs[action]
            #return action, probs[action]
            #action, action_prob = agent.get_action(state, episode)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action_prob, reward, next_state, done)
            wandb.log({'step':step, 'reward':reward})
            state = next_state
            total_reward+=reward

        if episode%sync_interval==0:
            agent.sync_action()
        wandb.log({'episode':episode,'total_reward':total_reward})
        print(total_reward)
        reward_history.append(total_reward)

if __name__=='__main__':
    device='cpu'
    #wandb.init(project='reinforcement poi', name='actor-critic')
    torch.autograd.set_detect_anomaly(True)
    simulation(device)
            