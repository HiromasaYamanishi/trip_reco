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
from torch.optim import Adam
sys.path.append('..')
from graph_test_new import get_data, MyHetero
import wandb

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        self.buffer.append(data)

    def __len__(self):
        return len(self.buffer)

    def get_batch(self):
        data = random.sample(self.buffer, self.batch_size)
        state = torch.stack([x[0] for x in data])
        action = torch.tensor([x[1] for x in data])
        reward = torch.tensor([x[2] for x in data])
        next_state = torch.stack([x[3] for x in data])
        done = torch.tensor([x[4] for x in data]).to(torch.int32)
        batch = {'state':state, 'action':action, 'reward':reward, 'next_state':next_state, 'done':done}
        return batch

class QNet(torch.nn.Module):
    def __init__(self, data, hidden_dim=128):
        super().__init__()
        self.action_size = data['word'].x.size()[0]
        self.emb_dim = data['word'].x.size()[1] + data['spot'].x.size()[1]
        self.hidden_dim = hidden_dim
        self.l1 = nn.Linear(self.action_size, self.hidden_dim)
        self.l2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.l3 = nn.Linear(self.hidden_dim, self.action_size)

    def forward(self, x):
        x = self.l1(x).relu()
        x = self.l2(x).relu()
        x = self.l3(x)
        return x

class DQNAgent:
    def __init__(self, data, episodes, device):
        self.episodes = episodes
        self.gamma = 0.98
        self.lr = 5e-4
        self.epsilon = np.linspace(0.5, 0.5, num=episodes)
        self.buffer_size = 10000
        self.batch_size = 32
        self.data = data
        self.device = device

        self.action_size = self.data['word'].x.size()[0]
        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.qnet = QNet(data).to(device)
        self.qnet_target = QNet(data).to(device)
        self.optimizer = Adam(self.qnet.parameters(), lr=self.lr)
        self.count_dist = np.load('/home/yamanishi/project/trip_recommend/model/RL/count.npy')

    def sync_action(self):
        self.qnet_target = copy.deepcopy(self.qnet)

    def get_action(self, state, episode):
        epsilon = self.epsilon[episode]
        if np.random.rand() < epsilon:
            return np.random.choice(self.action_size, p=self.count_dist)

        else:
            qs = self.qnet(state)
            return qs.argmax()

    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = self.replay_buffer.get_batch()
        reward = batch['reward'].to(self.device)
        qs = self.qnet(batch['state'].to(self.device))
        q = qs[torch.arange(self.batch_size), batch['action']]
        next_q = torch.max(self.qnet_target(batch['next_state'].to(self.device)),axis=1).values
        next_q.detach()
        target = reward +self.gamma * next_q
        loss = F.mse_loss(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

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
    agent = DQNAgent(data, episodes, device)
    reward_history = []
    step=0
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.get_action(state, episode)
            next_state, reward, done, info = env.step(action)

            agent.update(state, action, reward, next_state, done)
            wandb.log({'step':step, 'reward':reward})
            state = next_state
            total_reward+=reward

        if episode%sync_interval==0:
            agent.sync_action()
        wandb.log({'episode':episode,'total_reward':total_reward})
        print(total_reward)
        reward_history.append(total_reward)
        if episode%100==0:
            reward_mean = sum(reward_history)/100
            wandb.log({'episode':episode,'mean_reward':reward_mean})
            reward_history = []

if __name__=='__main__':
    device='cuda:6'
    wandb.init(project='reinforcement poi', name='train2')
    simulation(device)
            