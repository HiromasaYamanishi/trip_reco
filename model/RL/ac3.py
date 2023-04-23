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

class Actor(nn.Module):
    def __init__(self, data):
        super(Actor, self).__init__()
        self.action_size = data['word'].x.size()[1]
        self.linear1 = nn.Linear(self.action_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, self.action_size)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        output = self.linear3(output)
        distribution = Categorical(F.softmax(output, dim=-1))
        return distribution


class Critic(nn.Module):
    def __init__(self, data):
        super(Critic, self).__init__()
        self.action_size = data['word'].x.size()[1]
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 1)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        value = self.linear3(output)
        return value



def trainIters(actor, critic, n_iters):
    optimizerA = optim.Adam(actor.parameters())
    optimizerC = optim.Adam(critic.parameters())
    for iter in range(n_iters):
        state = env.reset()
        log_probs = []
        values = []
        rewards = []
        masks = []
        entropy = 0
        env.reset()

        for i in count():
            env.render()
            state = torch.FloatTensor(state).to(device)
            dist, value = actor(state), critic(state)

            action = dist.sample()
            next_state, reward, done, _ = env.step(action.cpu().numpy())

            log_prob = dist.log_prob(action).unsqueeze(0)
            entropy += dist.entropy().mean()

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.tensor([reward], dtype=torch.float, device=device))
            masks.append(torch.tensor([1-done], dtype=torch.float, device=device))

            state = next_state

            if done:
                print('Iteration: {}, Score: {}'.format(iter, i))
                break


        next_state = torch.FloatTensor(next_state).to(device)
        next_value = critic(next_state)
        returns = compute_returns(next_value, rewards, masks)

        log_probs = torch.cat(log_probs)
        returns = torch.cat(returns).detach()
        values = torch.cat(values)

        advantage = returns - values

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        optimizerA.zero_grad()
        optimizerC.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        optimizerA.step()
        optimizerC.step()
    torch.save(actor, 'model/actor.pkl')
    torch.save(critic, 'model/critic.pkl')
    env.close()

class Agent:
    def __init__(self, data, device):
        self.gamma = 0.1
        self.lr_pi = 2e-4
        self.lr_v = 5e-4
        self.action_size = data['word'].x.size()[1]

        self.policy = Policy(data).to(device)
        self.optimizer = Adam(self.policy.parameters(), self.lr_pi)
        self.data = data
        self.device = device

        self.count_dist = np.load('/home/yamanishi/project/trip_recommend/model/RL/count.npy')

    def get_action(self, state, episode):
        state = state.unsqueeze(0)
        probs, value = self.policy(state)
        probs = Categorical(probs)
        action = probs.sample()
        log_prob = probs.log_prob(action).unsqueeze(0)
        return action, log_prob, value

    def update(self, state, log_prob, reward, next_state, done, value):
        state = state.unsqueeze(0)
        next_state = next_state.unsqueeze(0)
        with torch.no_grad():
            _, next_v = self.policy(next_state)
        target = reward + self.gamma * next_v*(1-done)
        v =value
        delta = target.detach()
        delta.detach()
        loss_pi = -log_prob*delta
        loss_v = F.mse_loss(v, target)
        loss = loss_pi + loss_v
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
        self.current_reward = reward

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
            action, log_prob, value = agent.get_action(state, episode)
            next_state, reward, done, info = env.step(action)
            agent.update(state, log_prob, reward, next_state, done, value)
            wandb.log({'step':step, 'reward':reward})
            state = next_state
            total_reward+=reward

        if episode%sync_interval==0:
            agent.sync_action()
        wandb.log({'episode':episode,'total_reward':total_reward})
        print(total_reward)
        reward_history.append(total_reward)

if __name__=='__main__':
    device='cuda:2'
    #wandb.init(project='reinforcement poi', name='actor-critic')
    torch.autograd.set_detect_anomaly(True)
    simulation(device)
            