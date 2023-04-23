import torch
import torch.nn as nn
from torch_geometric.nn import Linear
from collections import defaultdict
from torch_scatter.scatter import scatter
#from attention import AttentionModule
#from heterolinear import HeteroLinear
import sys
import math
import yaml
import numpy as np
import os
#from get_data import get_data

def normalize(x):
    return x/torch.sum(x, dim=1).reshape(-1, 1)

class ContextLearn(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        jalan_graph_dir = '/home/yamanishi/project/trip_recommend/data/jalan/graph'
        self.people_context = normalize(torch.from_numpy(np.load(os.path.join(jalan_graph_dir, 'people_contexts.npy'))))
        self.season_context = normalize(torch.from_numpy(np.load(os.path.join(jalan_graph_dir, 'season_contexts.npy'))))
        self.sex_context = normalize(torch.from_numpy(np.load(os.path.join(jalan_graph_dir, 'sex_contexts.npy'))))
        self.age_context = normalize(torch.from_numpy(np.load(os.path.join(jalan_graph_dir, 'age_contexts.npy'))))
        self.time_context = normalize(torch.from_numpy(np.load(os.path.join(jalan_graph_dir, 'time_contexts.npy'))))
        

        self.hidden_channels = hidden_channels
        self.people_embedding = torch.nn.Embedding(num_embeddings=5, embedding_dim=hidden_channels)
        self.season_embedding = torch.nn.Embedding(num_embeddings=12, embedding_dim=hidden_channels)
        self.sex_embedding = torch.nn.Embedding(num_embeddings=5, embedding_dim=hidden_channels)
        self.age_embedding = torch.nn.Embedding(num_embeddings=5, embedding_dim=hidden_channels)
        self.time_embedding = torch.nn.Embedding(num_embeddings=4, embedding_dim=hidden_channels)
        torch.nn.init.normal_(self.people_embedding.weight, std=0.1)
        torch.nn.init.normal_(self.season_embedding.weight, std=0.1)
        torch.nn.init.normal_(self.sex_embedding.weight, std=0.1)
        torch.nn.init.normal_(self.age_embedding.weight, std=0.1)
        torch.nn.init.normal_(self.time_embedding.weight, std=0.1)
        self.linear = torch.nn.Linear(self.hidden_channels*5, self.hidden_channels)

    def forward(self):
        people_x, season_x, sex_x, age_x, time_x = self.people_embedding.weight, self.season_embedding.weight, self.sex_embedding.weight, self.age_embedding.weight, self.time_embedding.weight
        people_spot = torch.matmul(self.people_context, people_x)
        season_spot = torch.matmul(self.season_context, season_x)
        sex_spot = torch.matmul(self.sex_context, sex_x)
        age_spot = torch.matmul(self.age_context, age_x)
        time_spot = torch.matmul(self.time_context, time_x)
        spot_out = torch.cat([people_spot, season_spot, sex_spot, age_spot, time_spot], dim=1)
        spot_out = self.linear(spot_out)
        return spot_out

class ContextConv(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        jalan_graph_dir = '/home/yamanishi/project/trip_recommend/data/jalan/graph'
        self.people_context = normalize(torch.from_numpy(np.load(os.path.join(jalan_graph_dir, 'people_contexts.npy'))))
        self.season_context = normalize(torch.from_numpy(np.load(os.path.join(jalan_graph_dir, 'season_contexts.npy'))))
        self.sex_context = normalize(torch.from_numpy(np.load(os.path.join(jalan_graph_dir, 'sex_contexts.npy')))+1e-6)
        self.age_context = normalize(torch.from_numpy(np.load(os.path.join(jalan_graph_dir, 'age_contexts.npy'))))
        self.time_context = normalize(torch.from_numpy(np.load(os.path.join(jalan_graph_dir, 'time_contexts.npy'))))
        
        self.people_context_T = normalize(torch.from_numpy(np.load(os.path.join(jalan_graph_dir, 'people_contexts.npy'))).T)
        self.season_context_T = normalize(torch.from_numpy(np.load(os.path.join(jalan_graph_dir, 'season_contexts.npy'))).T)
        self.sex_context_T = normalize(torch.from_numpy(np.load(os.path.join(jalan_graph_dir, 'sex_contexts.npy'))).T+1e-6)
        self.age_context_T = normalize(torch.from_numpy(np.load(os.path.join(jalan_graph_dir, 'age_contexts.npy'))).T)
        self.time_context_T = normalize(torch.from_numpy(np.load(os.path.join(jalan_graph_dir, 'time_contexts.npy'))).T)
        self.hidden_channels = hidden_channels
        self.spot_linear = torch.nn.Linear(self.hidden_channels, self.hidden_channels)
        self.people_linear = torch.nn.Linear(self.hidden_channels, self.hidden_channels)
        self.season_linear = torch.nn.Linear(self.hidden_channels, self.hidden_channels) 
        self.sex_linear = torch.nn.Linear(self.hidden_channels, self.hidden_channels)
        self.age_linear = torch.nn.Linear(self.hidden_channels, self.hidden_channels)
        self.time_linear = torch.nn.Linear(self.hidden_channels, self.hidden_channels)
        self.linear = torch.nn.Linear(self.hidden_channels*5, self.hidden_channels)

    def forward(self, x_dict):
        spot_x, people_x, season_x, sex_x, age_x, time_x =x_dict['spot'], x_dict['people'], x_dict['season'], x_dict['sex'], x_dict['age'], x_dict['time']
        device = spot_x.device
        spot_x, people_x, season_x, sex_x, age_x, time_x = self.spot_linear(spot_x), self.people_linear(people_x), self.season_linear(season_x), self.sex_linear(sex_x), self.age_linear(age_x), self.time_linear(time_x)
        people_spot = torch.matmul(self.people_context.to(device), people_x)
        season_spot = torch.matmul(self.season_context.to(device), season_x)
        sex_spot = torch.matmul(self.sex_context.to(device), sex_x)
        age_spot = torch.matmul(self.age_context.to(device), age_x)
        time_spot = torch.matmul(self.time_context.to(device), time_x)
        spot_out = torch.mean(torch.stack([people_spot, season_spot, sex_spot, age_spot, time_spot], dim=1), dim=1)
        #spot_out = self.linear(spot_out)

        people_out = torch.matmul(self.people_context_T.to(device), spot_x)
        season_out = torch.matmul(self.season_context_T.to(device), spot_x)
        sex_out = torch.matmul(self.sex_context_T.to(device), spot_x)
        time_out = torch.matmul(self.time_context_T.to(device), spot_x)
        age_out = torch.matmul(self.age_context_T.to(device), spot_x)
        x_dict_out = {'spot': spot_out,
                        'people': people_out,
                        'sex': sex_out, 
                        'time': time_out, 
                        'age': age_out,
                        'season': season_out}
        x_dict_out = {k:v.relu() for k,v in x_dict_out.items()}
        return x_dict_out

class ContextGNN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers=2):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.hidden_channels = hidden_channels
        self.spot_embedding = torch.nn.Embedding(num_embeddings=42852, embedding_dim=hidden_channels)
        self.people_embedding = torch.nn.Embedding(num_embeddings=5, embedding_dim=hidden_channels)
        self.season_embedding = torch.nn.Embedding(num_embeddings=12, embedding_dim=hidden_channels)
        self.sex_embedding = torch.nn.Embedding(num_embeddings=5, embedding_dim=hidden_channels)
        self.age_embedding = torch.nn.Embedding(num_embeddings=5, embedding_dim=hidden_channels)
        self.time_embedding = torch.nn.Embedding(num_embeddings=4, embedding_dim=hidden_channels)
        torch.nn.init.normal_(self.spot_embedding.weight, std=0.1)
        torch.nn.init.normal_(self.people_embedding.weight, std=0.1)
        torch.nn.init.normal_(self.season_embedding.weight, std=0.1)
        torch.nn.init.normal_(self.sex_embedding.weight, std=0.1)
        torch.nn.init.normal_(self.age_embedding.weight, std=0.1)
        torch.nn.init.normal_(self.time_embedding.weight, std=0.1)
        self.layers = torch.nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(ContextConv(hidden_channels))

    def forward(self, x_dict):
        #x_dict['spot'] = self.spot_embedding.weight
        x_dict['season'] = self.season_embedding.weight
        x_dict['sex'] = self.sex_embedding.weight
        x_dict['time'] = self.time_embedding.weight
        x_dict['age'] = self.age_embedding.weight
        x_dict['people'] = self.people_embedding.weight

        for l in self.layers:
            x_dict = l(x_dict)
        return x_dict

            



    

if __name__=='__main__':
    with open('../config.yaml') as f:
        config = yaml.safe_load(f)

    config['k'] = 20
    config['device'] = 'cuda:1'
    config['explain_num'] = 10
    config['epoch_num'] = 2500
    config['model']['model_type'] = 'ggnn'
    config['model']['num_layers'] = 4
    config['model']['hidden_channels'] = 256
    config['model']['concat'] = True
    config['model']['ReLU'] = True
    config['trainer']['explain_span'] = 50
    config['trainer']['lr'] = 0.0003
    config['trainer']['loss_city_weight'] = 0
    config['trainer']['loss_category_weight'] = 0
    config['trainer']['loss_word_weight'] = 0.2
    config['trainer']['loss_pref_weight'] = 0
    config['trainer']['city_pop_weight']=0
    config['trainer']['spot_pop_weight']=0
    config['data']['word'] = True
    config['data']['city'] = True
    config['data']['category'] = True
    config['data']['pref'] = True
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    gnn = ContextGNN(128)
    out = gnn()
    print(out)