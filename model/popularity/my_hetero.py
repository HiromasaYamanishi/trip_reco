from __future__ import print_function, division
import matplotlib.pyplot as plt
import time
import copy
from typing import OrderedDict

import os
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.nn import ReLU
import pandas as pd
from torch_geometric.data import HeteroData
import torch.nn.functional as F
from torch_geometric.nn import GATConv,HGTConv, GCNConv, HANConv, SAGEConv, HeteroConv, GATv2Conv
from torch_geometric.nn import Linear, to_hetero, Sequential, GNNExplainer
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.loader import NeighborLoader, HGTLoader
sys.path.append('..')
from collect_data.preprocessing.preprocess_refactor import Path
from utils import save_plot, save_cor, EarlyStopping
from graph_test import get_data, train, test, train_epoch, calc_cor
from torch_geometric.nn.aggr import SumAggregation
from torch_scatter.scatter import scatter
from torch_scatter.utils import broadcast
import argparse
from utils import write_cor

class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv((-1, -1), hidden_channels, add_self_loops = False)
        #self.lin1 = Linear(-1, hidden_channels)
        self.conv2 = GATConv((-1, -1), hidden_channels, add_self_loops = False)
        #self.lin2 = Linear(-1, hidden_channels)
        self.conv3 = GATConv((-1, -1), out_channels, add_self_loops = False)
        #self.lin3 = Linear(-1, out_channels)

    def forward(self, x, edge_index):
        x, attention_weights = self.conv1(x, edge_index, return_attention_weights=True) #+ self.lin1(x)
        x = x.relu()
        x = self.conv2(x, edge_index) #+ self.lin2(x)
        x = x.relu()
        x = self.conv3(x, edge_index) #+ self.lin3(x)
        return x, attention_weights

class AttentionModule(torch.nn.Module):
    def __init__(self, input_dim, num_heads=4, split=1, out_dim=1000):
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.split = split
        self.out_dim = out_dim
        self.per_dim = out_dim//num_heads

        self.W = torch.nn.ModuleList([Linear(input_dim, self.per_dim, False, weight_initializer='glorot') for _ in range(num_heads)])
        self.q = torch.nn.ParameterList([])
        for _ in range(num_heads):
            q_ =torch.nn.Parameter(torch.zeros(size=(self.per_dim, 1)))
            nn.init.xavier_uniform_(q_.data, gain=1.414)
            self.q.append(q_)
        
        self.LeakyReLU = torch.nn.LeakyReLU(0.2)

    def forward(self, x):
        out = []
        x = x.resize(x.size()[0],self.split, self.input_dim)
        for i in range(self.num_heads):
            W = self.W[i]
            q = self.q[i]
            x_ = W(x)
            att = self.LeakyReLU(torch.matmul(x_, q))
            att = torch.nn.functional.softmax(att, dim=1)
            att = torch.broadcast_to(att, x_.size())
            x_= (x_*att).sum(dim=1)
            out.append(x_)
        return torch.cat(out, dim=1)

        
class MyHeteroConv(torch.nn.Module):
    def __init__(self, x_dict, edge_index_dict, out_channels, pre_channels=None, channel_dict=None):
        super().__init__()
        num_nodes = {k: v[0] for k, v in x_dict.items()}

        if pre_channels== None:
            num_features = {k: v.size()[1] for k,v in x_dict.items()}
        else:
            num_features = {k: pre_channels for k in x_dict.keys()}

        if channel_dict is not None:
            for k,v in channel_dict.items():
                num_features[k] = v

        self.linear = nn.ModuleDict({})
        for k in edge_index_dict.keys():
            self.linear['__'.join(k) + '__source'] = Linear(num_features[k[0]], out_channels, False, weight_initializer='glorot')
            self.linear['__'.join(k) + '__target'] = Linear(num_features[k[-1]], out_channels, False, weight_initializer='glorot')

    def forward(self, x_dict, edge_index_dict):
        x_dict_out = {}

        for k,v in edge_index_dict.items():
            source, target = k[0], k[-1]
            source_x = self.linear['__'.join(k) + '__source'](x_dict[source])
            target_x = self.linear['__'.join(k) + '__target'](x_dict[target])
            source_index = v[0].reshape(-1)
            target_index = v[1].reshape(-1)
            out = torch.zeros_like(target_x).to(device)
            source_x = source_x[source_index]

            target_x = target_x + scatter(source_x, target_index, out=out, dim=0, reduce='mean')
            if x_dict_out.get(target)!=None:
                x_dict_out[target] += target_x
            else:
                x_dict_out[target] = target_x

        #x_dict_out = {k: self.l2_norm(v) for k,v in x_dict_out.items()}    

        return x_dict_out


    def l2_norm(self, x):
        return x/(torch.norm(x, dim=1) + 1e-6).view(-1, 1).expand(x.size())

class MyHeteroAttentionConv(torch.nn.Module):
    def __init__(self, x_dict, edge_index_dict, out_channels, pre_channels=None):
        super().__init__()
        num_nodes = {k: v[0] for k, v in x_dict.items()}

        if pre_channels== None:
            num_features = {k: v.size()[1] for k,v in x_dict.items()}
        else:
            num_features = {k: pre_channels for k in x_dict.keys()}

        self.linear = nn.ModuleDict({})
        for k in edge_index_dict.keys():
            self.linear['__'.join(k) + '__source'] = Linear(num_features[k[0]], out_channels, False, weight_initializer='glorot')
            self.linear['__'.join(k) + '__target'] = Linear(num_features[k[-1]], out_channels, False, weight_initializer='glorot')
        
        self.attention = nn.ParameterDict({})
        for k in edge_index_dict.keys():
            a = torch.nn.Parameter(torch.zeros(size=(out_channels*2,1)))
            nn.init.xavier_uniform_(a.data, gain=1.414)
            self.attention['__'.join(k)] = a

        self.LeakyReLU = torch.nn.LeakyReLU(0.2)
    
    def forward(self, x_dict, edge_index_dict):
        x_dict_out = {}

        for k,v in edge_index_dict.items():
            source, target = k[0], k[-1]
            source_x = self.linear['__'.join(k) + '__source'](x_dict[source])
            target_x = self.linear['__'.join(k) + '__target'](x_dict[target])

            source_index = v[0].reshape(-1)
            target_index = v[1].reshape(-1)

            out = torch.zeros_like(target_x).to(device)
            source_x_tmp = source_x[source_index] #(53850, 128)
            target_x_tmp = target_x[target_index] #(53850, 128)
            X = torch.cat([source_x_tmp, target_x_tmp], dim=1)

            a = self.attention['__'.join(k)]
            attention = torch.exp(self.LeakyReLU(torch.matmul(X, a)))
            '''
            div = torch.ones(target_x_tmp.size()[0],1).to(device)
            source_x_tmp = scatter(source_x_tmp, target_index, out=out, dim=0, reduce='sum')

            out_div = torch.zeros(target_x.size()[0], 1).to(device)
            div = scatter(div, target_index, out=out_div, dim=0, reduce='sum')
            div[div<1]=1
            '''

            source_x_tmp = source_x_tmp * attention
            source_x_tmp = scatter(source_x_tmp, target_index, out=out, dim=0, reduce='sum')

            out_att = torch.zeros(target_x.size()[0], 1).to(device)
            attention_div = scatter(attention, target_index, out=out_att,dim=0, reduce='sum')

            target_x = target_x + source_x_tmp/ (attention_div+1e-6)
            
            if x_dict_out.get(target)!=None:
                x_dict_out[target] += target_x
            else:
                x_dict_out[target] = target_x

        return x_dict_out

class MyReLU(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_dict):
        for k in x_dict.keys():
            x_dict[k] = x_dict[k].relu()
        return x_dict

        

class MyHetero(torch.nn.Module):
    def __init__(self, x_dict, edge_index_dict, hidden_channels, out_channels,out_dim,multi=False):
        super().__init__()
        if multi==True:
            self.att = AttentionModule(input_dim=out_dim, num_heads=4, split=3, out_dim=out_dim)
            self.conv1 = MyHeteroConv(x_dict, edge_index_dict, out_channels=hidden_channels, pre_channels=None, channel_dict={'spot':out_dim})
        else:
            self.conv1 = MyHeteroConv(x_dict, edge_index_dict, out_channels=hidden_channels, pre_channels=None)
        self.relu1 = MyReLU()
        self.conv2 = MyHeteroConv(x_dict, edge_index_dict, out_channels=hidden_channels, pre_channels= hidden_channels)
        self.relu2 = MyReLU()
        self.linear = Linear(hidden_channels, out_channels)
        self.multi = multi

    def forward(self, x_dict, edge_index_dict):
        if self.multi == True:
            x_dict['spot'] = self.att(x_dict['spot'])
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = self.relu1(x_dict)
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = self.relu2(x_dict)
        out = self.linear(x_dict['spot'])
        return out

class MyAttentionHetero(torch.nn.Module):
    def __init__(self, x_dict, edge_index_dict, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = MyHeteroAttentionConv(x_dict, edge_index_dict, out_channels=hidden_channels, pre_channels=None)
        self.relu1 = MyReLU()
        self.conv2 = MyHeteroAttentionConv(x_dict, edge_index_dict, out_channels=hidden_channels, pre_channels= hidden_channels)
        self.relu2 = MyReLU()
        self.linear = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = self.relu1(x_dict)
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = self.relu2(x_dict)
        out = self.linear(x_dict['spot'])
        return out


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    print(device)

    parser = argparse.ArgumentParser()
    parser.add_argument('--tau', default=1)
    parser.add_argument('--model_name', default='ResNet')
    args = parser.parse_args()
    print(args)
    path = Path()
    multi=False
    data = get_data(model_name=args.model_name, multi=False)
    data.to(device)
    print(data)

    model = MyHetero(data.x_dict, data.edge_index_dict, hidden_channels=128, out_channels=1, out_dim=512,multi=False)
    model.to(device)
    #model.load_state_dict(torch.load('./model.pth'))
    train_epoch(model, data, epoch_num=150)
    torch.save(model.state_dict(), '../data/model/model_single.pth')
    cor = calc_cor(model, data)
    print(cor)
    with open('/home/yamanishi/project/trip_recommend/model/result.txt','a') as f:
        f.write(args.model_name)
    write_cor(cor)
    
    #model = MyHetero(in_channels = )

