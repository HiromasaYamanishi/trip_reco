import torch
import torch.nn as nn
from torch_geometric.nn import Linear
from collections import defaultdict
from torch_scatter.scatter import scatter
import yaml
import numpy as np
#from conv.attention import AttentionModule
#from conv.heterolinear import HeteroLinear
import sys
#from get_data import get_data

class AttentionModule(torch.nn.Module):
    def __init__(self, input_dim, num_heads=4, split=1,):
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.split = split
        self.out_dim = input_dim
        self.per_dim = input_dim//num_heads

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

class HeteroLinear(torch.nn.Module):
    def __init__(self, in_channels_dict, out_channels):
        super().__init__()
        self.linears = nn.ModuleDict()
        for node_type, in_channels in in_channels_dict.items():
            self.linears[node_type] = Linear(in_channels, out_channels, weight_initializer='glorot')

    def forward(self, x_dict):
        x_dict_out = {}
        for node_type, x in x_dict.items():
            x = self.linears[node_type](x)
            x_dict_out[node_type] = x
        return x_dict_out

class HeteroGCNConv(torch.nn.Module):
    def __init__(self, in_channels_dict, x_dict, edge_index_dict, out_channels, ReLU):
        super().__init__()

        self.linear = nn.ModuleDict({})
        self.div = defaultdict(int)
        for k in edge_index_dict.keys():
            self.linear['__'.join(k) + '__source'] = Linear(in_channels_dict[k[0]], out_channels, False, weight_initializer='glorot')
            self.linear['__'.join(k) + '__target'] = Linear(in_channels_dict[k[-1]], out_channels, False, weight_initializer='glorot')
            self.div[k[-1]]+=1

        self.div_all = {}
        for k,v in edge_index_dict.items():
            source_div =  torch.zeros(x_dict[k[0]].size(0)).long()
            target_div =  torch.zeros(x_dict[k[-1]].size(0)).long()
            source_value, source_count = torch.unique(v[0].cpu(), return_counts=True)
            target_value, target_count = torch.unique(v[1].cpu(), return_counts=True)
            source_div[source_value.long()] = source_count
            target_div[target_value.long()] = target_count
            source_div[source_div==0] = 1e-6         
            target_div[target_div==0] = 1e-6
            source_div = source_div[v[0]]
            target_div = target_div[v[1]]
            self.div_all['__'.join(k)] = torch.sqrt(source_div * target_div) 
            del source_div, target_div 
        self.ReLU = ReLU

    def forward(self, x_dict, edge_index_dict):
        x_dict_out = {}

        for k,v in edge_index_dict.items():
            source, target = k[0], k[-1]
            source_x = self.linear['__'.join(k) + '__source'](x_dict[source])
            target_x = self.linear['__'.join(k) + '__target'](x_dict[target])
            source_index = v[0].reshape(-1)
            target_index = v[1].reshape(-1)
            out = torch.zeros_like(target_x).to(target_x.device)
            source_x = source_x[source_index]
            source_x = source_x/self.div_all['__'.join(k)].unsqueeze(1).to(source_x.device)

            #target_x = target_x + scatter(source_x, target_index, out=out, dim=0, reduce='mean')
            target_x = scatter(source_x, target_index, out=out, dim=0, reduce='sum')
            if x_dict_out.get(target)!=None:
                x_dict_out[target] += target_x
            else:
                x_dict_out[target] = target_x

        #x_dict_out = {k: self.l2_norm(v) for k,v in x_dict_out.items()}    

        x_dict_out = {k: (v/self.div[k]).relu() for k,v in x_dict_out.items()}  
        if self.ReLU: 
            x_dict_out = {k: v.relu() for k,v in x_dict_out.items()} 
        return x_dict_out

class HeteroGCNLightConv(torch.nn.Module):
    def __init__(self, in_channels_dict, x_dict, edge_index_dict, config):
        super().__init__()

        self.linear = nn.ModuleDict({})
        self.div = defaultdict(int)
        for k in edge_index_dict.keys():
            self.div[k[-1]]+=1

        self.div_all = {}
        for k,v in edge_index_dict.items():
            source_div =  torch.zeros(x_dict[k[0]].size(0)).long()
            target_div =  torch.zeros(x_dict[k[-1]].size(0)).long()
            source_value, source_count = torch.unique(v[0].cpu(), return_counts=True)
            target_value, target_count = torch.unique(v[1].cpu(), return_counts=True)
            source_div[source_value.long()] = source_count
            target_div[target_value.long()] = target_count
            source_div[source_div==0] = 1e-6         
            target_div[target_div==0] = 1e-6
            source_div = source_div[v[0]]
            target_div = target_div[v[1]]
            self.div_all['__'.join(k)] = torch.sqrt(source_div * target_div) 
            del source_div, target_div 
        self.ReLU = config['model']['ReLU']
        self.config = config

    def forward(self, x_dict, edge_index_dict):
        x_dict_out = {}

        for k,v in edge_index_dict.items():
            source, target = k[0], k[-1]
            source_x = x_dict[source]
            target_x = x_dict[target]
            source_index = v[0].reshape(-1)
            target_index = v[1].reshape(-1)
            out = torch.zeros_like(target_x).to(target_x.device)
            source_x = source_x[source_index]
            source_x = source_x/self.div_all['__'.join(k)].unsqueeze(1).to(source_x.device)
            if self.config['trainer']['drop_edge']:
                if not ('spot' in k and 'user' in k):
                    ratio = np.random.uniform(0.5, 0.95)
                    index = np.random.randint(0, source_x.size(0),int(source_x.size(0)*ratio))
                    source_x[index]=0

            if self.config['trainer']['drop_attr']:
                if not ('spot' in k and 'user' in k):
                    ratio = np.random.uniform(0, 0.1)
                    index = np.random.randint(0, source_x.size(1),int(source_x.size(1)*ratio))
                    source_x[:, index]=0               
            #target_x = target_x + scatter(source_x, target_index, out=out, dim=0, reduce='mean')
            target_x = scatter(source_x, target_index, out=out, dim=0, reduce='sum')
            if x_dict_out.get(target)!=None:
                x_dict_out[target] += target_x
            else:
                x_dict_out[target] = target_x

        #x_dict_out = {k: self.l2_norm(v) for k,v in x_dict_out.items()}    

        x_dict_out = {k: (v/self.div[k]).relu() for k,v in x_dict_out.items()}  
        if self.ReLU: 
            x_dict_out = {k: v.relu() for k,v in x_dict_out.items()} 
        return x_dict_out


class HeteroGCN(torch.nn.Module):
    def __init__(self, data, config, out_channels=1,multi=True):
        super().__init__()
        self.hidden_channels = config['model']['hidden_channels']
        self.num_layers = config['model']['num_layers']
        self.concat = config['model']['concat']
        self.ReLU = config['model']['ReLU']

        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict
        self.layers = torch.nn.ModuleList()
        self.multi = multi
        self.first_in_channels_dict = {node_type: x.size(1) for node_type, x in x_dict.items()}
        self.mid_in_channels_dict = {node_type: self.hidden_channels for node_type in x_dict.keys()}
        if multi==True:
            self.att = AttentionModule(input_dim=512, split=5)
            self.first_in_channels_dict['spot'] = 512
       
        self.layers.append(HeteroGCNConv(self.first_in_channels_dict, x_dict, edge_index_dict, self.hidden_channels, self.ReLU))

        for i in range(self.num_layers-1):
            self.layers.append(HeteroGCNConv(self.mid_in_channels_dict, x_dict, edge_index_dict, self.hidden_channels, self.ReLU))
        self.linears = HeteroLinear(self.mid_in_channels_dict, out_channels)
        self.multi = multi

    def forward(self, x_dict, edge_index_dict):
        x_dict_all = {node_type: [] for node_type in x_dict.keys()}
        if self.multi==True:
            x_dict['spot'] = self.att(x_dict['spot'])

        for l in self.layers:
            x_dict = l(x_dict, edge_index_dict)
            if not self.concat:continue
            for node_type in x_dict.keys():
                x_dict_all[node_type].append(x_dict[node_type])
        
        out_dict = self.linears(x_dict)
        if self.concat:
            #x_dict = {node_type: torch.cat(x, dim=1) for node_type, x in x_dict_all.items()}
            x_dict = {node_type: torch.mean(torch.stack(x, dim=1), dim=1) for node_type, x in x_dict_all.items()}
        return x_dict, out_dict