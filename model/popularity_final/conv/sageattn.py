import torch
import torch.nn as nn
from torch_geometric.nn import Linear
from collections import defaultdict
from torch_scatter.scatter import scatter
#from attention import AttentionModule
#from heterolinear import HeteroLinear
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

class HeteroSAGEAttentionConv(torch.nn.Module):
    def __init__(self, in_channels_dict, edge_index_dict, out_channels,):
        super().__init__()

        self.linear = nn.ModuleDict({})
        self.div = defaultdict(int)
        for k in edge_index_dict.keys():
            self.linear['__'.join(k) + '__source'] = Linear(in_channels_dict[k[0]], out_channels, False, weight_initializer='glorot')
            self.linear['__'.join(k) + '__target'] = Linear(in_channels_dict[k[-1]], out_channels, False, weight_initializer='glorot')
            self.div[k[-1]]+=1

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
            out = torch.zeros_like(target_x).to(target_x.device)
            source_x = source_x[source_index]
            source_x_tmp = source_x[source_index] #(53850, 128)
            target_x_tmp = target_x[target_index] #(53850, 128)
            X = torch.cat([source_x_tmp, target_x_tmp], dim=1)

            a = self.attention['__'.join(k)]
            attention = torch.exp(self.LeakyReLU(torch.matmul(X, a)))
            source_x_tmp = source_x_tmp * attention
            source_x_tmp = scatter(source_x_tmp, target_index, out=out, dim=0, reduce='sum')

            out_att = torch.zeros(target_x.size()[0], 1).to(target_x.device)
            attention_div = scatter(attention, target_index, out=out_att,dim=0, reduce='sum')

            target_x = target_x + source_x_tmp/ (attention_div+1e-6)

            if x_dict_out.get(target)!=None:
                x_dict_out[target] += target_x
            else:
                x_dict_out[target] = target_x

        x_dict_out = {k: (v/self.div[k]).relu() for k,v in x_dict_out.items()}   
        x_dict_out = {k: v.relu() for k,v in x_dict_out.items()} 
        return x_dict_out


class HeteroSAGEAttention(torch.nn.Module):
    def __init__(self, x_dict, edge_index_dict, num_layers, hidden_channels, out_channels,multi=False):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.multi = multi
        self.first_in_channels_dict = {node_type: x.size(1) for node_type, x in x_dict.items()}
        self.mid_in_channels_dict = {node_type: hidden_channels for node_type in x_dict.keys()}
        if multi==True:
            self.att = AttentionModule(input_dim=512, split=5)
            self.first_in_channels_dict['spot'] = 512
       
        self.layers.append(HeteroSAGEAttentionConv(self.first_in_channels_dict, edge_index_dict, hidden_channels))

        for i in range(num_layers-1):
            self.layers.append(HeteroSAGEAttentionConv(self.mid_in_channels_dict, edge_index_dict, hidden_channels))
        self.linears = HeteroLinear(self.mid_in_channels_dict, out_channels)
        self.multi = multi

    def forward(self, x_dict, edge_index_dict):
        if self.multi==True:
            x_dict['spot'] = self.att(x_dict['spot'])

        for l in self.layers:
            x_dict = l(x_dict, edge_index_dict)

        out_dict = self.linears(x_dict)
        
        return x_dict, out_dict

if __name__=='__main__':
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    data = get_data(category=True, city=True, prefecture=True, multi=True)
    
    model = HeteroSAGEAttention(data.x_dict, data.edge_index_dict, num_layers=5,hidden_channels=128, out_channels=1,multi=True)
    data.to(device)
    model.to(device)
    print(model)
    x_dict, out_dict= model(data.x_dict, data.edge_index_dict)
    print(x_dict)