import torch
import torch.nn as nn
from torch_geometric.nn import Linear
from collections import defaultdict
from torch_scatter.scatter import scatter, scatter_add
#from attention import AttentionModule
#from heterolinear import HeteroLinear
import sys
#from get_data import get_data
import yaml
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense import Linear
from torch_geometric.nn.inits import glorot, reset
from torch_geometric.typing import Adj, EdgeType, Metadata, NodeType, OptTensor
from torch_geometric.utils import softmax


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
    
def group(
    xs: List[Tensor],
    q: nn.Parameter,
    k_lin: nn.Module,
) -> Tuple[OptTensor, OptTensor]:

    if len(xs) == 0:
        return None, None
    else:
        num_edge_types = len(xs)
        out = torch.stack(xs)
        if out.numel() == 0:
            return out.view(0, out.size(-1)), None
        attn_score = (q * torch.tanh(k_lin(out)).mean(1)).sum(-1)
        attn = F.softmax(attn_score, dim=0)
        out = torch.sum(attn.view(num_edge_types, 1, -1) * out, dim=0)
        return out

class HeteroLSTMConv(torch.nn.Module):
    def __init__(self, in_channels_dict, edge_index_dict, out_channels, ReLU, config):
        super().__init__()

        self.linear = nn.ModuleDict({})
        self.div = defaultdict(int)
        for k in edge_index_dict.keys():
            self.linear['__'.join(k) + '__source'] = Linear(in_channels_dict[k[0]], out_channels, False, weight_initializer='glorot')
            self.linear['__'.join(k) + '__target'] = Linear(in_channels_dict[k[-1]], out_channels, False, weight_initializer='glorot')
            self.div[k[-1]]+=1

        self.lstm = nn.ModuleDict({})
        for k in edge_index_dict.keys():
            self.lstm['__'.join(k)] = nn.LSTMCell(out_channels, out_channels)
        self.out_channels = out_channels
        self.ReLU = ReLU
        self.config = config
        self.heads = 8
        self.dim = out_channels//self.heads
        self.lin_src = nn.ParameterDict()
        self.lin_dst = nn.ParameterDict()
        for k in edge_index_dict.keys():
            self.lin_src['__'.join(k)] = nn.Parameter(torch.Tensor(1, self.heads, self.dim))
            self.lin_dst['__'.join(k)] = nn.Parameter(torch.Tensor(1, self.heads, self.dim))
        glorot(self.lin_dst)
        glorot(self.lin_src)
        self.k_lin_dict = torch.nn.ModuleDict()
        for node_type in self.div.keys():
            self.k_lin_dict[node_type] = nn.Linear(out_channels, out_channels)        
            self.k_lin_dict[node_type].reset_parameters()
        self.q_dict = torch.nn.ParameterDict()
        for node_type in self.div.keys():
            self.q_dict[node_type] = nn.Parameter(torch.Tensor(1, out_channels))
            glorot(self.q_dict[node_type])
            
    def group(
        xs: List[Tensor],
        q: nn.Parameter,
        k_lin: nn.Module,
    ) -> Tuple[OptTensor, OptTensor]:

        if len(xs) == 0:
            return None, None
        else:
            num_edge_types = len(xs)
            out = torch.stack(xs)
            if out.numel() == 0:
                return out.view(0, out.size(-1)), None
            attn_score = (q * torch.tanh(k_lin(out)).mean(1)).sum(-1)
            attn = F.softmax(attn_score, dim=0)
            out = torch.sum(attn.view(num_edge_types, 1, -1) * out, dim=0)
            return out, attn        
        
    def forward(self, x_dict, edge_index_dict):
        x_dict_out = {node_type:[] for node_type in x_dict.keys()}
        target_tmp={}
        aggregated_tmp = {}
        aggregate_meta={}
        for k,v in edge_index_dict.items():
            source, target = k[0], k[-1]
            source_x = self.linear['__'.join(k) + '__source'](x_dict[source])
            target_x = self.linear['__'.join(k) + '__target'](x_dict[target])
            source_index = v[0].reshape(-1)
            target_index = v[1].reshape(-1)
            out = torch.zeros_like(target_x).to(target_x.device)
            if self.config['node_att']:
                source_x = source_x[source_index].view(-1, self.heads, self.dim)
                target_x_ = target_x[target_index].view(-1, self.heads, self.dim)
                alpha_source = (source_x*self.lin_src['__'.join(k)]).sum(dim=-1)
                #alpha_target = (target_x_*self.lin_dst['__'.join(k)]).sum(dim=-1)
                alpha = alpha_source#+alpha_target
                alpha = F.leaky_relu(alpha_source, 0.2)
                alpha = softmax(alpha, target_index)
                alpha = F.dropout(alpha, p=self.config['model']['dropout'], training=self.training)
                #target_x = target_x + scatter(source_x, target_index, out=out, dim=0, reduce='mean')
                #print(source_x.shape)
                #print(alpha_source.shape)
                source_x = (source_x * alpha_source.view(-1, self.heads, 1)).view(-1, self.out_channels)
                aggregated = scatter_add(source_x, target_index, out=out, dim=0)
            else:
                source_x = source_x[source_index]
                aggregated = scatter(source_x, target_index, out=out, dim=0, reduce='mean')
            target_tmp[k]=target_x
            aggregated_tmp[k]=aggregated
            if aggregate_meta.get(target)!=None:
                aggregate_meta[target]+aggregated
            else:
                aggregate_meta[target]=aggregated
        aggregate_meta = {k: v/self.div[k] for k,v in aggregate_meta.items()}
        if self.config['model']['meta']:
            x_dict_out_tmp = {k: self.lstm['__'.join(k)](target_tmp[k], (aggregated_tmp[k], aggregate_meta[k[-1]]))[0] for k in edge_index_dict.keys()}
        else:
            x_dict_out_tmp = {k: self.lstm['__'.join(k)](target_tmp[k], (aggregated_tmp[k], aggregated_tmp[k]))[0] for k in edge_index_dict.keys()}
        for k,v in x_dict_out_tmp.items():
            x_dict_out[k[-1]].append(v)
        #x_dict_out = {k: self.l2_norm(v) for k,v in x_dict_out.items()}    
        if self.config['sem_att']:
            x_dict_out = {k: group(v, self.q_dict[k], self.k_lin_dict[k]) for k,v in x_dict_out.items()}
        else:
            x_dict_out = {k:torch.mean(torch.stack(v, dim=1), dim=1) for k,v in x_dict_out.items()}
        #x_dict_out = {k: v/self.div[k] for k,v in x_dict_out.items()}   
        if self.ReLU:
            x_dict_out = {k: v.relu() for k,v in x_dict_out.items()} 
        return x_dict_out
    '''
    def forward(self, x_dict, edge_index_dict):
        x_dict_out = {}
        for node_type in x_dict.keys():
            aggregate_meta = None
            edge_type_list = []
            for edge_type in edge_index_dict.keys():
                if edge_type[-1]==node_type:
                    edge_type_list.append(edge_type)
            x_dict_out_tmp = {}
            for edge_type in edge_type_list:
                source, target = edge_type[0], edge_type[-1]
                source_x = self.linear['__'.join(edge_type)+'__source'](x_dict[source])
                target_x = self.linear['__'.join(edge_type)+'__target'](x_dict[target])
                source_index = edge_index_dict[edge_type][0].reshape(-1)
                target_index = edge_index_dict[edge_type][1].reshape(-1)

                out = torch.zeros_like(target_x).to(target_x.device)
                source_x = source_x[source_index]
                aggregated = scatter(source_x, target_index, out=out, dim=0, reduce='mean')
                if aggregate_meta==None:
                    aggregate_meta=aggregated
                else:
                    aggregate_meta+=aggregated
                target_x =  self.gru['__'.join(edge_type)](aggregated, target_x)
                x_dict_out_tmp[edge_type] = target_x
            aggregate_meta/=self.div[node_type]
            x_dict_out_tmp = {k: self.gru_meta['__'.join(edge_type)](aggregate_meta, v) for k,v in x_dict_out_tmp.items()}
            for out in x_dict_out_tmp.values():
                if x_dict_out.get(node_type)!=None:
                    x_dict_out[node_type] += out
                else:
                    x_dict_out[node_type] = out
            
        x_dict_out = {k: v/self.div[k] for k,v in x_dict_out.items()} 
        if self.ReLU:
            x_dict_out = {k: v.relu() for k,v in x_dict_out.items()} 
        return x_dict_out
        '''


class HAL(torch.nn.Module):
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
       
        self.layers.append(HeteroLSTMConv(self.first_in_channels_dict, edge_index_dict, self.hidden_channels, self.ReLU, config))

        for i in range(self.num_layers-1):
            self.layers.append(HeteroLSTMConv(self.mid_in_channels_dict, edge_index_dict, self.hidden_channels, self.ReLU, config))
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
        if self.concat==True:
            #x_dict = {node_type: torch.cat(x, dim=1) for node_type, x in x_dict_all.items()}
            x_dict = {node_type: torch.mean(torch.stack(x, dim=1), dim=1) for node_type, x in x_dict_all.items()}
        return x_dict, out_dict

if __name__=='__main__':
    with open('../config.yaml') as f:
        config = yaml.safe_load(f)

    config['k'] = 20
    config['device'] = 'cuda:1'
    config['explain_num'] = 10
    config['epoch_num'] = 2500
    config['model']['model_type'] = 'ggnn'
    config['model']['num_layers'] = 4
    config['model']['hidden_channels'] = 128
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
    config['model']['meta'] = True
    config['model']['dropout'] = 0.1
    config['model']['node_att'] = False
    config['model']['sem_att'] = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = get_data(category=True, city=True, prefecture=True, multi=True)
    
    model = HAL(data, config)
    data.to(device)
    model.to(device)
    print(model)
    x_dict, out_dict= model(data.x_dict, data.edge_index_dict)
    optim = torch.optim.Adam(model.parameters())
    y = torch.rand(42852).to(device)
    loss = torch.nn.functional.mse_loss(y, out_dict['spot'])
    optim.zero_grad()
    loss.backward()
    optim.step()
    print(x_dict)