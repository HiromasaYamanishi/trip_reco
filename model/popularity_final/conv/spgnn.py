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
from conv.attention import AttentionModule
#from get_data import get_data


class SPGGNNConv(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_channels = config['model']['hidden_channels']
        self.dist_emb = torch.nn.Embedding(num_embeddings=20, embedding_dim=self.hidden_channels)
        torch.nn.init.normal_(self.dist_emb.weight, std=0.1)
        self.act = torch.nn.LeakyReLU(0.2)
        self.sigma = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.W1 = torch.nn.Parameter(torch.rand(self.hidden_channels*2, self.hidden_channels))
        self.W2 = torch.nn.Parameter(torch.rand(self.hidden_channels*2, 1))
        torch.nn.init.normal_(self.W1, std=0.1)
        torch.nn.init.normal_(self.W2, std=0.1)
        self.linear = Linear(self.hidden_channels, self.hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        x = x_dict['spot']
        edge_index = edge_index_dict[('spot', 'near', 'spot')]
        distances = edge_attr_dict[('spot', 'near', 'spot')]
        distances = torch.div(distances, 50, rounding_mode='floor')
        source_x = x[edge_index[0]]
        target_x = x[edge_index[1]]
        x = torch.cat([source_x, target_x], dim=1)
        x = self.act(torch.matmul(x, self.W1))
        dist_emb = self.dist_emb(distances)
        x = torch.cat([x, dist_emb], dim=1)
        attention_value = torch.matmul(x, self.W2).reshape(-1)
        attention_value = self.sigma(attention_value).reshape(-1, 1)
        attention_value = torch.exp(attention_value)
        source_x = source_x*attention_value
        out = torch.zeros(x_dict['spot'].size(0), self.hidden_channels).to(target_x.device)
        aggregated = scatter(source_x, edge_index[1], out=out, dim=0, reduce='sum')
        attention_out = torch.zeros(x_dict['spot'].size(0), 1).to(target_x.device)
        attention_aggregated = scatter(attention_value, edge_index[1], out=attention_out, dim=0, reduce='sum')
        aggregated = aggregated/(attention_aggregated+1e-6)
        out_x_dict = {'spot': aggregated.relu()}
        return out_x_dict


class SPGNN(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList()
        self.config = config
        self.num_layers = config['model']['spgnn_layers']
        self.layers = torch.nn.ModuleList()
        self.spot_embedding = torch.nn.Embedding(num_embeddings=42852, embedding_dim=config['model']['hidden_channels'])
        for i in range(self.num_layers):
            self.layers.append(SPGGNNConv(config))
        
    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        #x_dict['spot'] = self.spot_embedding.weight
        for l in self.layers:
            x_dict = l(x_dict, edge_index_dict, edge_attr_dict)
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
    data = get_data(category=True, city=True, prefecture=True, multi=True)
    
    model = HeteroGGNN(data, config)
    data.to(device)
    model.to(device)
    print(model)
    x_dict, out_dict= model(data.x_dict, data.edge_index_dict)
    print(x_dict)