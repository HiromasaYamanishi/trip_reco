import torch
import torch.nn as nn
from torch_geometric.nn import Linear
from collections import defaultdict
from torch_scatter.scatter import scatter
#from attention import AttentionModule
#from heterolinear import HeteroLinear
import sys
import yaml
from conv.attention import AttentionModule
#from get_data import get_data


class HeteroSAGEConv(torch.nn.Module):
    def __init__(self, in_channels_dict, edge_index_dict, config):
        super().__init__()

        self.linear = nn.ModuleDict({})
        self.div = defaultdict(int)
        out_channels = config['model']['hidden_channels']
        for k in edge_index_dict.keys():
            self.linear['__'.join(k) + '__source'] = Linear(in_channels_dict[k[0]], out_channels, False, weight_initializer='glorot')
            self.linear['__'.join(k) + '__target'] = Linear(in_channels_dict[k[-1]], out_channels, False, weight_initializer='glorot')
            self.div[k[-1]]+=1
    
        self.ReLU = True

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

            #target_x = target_x + scatter(source_x, target_index, out=out, dim=0, reduce='mean')
            target_x =  target_x + scatter(source_x, target_index, out=out, dim=0, reduce='mean')
            if x_dict_out.get(target)!=None:
                x_dict_out[target] += target_x
            else:
                x_dict_out[target] = target_x

        #x_dict_out = {k: self.l2_norm(v) for k,v in x_dict_out.items()}    

        x_dict_out = {k: (v/self.div[k]).relu() for k,v in x_dict_out.items()}  
        if self.ReLU: 
            x_dict_out = {k: v.relu() for k,v in x_dict_out.items()} 
        return x_dict_out


class TPGNN(torch.nn.Module):
    def __init__(self, data, config):
        super().__init__()
        self.layers = nn.ModuleList()
        self.config = config
        self.num_layers = config['model']['tpgnn_layers']
        self.hidden_channels = config['model']['hidden_channels']
        self.layers = torch.nn.ModuleList()
        x_dict = data.x_dict
        self.first_in_channels_dict = {node_type: x.size(1) for node_type, x in x_dict.items()}
        self.mid_in_channels_dict = {node_type: self.hidden_channels for node_type in x_dict.keys()}
        
        for i in range(self.num_layers):
            if i==0:
                self.layers.append(HeteroSAGEConv(self.first_in_channels_dict, data.edge_index_dict, config))
            else:
                self.layers.append(HeteroSAGEConv(self.mid_in_channels_dict, data.edge_index_dict, config))
        
    def forward(self, x_dict, edge_index_dict,):
        for l in self.layers:
            x_dict = l(x_dict, edge_index_dict)
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