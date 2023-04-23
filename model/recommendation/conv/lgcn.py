import torch
import torch.nn as nn
from torch_geometric.nn import Linear
from collections import defaultdict
from torch_scatter.scatter import scatter
#from attention import AttentionModule
#from heterolinear import HeteroLinear
from torch import optim
import sys
import yaml
import time
#from get_data import get_data

class LGCNConv(torch.nn.Module):
    def __init__(self, data, config):
        super().__init__()
        self.user_spot = data['user', 'visit', 'spot'].edge_index
        self.spot_user = data['spot', 'revvisit', 'user'].edge_index
        self.spot_div = torch.zeros(data.x_dict['spot'].weight.size(0))
        self.user_div = torch.zeros(data.x_dict['user'].weight.size(0))
        spot_value, spot_count = torch.unique(self.spot_user[0].cpu(), return_counts=True)
        user_value, user_count = torch.unique(self.spot_user[1].cpu(), return_counts=True)
        self.spot_div[spot_value.long()] = spot_count.float()
        self.user_div[user_value.long()] = user_count.float()
        self.spot_div[self.spot_div==0] = 1e-6
        self.user_div[self.user_div==0] = 1e-6
        self.device = config['device']
        self.spot_div_all =self.spot_div[self.spot_user[0]]
        self.user_div_all = self.user_div[self.spot_user[1]]

    def forward(self, user_x, spot_x):
        device = self.device
        #user_x = user_x#/ torch.sqrt(self.user_div.to(self.device)).unsqueeze(1)
        #spot_x = spot_x#/ torch.sqrt(self.spot_div.to(self.device)).unsqueeze(1)
        time.sleep(5)
        spot_out = torch.zeros_like(spot_x).to(self.device)
        user_out = torch.zeros_like(user_x).to(self.device)
        spot_x = spot_x[self.user_spot[1]]/ torch.sqrt(self.spot_div_all).unsqueeze(1).to(device)
        user_x = user_x[self.user_spot[0]]/ torch.sqrt(self.user_div_all).unsqueeze(1).to(device)
        print('user x', user_x)
        print('spot x', spot_x)
        print('spot user', self.spot_user[0])
        print('user_spot', self.spot_user[1])
        print('spot div all', self.spot_div_all)
        print('user div all', self.user_div_all)
        spot_out = scatter(user_x.to(device), self.spot_user[0].to(device), out=spot_out.to(device), dim=0)
        user_out = scatter(spot_x.to(device), self.spot_user[1].to(device), out=user_out.to(device), dim=0)
        print('scattered spot out', spot_out)
        print('scattered user out', user_out)
        time.sleep(5)
        spot_out = spot_out / torch.sqrt(self.spot_div.to(device)).unsqueeze(1)
        user_out = user_out / torch.sqrt(self.user_div.to(device)).unsqueeze(1)
        #x_dict_out = {'spot': spot_out, 'user': user_out}
        return user_out, spot_out

class LGCN(torch.nn.Module):
    def __init__(self, data, config):
        super().__init__()
        self.user_emb = data['user'].x.weight
        self.spot_emb = data['spot'].x.weight
        self.num_layers = config['model']['num_layers']
        self.layers = torch.nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(LGCNConv(data, config))
    
    def forward(self,):
        user_out_concat = self.user_emb
        spot_out_concat = self.spot_emb
        user_out, spot_out = self.layers[0](self.user_emb, self.spot_emb)
        user_out_concat=user_out_concat+user_out
        spot_out_concat=spot_out_concat+spot_out
        for i in range(1, self.num_layers):
            user_out, spot_out = self.layers[i](user_out, spot_out)
            time.sleep(5)
            user_out_concat = user_out_concat+user_out
            spot_out_concat = spot_out_concat+spot_out
        user_out_concat/=(self.num_layers+1)
        spot_out_concat/=(self.num_layers+1)
        return user_out_concat, spot_out_concat


if __name__=='__main__':
    with open('../config.yaml') as f:
        config = yaml.safe_load(f)

    config['k'] = 20
    config['device'] = 'cuda:1'
    config['explain_num'] = 10
    config['epoch_num'] = 2500
    config['model']['model_type'] = 'ggnn'
    config['model']['num_layers'] = 2
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
    data = get_data(word=False, category=False, city=False, prefecture=False, multi=True)
    data['spot'].x = torch.rand(len(data['spot'].x), 128).to(device)
    data['spot'].x = torch.nn.Embedding(num_embeddings=len(data['spot'].x), embedding_dim=config['model']['hidden_channels'])
    data['user'].x = torch.nn.Embedding(num_embeddings=len(data['user'].x), embedding_dim=config['model']['hidden_channels'])
    torch.nn.init.normal_(data['user'].x.weight, std=0.1)
    torch.nn.init.normal_(data['spot'].x.weight, std=0.1)
    data.to(device)
    lgcn = LGCN(data, config)
    optim = optim.Adam(lgcn.parameters(), lr=1e-3)
    user_out, spot_out = lgcn()
    user_pos = user_out[data['user'].user_pos]
    spot_pos = user_out[data['user'].item_pos]
    spot_neg = user_out[data['user'].item_neg]
    pos_scores = torch.sum(torch.mul(user_pos, spot_pos), dim=1)
    neg_scores = torch.sum(torch.mul(user_pos, spot_neg), dim=1)
    loss = torch.mean(torch.nn.functional.softplus(neg_scores-pos_scores))
    reg_loss = (1/2) * (user_pos.norm(2).pow(2)+
                        spot_pos.norm(2).pow(2)+
                        spot_neg.norm(2).pow(2))
    loss+=reg_loss
    optim.zero_grad()
    loss.backward()
    optim.step()
    #conv = LGCNConv(data)
    
    #data.to(device)
    #conv.to(device)
    #x_dict_out = conv(data.x_dict, data.edge_index_dict)
    #print(x_dict_out['spot'], x_dict_out['user'])
    