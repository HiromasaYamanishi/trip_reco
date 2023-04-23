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
from conv.lgcn import LGCNConv
from conv.ggnn import HeteroGGNNConv
#from lgcn import LGCNConv
#from ggnn import HeteroGGNNConv

class LGCNGGNN(torch.nn.Module):
    def __init__(self, data, config):
        super().__init__()
        self.device = config['device']
        self.user_emb = data['user'].x.cuda()
        self.spot_emb = data['spot'].x.cuda()
        self.num_layers = config['model']['num_layers']
        self.hidden_channels = config['model']['hidden_channels']
        self.concat = config['model']['concat']
        self.ReLU = config['model']['ReLU']
        self.pool = config['model']['pool']
        self.lgcn_layers = torch.nn.ModuleList()
        self.ggnn_layers = torch.nn.ModuleList()
        for i in range(self.num_layers):
            self.lgcn_layers.append(LGCNConv(data, config))
        first_in_channels_dict = {node_type: x.size(1) for node_type, x in data.x_dict.items()}
        mid_in_channels_dict = {node_type: self.hidden_channels for node_type in data.x_dict.keys()}
        self.ggnn_layers.append(HeteroGGNNConv(first_in_channels_dict, data.edge_index_dict, self.hidden_channels, self.ReLU, self.pool))
        for i in range(self.num_layers):
            self.ggnn_layers.append(HeteroGGNNConv(mid_in_channels_dict, data.edge_index_dict, self.hidden_channels, self.ReLU, self.pool))
        self.f = nn.Sigmoid()

        self.lgcn_spot_emb =  torch.nn.Embedding(num_embeddings=data['spot'].x.size(0), embedding_dim=128).cuda()
        self.lgcn_user_emb = torch.nn.Embedding(num_embeddings=data['user'].x.size(0), embedding_dim=128).cuda()
        nn.init.normal_(self.lgcn_spot_emb.weight, std=0.1)
        nn.init.normal_(self.lgcn_user_emb.weight, std=0.1)

        self.ggnn_spot_emb = torch.nn.Embedding(num_embeddings=data['spot'].x.size(0), embedding_dim=data['spot'].x.size(1)).cuda()
        self.ggnn_user_emb = torch.nn.Embedding(num_embeddings=data['user'].x.size(0), embedding_dim=data['user'].x.size(1)).cuda()
        self.ggnn_spot_emb.weight = torch.nn.Parameter(data['spot'].x)
        self.ggnn_user_emb.weight = torch.nn.Parameter(data['user'].x)
    
    def forward(self,x_dict, edge_index_dict):
        user_out_concat = self.lgcn_user_emb.weight.to(self.device)
        spot_out_concat = self.lgcn_spot_emb.weight.to(self.device)
        user_out, spot_out = self.lgcn_layers[0](self.lgcn_user_emb.weight, self.lgcn_spot_emb.weight)
        user_out_concat=user_out_concat+user_out
        spot_out_concat=spot_out_concat+spot_out
        for i in range(1, self.num_layers):
            user_out, spot_out = self.lgcn_layers[i](user_out, spot_out)
            user_out_concat = user_out_concat+user_out
            spot_out_concat = spot_out_concat+spot_out
        user_out_concat/=(self.num_layers+1)
        spot_out_concat/=(self.num_layers+1)

        x_dict['spot'] = self.ggnn_spot_emb.weight
        x_dict['user'] = self.ggnn_user_emb.weight
        for l in self.ggnn_layers:
            x_dict = l(x_dict, edge_index_dict)
            
        spot_emb = x_dict['spot']
        user_emb = x_dict['user']

        user_out = torch.cat([user_out_concat, user_emb], dim=1)
        spot_out = torch.cat([spot_out_concat, spot_emb], dim=1)
        return user_out, spot_out

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.forward()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.user_emb(users)
        pos_emb_ego = self.spot_emb(pos_items)
        neg_emb_ego = self.spot_emb(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def reg_loss(self, user, pos, neg):
        user_emb_0 = self.lgcn_user_emb(user)
        pos_emb_0 = self.lgcn_spot_emb(pos)
        neg_emb_0 = self.lgcn_spot_emb(neg)
        user_emb_1 = self.ggnn_user_emb(user)
        pos_emb_1 = self.ggnn_spot_emb(pos)
        neg_emb_1 = self.ggnn_spot_emb(neg)
        reg_loss = (1/2)*(user_emb_0.norm(2).pow(2)+
                            pos_emb_0.norm(2).pow(2)+
                            neg_emb_0.norm(2).pow(2)+
                            user_emb_1.norm(2).pow(2)+
                            pos_emb_1.norm(2).pow(2)+
                            neg_emb_1.norm(2).pow(2))
        return reg_loss
    
    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
        userEmb0, posEmb0, negEmb0) = self.getEmbedding(users, pos, neg)
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2)+
                            posEmb0.norm(2).pow(2)+
                            negEmb0.norm(2).pow(2))/len(users)

        pos_scores = torch.sum(torch.mul(users_emb, pos_emb), dim=1)
        neg_scores = torch.sum(torch.mul(users_emb, neg_emb), dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores-pos_scores))

        return loss, reg_loss


if __name__=='__main__':
    with open('../config.yaml') as f:
        config = yaml.safe_load(f)
    device = 'cuda:3'
    config['k'] = 20
    config['device'] = device
    config['explain_num'] = 10
    config['epoch_num'] = 2500
    config['model']['model_type'] = 'ggnn'
    config['model']['num_layers'] = 2
    config['model']['hidden_channels'] = 64
    config['model']['concat'] = False
    config['model']['ReLU'] = True
    config['model']['pool'] = 'mean'
    config['trainer']['explain_span'] = 50
    config['trainer']['lr'] = 0.0003
    config['trainer']['loss_city_weight'] = 0
    config['trainer']['loss_category_weight'] = 0
    config['trainer']['loss_word_weight'] = 0.2
    config['trainer']['loss_pref_weight'] = 0
    config['trainer']['city_pop_weight']=0
    config['trainer']['spot_pop_weight']=0
    config['data']['word'] = False
    config['data']['city'] = True
    config['data']['category'] = True
    config['data']['pref'] = True
    data = get_data(word=False, category=True, city=True, prefecture=True, multi=False)
    data = data.to(device)
    x_dict = data.x_dict
    del x_dict['pref']
    del x_dict['city']
    print(x_dict)
    exit()

    lgcn = LGCNGGNN(data, config)
    lgcn=lgcn.to(device)
    print(lgcn)
    optim = optim.Adam(lgcn.parameters(), lr=1e-3)
    user_out, spot_out = lgcn(data.x_dict, data.edge_index_dict)
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
    print('backward')
    optim.step()
    #conv = LGCNConv(data)
    
    #data.to(device)
    #conv.to(device)
    #x_dict_out = conv(data.x_dict, data.edge_index_dict)
    #print(x_dict_out['spot'], x_dict_out['user'])
    