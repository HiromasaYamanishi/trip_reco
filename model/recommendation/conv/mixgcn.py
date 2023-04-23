import torch
import torch.nn as nn
from torch_geometric.nn import Linear
from collections import defaultdict
from torch_scatter.scatter import scatter
import yaml
import numpy as np
import os
#from conv.attention import AttentionModule
#from conv.heterolinear import HeteroLinear
import sys
from torch_geometric.nn.conv.gcnlight_conv import GCNLightConv
import pandas as pd
#from get_data import get_data

class UserSpotConv(torch.nn.Module):
    def __init__(self, user_spot):
        super().__init__()
        self.n_user = 27094
        self.m_spot = 42852
        self.user_div = torch.zeros(self.n_user).long()
        self.spot_div = torch.zeros(self.m_spot).long()
        user_value, user_count = torch.unique(user_spot[0].cpu(), return_counts=True)
        self.user_div[user_value.long()] = user_count
        spot_value, spot_count = torch.unique(user_spot[1].cpu(), return_counts=True)
        self.spot_div[spot_value.long()] = spot_count
        self.div = torch.sqrt(self.user_div[user_spot[0]]*self.spot_div[user_spot[1]]).unsqueeze(1)
        self.user_spot = user_spot

    def forward(self, spot_x, user_x):
        user_spot = self.user_spot.to(spot_x.device)
        source_spot = spot_x[user_spot[1]]
        source_spot = source_spot/self.div.to(spot_x.device)
        user_out = torch.zeros_like(user_x).to(user_x.device)
        user_out = scatter(source_spot, user_spot[0], out=user_out, dim=0, reduce='sum')
        source_user = user_x[user_spot[0]]
        source_user = source_user/self.div.to(spot_x.device)
        spot_out = torch.zeros_like(spot_x).to(spot_x.device)
        spot_out = scatter(source_user, user_spot[1], out=spot_out, dim=0, reduce='sum')
        del source_user, source_spot, 
        return spot_out, user_out

class MixGCN(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config['device']
        self.user_spot = torch.from_numpy(np.load('/home/yamanishi/project/trip_recommend/model/recommendation/data/train_edge.npy')).to(self.device)
        self.hidden_channels = config['model']['hidden_channels']
        self.n_users = 27094
        self.m_items = 42852
        self.user_embedding = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.hidden_channels)
        self.spot_embedding = torch.nn.Embedding(num_embeddings=self.m_items, embedding_dim=self.hidden_channels)
        torch.nn.init.normal_(self.user_embedding.weight, std=0.1)
        torch.nn.init.normal_(self.spot_embedding.weight, std=0.1)
        self.num_layers = config['model']['num_layers']
        self.layers = torch.nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(UserSpotConv(self.user_spot))

        self.jalan_graph_dir = '/home/yamanishi/project/trip_recommend/data/jalan/graph'
        self.jalan_spot_dir = '/home/yamanishi/project/trip_recommend/data/jalan/spot'
        self.city = torch.from_numpy(np.load(os.path.join(self.jalan_graph_dir, 'spot_city.npy')))[1]
        self.pref = torch.from_numpy(np.load(os.path.join(self.jalan_graph_dir, 'spot_pref.npy')))[1]
        self.n_city = max(self.city) + 1
        self.n_pref = max(self.pref) + 1

        self.df = pd.read_csv(os.path.join(self.jalan_spot_dir, 'experience_light.csv'))
        self.popularity = torch.from_numpy(self.df['review_count'].values)

    def forward(self):
        spot_out, user_out = [], []
        spot_x = self.spot_embedding.weight
        user_x = self.user_embedding.weight
        spot_out.append(spot_x)
        user_out.append(user_x)
        for i in range(self.num_layers):
            spot_x, user_x = self.layers[i](spot_x, user_x)
            spot_out.append(spot_x)
            user_out.append(user_x)
        spot_out = torch.stack(spot_out, dim=1) #[num_spot, num_channel, hidden_dim]
        user_out = torch.stack(user_out, dim=1) #[num_user, num_channel, hidden_dim]
        return spot_out, user_out

    def geomix(self, spot_x_out, user_x_out, users, pos, neg):
        #users: [batch] pos: [batch], neg: [batch, n_neg]
        city_out = torch.zeros((self.n_city, spot_x_out.size(1), spot_x_out.size(2)))
        city_embedding = scatter(spot_x_out, self.city.to(spot_x_out.device), out=city_out.to(spot_x_out.device), dim=0, reduce='mean')
        pref_out = torch.zeros((self.n_pref, spot_x_out.size(1), spot_x_out.size(2)))
        pref_embedding = scatter(spot_x_out, self.pref.to(spot_x_out.device), out=pref_out.to(spot_x_out.device), dim=0, reduce='mean')

        batch_size = neg.size(0)
        n_neg = neg.size(1)
        users_emb = user_x_out[users] #[batch, num_channel, hidden_dim]
        pos_emb = spot_x_out[pos] #[batch, num_channels, hidden_dim]
        neg_emb = spot_x_out[neg.reshape(-1)].reshape(batch_size, n_neg, -1, self.hidden_channels) #[batch, n_neg,num_channels,hidden_size]
        city_emb = city_embedding[self.city[pos]]
        pref_emb = pref_embedding[self.pref[pos]]
        preference_emb = pos_emb-pref_emb

        """positive mixing"""
        seed_geo = torch.rand(batch_size, 1, pos_emb.shape[1], 1).to(pos_emb.device)
        neg_emb = seed_geo*preference_emb.unsqueeze(dim=1) + (1-seed_geo)*neg_emb
        seed = torch.rand(batch_size, 1, pos_emb.shape[1], 1).to(pos_emb.device)
        neg_emb = seed*pos_emb.unsqueeze(dim=1) + (1-seed)*neg_emb
        """hop mixing"""
        scores = (users_emb.unsqueeze(dim=1)*neg_emb).sum(dim=-1) #[batch, n_neg, num_channels]
        indices = torch.max(scores, dim=1)[1].detach().to(pos_emb.device) #[batch, num_channels]
        neg_emb = neg_emb.permute([0, 2, 1, 3]) #[batch, num_channels, n_neg, hidden_dim]
        neg_emb = neg_emb[[[i] for i in range(batch_size)],
                              range(neg_emb.shape[1]), indices, :]
        return users_emb, pos_emb, neg_emb       
        

    def mixgcf(self, spot_x_out, user_x_out, users, pos, neg):
        #users: [batch] pos: [batch], neg: [batch, n_neg]
        batch_size = neg.size(0)
        n_neg = neg.size(1)
        users_emb = user_x_out[users] #[batch, num_channel, hidden_dim]
        pos_emb = spot_x_out[pos] #[batch, num_channels, hidden_dim]
        neg_emb = spot_x_out[neg.reshape(-1)].reshape(batch_size, n_neg, -1, self.hidden_channels) #[batch, n_neg,num_channels,hidden_size]
        """positive mixing"""
        seed = torch.rand(batch_size, 1, pos_emb.shape[1], 1).to(pos_emb.device)
        neg_emb = seed*pos_emb.unsqueeze(dim=1) + (1-seed)*neg_emb
        """hop mixing"""
        scores = (users_emb.unsqueeze(dim=1)*neg_emb).sum(dim=-1) #[batch, n_neg, num_channels]
        indices = torch.max(scores, dim=1)[1].detach().to(pos_emb.device) #[batch, num_channels]
        neg_emb = neg_emb.permute([0, 2, 1, 3]) #[batch, num_channels, n_neg, hidden_dim]
        neg_emb = neg_emb[[[i] for i in range(batch_size)],
                              range(neg_emb.shape[1]), indices, :]
        return users_emb, pos_emb, neg_emb


    def bpr_loss(self, spot_x_out, user_x_out, users, pos, neg):
        if self.config['model']['mix'] == 'geo':
            users_emb, pos_emb, neg_emb = self.geomix(spot_x_out, user_x_out, users, pos, neg)

        else:
            users_emb, pos_emb, neg_emb = self.mixgcf(spot_x_out, user_x_out, users, pos, neg)

        users_emb_ego = self.user_embedding(users.to(self.device))
        pos_emb_ego = self.spot_embedding(pos.to(self.device))
        neg_emb_ego = self.spot_embedding(neg.to(self.device))
        users_emb = torch.mean(users_emb, dim=1)
        pos_emb = torch.mean(pos_emb, dim=1)
        neg_emb = torch.mean(neg_emb, dim=1)
        
        pos_scores = torch.sum(torch.mul(users_emb, pos_emb), dim=1) 
        neg_scores = torch.sum(torch.mul(users_emb, neg_emb), dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores-pos_scores))
        reg_loss = (1/2)*(users_emb_ego.norm(2).pow(2) + 
                            pos_emb_ego.norm(2).pow(2) +
                            neg_emb_ego.norm(2).pow(2))/len(users)
        del users_emb, pos_emb, neg_emb
        del users_emb_ego,pos_emb_ego, neg_emb_ego, pos_scores, neg_scores
        del users, pos, neg

        return loss, reg_loss

    def stageOne(self, users, pos, neg):
        spot_x_out, user_x_out = self.forward()
        loss, reg_loss = self.bpr_loss(spot_x_out, user_x_out, users, pos, neg)
        del spot_x_out, user_x_out
        return loss, reg_loss

    @torch.no_grad()
    def getRating(self):
        spot_out, user_out = self.forward()
        spot_out = torch.mean(spot_out, dim=1)
        user_out = torch.mean(user_out, dim=1)
        rating = torch.matmul(user_out, spot_out.T)
        return rating




if __name__=='__main__':
    with open('../config.yaml') as f:
        config = yaml.safe_load(f)
    config['k'] = 20
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

    n_spot = 42852
    n_user = 27094

    spot_x = torch.rand(n_spot, 128)
    user_x = torch.rand(n_user, 128)
    user_spot = torch.from_numpy(np.load('/home/yamanishi/project/trip_recommend/model/recommendation/data/train_edge.npy'))
    device = 'cuda:0'
    config['device'] = device
    config['model']['hidden_channels'] = 128
    config['model']['num_layers'] = 3
    config['model']['pre'] = True
    config['model']['mid'] = True
    config['model']['post'] = True
    config['model']['cat'] = False
    config['model']['mix'] = 'geo'
    model = MixGCN(config).to(device)
    spot_x_out, user_x_out = model()
    users = torch.randint(0,100, (20,))
    pos = torch.randint(0,100, (20,))
    neg = torch.randint(0, 100, (20,16))
    optim = torch.optim.Adam(model.parameters())
    loss, reg_loss = model.bpr_loss(spot_x_out, user_x_out, users, pos, neg)
    optim.zero_grad()
    loss = loss+1e-4*reg_loss
    loss.backward()
    optim.step()
    rating = model.getRating()
    print(rating.size())
    #print(spot_x_out, user_x_out)

    '''
    data = get_data(word=True, category=True, city=True, prefecture=True, multi=True)
    
    model = HeteroLightGCN(data,config)
    data.to(device)
    model.to(device)
    print(model)
    x_dict, out_dict= model(data.x_dict, data.edge_index_dict)
    print(x_dict)
    '''