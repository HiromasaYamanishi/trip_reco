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
import random
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


class BiMultiGCN(torch.nn.Module):
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
        self.user_conv = GCNLightConv(self.hidden_channels, self.hidden_channels, add_self_loops=False, bias=False)
        self.station_conv = GCNLightConv(self.hidden_channels, self.hidden_channels, add_self_loops=False, bias=False)
        self.city_conv = GCNLightConv(self.hidden_channels, self.hidden_channels, add_self_loops=False, bias=False)
        self.category_conv = GCNLightConv(self.hidden_channels, self.hidden_channels, add_self_loops=False, bias=False)

        self.user_A = torch.load(os.path.join(self.jalan_graph_dir, 'user_user_A.pt')).to_dense().to_sparse()
        self.user_A = self.user_A.coalesce().to(self.device)
        #self.spot_city_spot_sparse = torch.load(os.path.join(self.jalan_graph_dir, 'spot_city_spot_A.pt'))
        user_user = torch.load(os.path.join(self.jalan_graph_dir, 'user_user_A.pt')).to_dense()
        self.user_edge_index = torch.load('/home/yamanishi/project/trip_recommend/model/recommendation/data/user_user_edge_index.pt').to(self.device)
        self.user_edge_weight = torch.load('/home/yamanishi/project/trip_recommend/model/recommendation/data/user_user_edge_weight.pt').to(self.device)
        print(self.user_edge_index.size())
        print(self.user_edge_weight.size())
        print('loaded user A')
        
        self.city_A = torch.load(os.path.join(self.jalan_graph_dir, 'spot_city_spot_A.pt')).to_dense().to_sparse()
        self.city_A = self.city_A.coalesce().to(self.device)
        #self.spot_city_spot_sparse = torch.load(os.path.join(self.jalan_graph_dir, 'spot_city_spot_A.pt'))
        spot_city_spot = torch.load(os.path.join(self.jalan_graph_dir, 'spot_city_spot_A.pt')).to_dense()
        self.city_edge_index = torch.load('/home/yamanishi/project/trip_recommend/model/recommendation/data/spot_city_spot_edge_index.pt').to(self.device)
        self.city_edge_weight = torch.load('/home/yamanishi/project/trip_recommend/model/recommendation/data/spot_city_spot_edge_weight.pt').to(self.device)
        print(self.city_edge_index.size())
        print(self.city_edge_weight.size())
        print('loaded city A')
        self.station_A = torch.load(os.path.join(self.jalan_graph_dir, 'spot_station_spot_A.pt')).to_dense().to_sparse()
        self.station_A = self.station_A.coalesce().to(self.device)
        #self.spot_station_spot_sparse = torch.load(os.path.join(self.jalan_graph_dir, 'spot_station_spot_A.pt'))
        spot_station_spot = torch.load(os.path.join(self.jalan_graph_dir, 'spot_station_spot_A.pt')).to_dense()
        self.station_edge_index = torch.load('/home/yamanishi/project/trip_recommend/model/recommendation/data/spot_station_spot_edge_index.pt').to(self.device)
        self.station_edge_weight = torch.load('/home/yamanishi/project/trip_recommend/model/recommendation/data/spot_station_spot_edge_weight.pt').to(self.device)
        print(self.station_edge_index.size())
        print(self.station_edge_weight.size())
        print('loaded station A')
        self.category_A = torch.load(os.path.join(self.jalan_graph_dir, 'spot_dist_spot_A.pt')).to_dense().to_sparse()
        self.category_A = self.category_A.coalesce().to(self.device)
        spot_category_spot = torch.load(os.path.join(self.jalan_graph_dir, 'spot_dist_spot_A.pt')).to_dense()
        self.category_edge_index = torch.load('/home/yamanishi/project/trip_recommend/model/recommendation/data/spot_dist_spot_edge_index.pt').to(self.device)
        self.category_edge_weight = torch.load('/home/yamanishi/project/trip_recommend/model/recommendation/data/spot_dist_spot_edge_weight.pt').to(self.device)
        print(self.category_edge_index.size())
        print(self.category_edge_weight.size())
        print('loaded category A')

        self.jalan_spot_dir = '/home/yamanishi/project/trip_recommend/data/jalan/spot'
        self.df = pd.read_csv(os.path.join(self.jalan_spot_dir, 'experience_light.csv'))
        self.popularity = torch.from_numpy(self.df['review_count'].values)

    def dropout(self, edge_index, edge_weight):
        ratio = random.uniform(0.6, 0.9)
        index = np.random.randint(0, edge_weight.size(0),int(edge_weight.size(0)*ratio))
        return edge_index[:,index], edge_weight[index]

    def forward(self):
        spot_x = self.spot_embedding.weight
        user_x = self.user_embedding.weight
        category_edge_index, category_edge_weight = self.dropout(self.category_edge_index, self.category_edge_weight)
        spot_x_cat = self.category_conv(spot_x, category_edge_index, category_edge_weight)
        spot_x_cit = self.city_conv(spot_x, city_edge_index, city_edge_weight)
        spot_x_sta = self.station_conv(spot_x, station_edge_index, station_edge_weight)
        spot_x = spot_x + (spot_x_cat+spot_x_cit+spot_x_sta)/3
            
        user_edge_index, user_edge_weight = self.dropout(self.user_edge_index, self.user_edge_weight)
        user_x_user = self.user_conv(user_x, user_edge_index, user_edge_weight)
        user_x = user_x + user_x_user
        spot_x_out = spot_x
        user_x_out = user_x
        del spot_x_cat, spot_x_cit, spot_x_sta, user_x_user
        for i in range(self.num_layers):
            spot_x, user_x = self.layers[i](spot_x, user_x)
            spot_x_out = spot_x_out + spot_x
            user_x_out = user_x_out + user_x

        spot_x_out = spot_x_out/(self.num_layers+1)
        user_x_out = user_x_out/(self.num_layers+1)

        return spot_x_out, user_x_out

    def bpr_loss(self, spot_x_out, user_x_out, users, pos, neg):
        users_emb = user_x_out[users]
        pos_emb = spot_x_out[pos]
        neg_emb = spot_x_out[neg]
        users_emb_ego = self.user_embedding(users.to(self.device))
        pos_emb_ego = self.spot_embedding(pos.to(self.device))
        neg_emb_ego = self.spot_embedding(neg.to(self.device))
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
    n_user = 27904

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
    model = BiMultiGCN(config).to(device)
    #spot_x_out, user_x_out = model()
    users = torch.randint(0,100, (20,))
    pos = torch.randint(0,100, (20,))
    neg = torch.randint(0, 100, (20,))
    optim = torch.optim.Adam(model.parameters())
    #loss, reg_loss = model.bpr_loss(spot_x_out, user_x_out, users, pos, neg)
    loss, reg_loss = model.stageOne(users, pos, neg)
    print(loss, reg_loss)
    optim.zero_grad()
    loss = loss+1e-4*reg_loss
    loss.backward()
    optim.step()
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