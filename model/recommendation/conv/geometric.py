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
from torch_geometric.nn.conv.gatlight_conv import GATLightConv
from torch_geometric.nn.conv import GATConv
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
#from get_data import get_data


class AnyModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_users = 27094
        self.m_items = 42852
        self.n_category = 290
        self.n_city = 1709
        self.config = config
        self.device = config['device']
        self.user_spot = torch.from_numpy(np.load('/home/yamanishi/project/trip_recommend/model/recommendation/data/train_edge.npy')).to(self.device)
        self.spot_user =  torch.stack([self.user_spot[1], self.user_spot[0]]).long()
        self.spot_category = torch.from_numpy(np.load('/home/yamanishi/project/trip_recommend/data/jalan/graph/spot_category.npy')).to(self.device)
        self.category_spot =  torch.stack([self.spot_category[1], self.spot_category[0]]).long()
        self.spot_category[1]+=self.n_users+self.m_items
        self.category_spot[0]+=self.n_users+self.m_items
        self.spot_city = torch.from_numpy(np.load('/home/yamanishi/project/trip_recommend/data/jalan/graph/spot_city.npy')).to(self.device)
        self.city_spot =  torch.stack([self.spot_city[1], self.spot_city[0]]).long()
        self.spot_city[1]+=self.n_users+self.m_items+self.n_category
        self.city_spot[0]+=self.n_users+self.m_items+self.n_category
        self.hidden_channels = config['model']['hidden_channels']
        self.user_spot[0]+=self.m_items
        self.spot_user[1]+=self.m_items
        self.train_edge_index = torch.cat([self.user_spot, self.spot_user, self.category_spot, self.spot_category, self.city_spot, self.spot_city], dim=1)
        self.all_embedding = torch.nn.Embedding(num_embeddings=self.m_items+self.n_users+self.n_category+self.n_city, embedding_dim=self.hidden_channels) #[spot, user]
        torch.nn.init.normal_(self.all_embedding.weight, std=0.1)
        self.num_layers = config['model']['num_layers']
        self.layers = torch.nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(GCNLightConv(self.hidden_channels, self.hidden_channels, add_self_loops=False, bias=False))
            #self.layers.append(GATLightConv(self.hidden_channels, self.hidden_channels, add_self_loops=False, bias=False))
        self.jalan_graph_dir = '/home/yamanishi/project/trip_recommend/data/jalan/graph'

        self.jalan_spot_dir = '/home/yamanishi/project/trip_recommend/data/jalan/spot'
        self.df = pd.read_csv(os.path.join(self.jalan_spot_dir, 'experience_light.csv'))
        self.popularity = torch.from_numpy(self.df['review_count'].values)
        

    def forward(self):
        x = self.all_embedding.weight
        x_out = x
        for i in range(self.num_layers):
            x = self.layers[i](x, self.train_edge_index)
            x_out = x_out+x
        x_out = x_out/(1+self.num_layers)

        return x_out

    def bpr_loss(self, spot_x_out, user_x_out, users, pos, neg):
        users_emb = user_x_out[users]
        pos_emb = spot_x_out[pos]
        neg_emb = spot_x_out[neg]

        users_emb_ego = self.all_embedding((users+self.m_items).to(self.device))
        pos_emb_ego = self.all_embedding(pos.to(self.device))
        neg_emb_ego = self.all_embedding(neg.to(self.device))
        pos_scores = torch.sum(torch.mul(users_emb, pos_emb), dim=1)
        neg_scores = torch.sum(torch.mul(users_emb, neg_emb), dim=1)

        cat_cit_ind=torch.arange(self.n_users + self.m_items, self.n_users + self.m_items +self.n_category+self.n_city, 1)
        cat_cit_emb = self.all_embedding(cat_cit_ind.to(self.device))
        loss = torch.mean(torch.nn.functional.softplus((neg_scores-pos_scores)))
        reg_loss = (1/2)*(users_emb_ego.norm(2).pow(2) + 
                            pos_emb_ego.norm(2).pow(2) +
                            neg_emb_ego.norm(2).pow(2))/len(users)
        reg_loss = reg_loss+(1/2)*(cat_cit_emb.norm(2).pow(2))/len(cat_cit_ind)
        del users_emb, pos_emb, neg_emb
        del users_emb_ego,pos_emb_ego, neg_emb_ego, pos_scores, neg_scores
        del users, pos, neg

        return loss, reg_loss


    def stageOne(self, users, pos, neg):
        x = self.forward()
        spot_x = x[:self.m_items]
        user_x = x[self.m_items:self.m_items+self.n_users]
        loss, reg_loss = self.bpr_loss(spot_x, user_x, users, pos, neg)
        return loss, reg_loss

    @torch.no_grad()
    def getRating(self):
        x = self.forward()
        spot_out = x[:self.m_items]
        user_out = x[self.m_items:self.m_items+self.n_users]
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
    device = 'cuda:1'
    config['device'] = device
    config['model']['hidden_channels'] = 128
    config['model']['num_layers'] = 3
    config['model']['pre'] = True
    config['model']['mid'] = False
    config['model']['post'] = False
    config['model']['cat'] = False
    model = AnyModel(config).to(device)
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