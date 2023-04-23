from __future__ import print_function, division
import matplotlib.pyplot as plt
import time
import copy
import math
from typing import OrderedDict

import os
import sys
from collections import defaultdict
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.nn import ReLU
import pandas as pd
from torch_geometric.data import HeteroData
import torch.nn.functional as F
from torch.nn.functional import relu
from torch_geometric.nn import GATConv,HGTConv, GCNConv, HANConv, SAGEConv, HeteroConv, GATv2Conv
from torch_geometric.nn import Linear, to_hetero, Sequential, GNNExplainer
from torch_geometric.loader import NeighborLoader, HGTLoader
sys.path.append('..')
sys.path.append('../../')
from data.jalan.preprocessing import Path
from utils import save_plot, save_cor, EarlyStopping
from torch_scatter.scatter import scatter
from torch.nn import LayerNorm


def get_data(model_name='ResNet', multi=False, word=True, category=False, city=False, prefecture=False):
    data = HeteroData()
    path = Path()
    df = pd.read_csv('/home/yamanishi/project/trip_recommend/data/jalan/spot/experience_light.csv')
    #df['y']=(df['page_view']/df['page_view'].max())*100
    #df['y'] = np.log10(df['page_view']+1)
    #df['y'] = df['jalan_review_count']
    df['y'] = np.log10(df['review_count'])
    #df['y'] = df['jalan_review_rate']
    mask = np.load(path.valid_idx_path)
    assert len(df)==len(mask)
    train_mask = torch.tensor(mask>=2)
    val_mask = torch.tensor(mask==0)
    test_mask = torch.tensor(mask==1)
    data['spot'].train_mask = train_mask
    data['spot'].valid_mask = val_mask
    data['spot'].test_mask = test_mask
    data['spot'].y = torch.from_numpy(df['y'].values).float()
    #data['spot'].x = torch.from_numpy(np.load(path.spot_img_emb_path)).float()#np.load('spot_emb.npy') #[num_spots, num_features]
    if multi==False:
        data['spot'].x = torch.from_numpy(np.load('/home/yamanishi/project/trip_recommend/data/jalan/graph/spot_img_emb_ResNet.npy')).float()
    else:
        if model_name=='ResNet':
            img_emb_path = '/home/yamanishi/project/trip_recommend/data/jalan/graph/spot_img_emb_multi_ResNet.npy'
            data['spot'].x = torch.from_numpy(np.load(img_emb_path))
        else:
            img_emb_path = os.path.join(path.data_graph_dir, f'spot_img_emb_multi_{model_name}.npy')
            data['spot'].x = torch.from_numpy(np.load(img_emb_path))

    #data['spot'].x = torch.from_numpy(np.load(path.spot_img_emb_clip_path)).float()

    #num_words = np.load(path.word_embs_wiki_path).shape[0]
    #data['word'].x = torch.rand((num_words, 300))
    #data['word'].x = torch.from_numpy(np.load(path.word_emb_clip_path)).float()
    #category_size = len(df['category'].unique())
    #city_size = len(df['city'].unique())
    #pref_size = len(df['都道府県'].unique())
    if word==True:    
    #data['spot','near','spot'].edge_index = torch.from_numpy(np.load(path.spot_spot_path)).long()
        data['word'].x = torch.from_numpy(np.load(path.word_embs_ensemble_path)).float() #[num_spots, num_features]
        #data['word'].x = torch.from_numpy(np.load(path.word_embs_finetune_path)).float()
        spot_word = torch.from_numpy(np.load(path.spot_word_path)).long()
        word_spot = torch.stack([spot_word[1], spot_word[0]]).long()
        data["spot", "relate", "word"].edge_index = torch.from_numpy(np.load(path.spot_word_path)).long() #[2, num_edges_describe]
        data['word', 'revrelate', 'spot'].edge_index = word_spot
    
    if category==True:
        #torch.from_numpy(np.load(path.category_img_emb_path)).float()#torch.rand(category_size, 5)#torch.rand(category_size,10)
        spot_category = torch.from_numpy(np.load(path.spot_category_path)).long()
        category_spot = torch.stack([spot_category[1], spot_category[0]]).long()
        category_size = len(spot_category[1].unique())
        data['category'].x =torch.nn.functional.one_hot(torch.arange(0,category_size), num_classes=category_size).float()
        data['spot', 'has', 'category'].edge_index = torch.from_numpy(np.load(path.spot_category_path)).long()
        data['category', 'revhas', 'spot'].edge_index = category_spot
    if city==True:
        data['city'].x = torch.from_numpy(np.load(path.city_attr_path)).float()#torch.rand(city_size, 5)
        data['city'].y = torch.from_numpy(np.load(path.city_popularity_path))
        data['city'].y = torch.log(data['city'].y)
        city_valid = np.load('/home/yamanishi/project/trip_recommend/data/jalan/graph/city_split.npy')
        data['city'].train_mask= torch.tensor(city_valid>=2)
        data['city'].valid_mask= torch.tensor(city_valid==0)
        data['city'].test_mask= torch.tensor(city_valid==1)
        spot_city = torch.from_numpy(np.load(path.spot_city_path)).long()
        city_spot = torch.stack([spot_city[1], spot_city[0]]).long()
        city_city = torch.from_numpy(np.load(path.city_city_path)).long()
        data['spot','belongs','city'].edge_index = torch.from_numpy(np.load(path.spot_city_path)).long()
        data['city','revbelong','spot'].edge_index = city_spot
        data['city', 'cityadj','city'] = city_city

    if prefecture==True and city==True:
        data['pref'].x =  torch.from_numpy(np.load(path.pref_attr_path)).float()
        data['pref'].y = torch.from_numpy(np.load(path.pref_popularity_path))
        data['pref'].y = data['pref'].y/data['pref'].y.max()
        pref_valid = np.load('/home/yamanishi/project/trip_recommend/data/jalan/graph/pref_split.npy')
        data['pref'].train_mask= torch.tensor(pref_valid>=2)
        data['pref'].valid_mask= torch.tensor(pref_valid==0)
        data['pref'].test_mask= torch.tensor(pref_valid==1)
        data['pref', 'prefadj', 'pref'].edge_index = torch.from_numpy(np.load(path.pref_pref_path)).long()
        city_pref = torch.from_numpy(np.load(path.city_pref_path))
        pref_city = torch.stack([city_pref[1], city_pref[0]]).long()
        data['city','belong','pref'].edge_index = city_pref
        data['pref', 'rebelong','city'].edge_index = pref_city

    data['user'].x = torch.load('./data/user_x.pt')
    user_spot = torch.from_numpy(np.load('/home/yamanishi/project/trip_recommend/model/recommendation/data/train_edge.npy'))
    spot_user = torch.stack([user_spot[1], user_spot[0]]).long()
    data['user', 'visit', 'spot'].edge_index = user_spot
    data['spot', 'revvisit', 'user'].edge_index = spot_user
    data['user'].user_pos = torch.load('./data/user_pos.pt')
    data['user'].user_neg = torch.load('./data/user_neg.pt')
    data['user'].item_pos = torch.load('./data/item_pos.pt')
    data['user'].item_neg = torch.load('./data/item_neg.pt')
    
    return data

def get_dataloaders(data):
    train_loader = NeighborLoader(
        data,
        num_neighbors=[512]*2,
        batch_size=128,
        input_nodes=('spot', data['spot'].train_mask)
    )
    valid_loader = NeighborLoader(
        data,
        num_neighbors=[512]*2,
        batch_size=128,
        input_nodes=('spot',data['spot'].valid_mask)
    )
    '''
    train_loader = HGTLoader(
        data,
        num_samples=[1024]*4,
        batch_size=128,
        input_nodes=('spot', data['spot'].train_mask)
    )
    valid_loader = HGTLoader(
        data,
        num_samples=[1024]*4,
        batch_size=128,
        input_nodes=('spot',data['spot'].valid_mask)
    )
    '''
    return {'train': train_loader, 'valid': valid_loader}

class MyHeteroConv(torch.nn.Module):
    def __init__(self, x_dict, edge_index_dict, out_channels, pre_channels=None, channel_dict=None):
        super().__init__()
        num_nodes = {k: v[0] for k, v in x_dict.items()}

        if pre_channels== None:
            num_features = {k: v.size()[1] for k,v in x_dict.items()}
        else:
            num_features = {k: pre_channels for k in x_dict.keys()}

        if channel_dict is not None:
            for k,v in channel_dict.items():
                num_features[k] = v

        self.linear = nn.ModuleDict({})
        self.div = defaultdict(int)
        for k in edge_index_dict.keys():
            self.linear['__'.join(k) + '__source'] = Linear(num_features[k[0]], out_channels, False, weight_initializer='glorot')
            self.linear['__'.join(k) + '__target'] = Linear(num_features[k[-1]], out_channels, False, weight_initializer='glorot')
            self.div[k[-1]]+=1

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
        return x_dict_out


    def l2_norm(self, x):
        return x/(torch.norm(x, dim=1) + 1e-6).view(-1, 1).expand(x.size())

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, input_dim, split, num_heads):
        super().__init__()
        self.input_dim = input_dim
        self.split = split
        self.num_heads = num_heads
        self.Wq = torch.nn.ParameterList()
        for i in range(num_heads):
            wq = torch.nn.Parameter(torch.rand(input_dim, input_dim//self.num_heads))
            self.reset_param(wq)
            self.Wq.append(wq)
        self.Wk = torch.nn.ParameterList()
        for i in range(num_heads):
            wk = torch.nn.Parameter(torch.rand(input_dim, input_dim//self.num_heads))
            self.reset_param(wk)
            self.Wk.append(wk)

        self.Wv = torch.nn.ParameterList()
        for i in range(num_heads):
            wv = torch.nn.Parameter(torch.rand(input_dim, input_dim//self.num_heads))
            self.reset_param(wv)
            self.Wv.append(wv)

    def reset_param(self,t):
        #Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv,stdv)

    def forward(self, x):
        out = []
        for i in range(self.num_heads):
            z = x.reshape(-1, self.split, self.input_dim)
            q = torch.matmul(z, self.Wq[i])
            k = torch.matmul(z, self.Wk[i])
            v = torch.matmul(z, self.Wv[i])
            k = torch.permute(k, (0,2,1))
            qk = torch.bmm(q,k)
            attn = F.softmax(qk/torch.sqrt(torch.tensor(self.input_dim)),dim=2)
            z = torch.matmul(attn, v)
            out.append(z)
        return torch.cat(out, dim=2)
        
class FFN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(relu(self.linear1(x)))

class Transformer(torch.nn.Module):
    def __init__(self, input_dim, num_heads, split,):
        super().__init__()
        self.input_dim = input_dim
        self.num_head = num_heads
        self.split = split
        self.att = MultiHeadAttention(input_dim=input_dim, split=split, num_heads=num_heads)
        self.ln1 = nn.LayerNorm(input_dim)
        self.do1 = nn.Dropout(0.2)
        self.ffn = FFN(input_dim * self.split, input_dim * self.split)
        self.ln2 = nn.LayerNorm(input_dim)
        self.do2 = nn.Dropout(0.2)

    def forward(self, x):
        y = self.att(x)
        y = y.reshape(-1,self.split*self.input_dim)
        y = self.do1(y)
        x = x + y
        y = self.ffn(x)
        y = self.do2(y)
        x = x + y
        x = x.reshape(-1, self.split, self.input_dim)
        x = torch.mean(x, dim=1)
        return x

class MyHeteroAttentionConv(torch.nn.Module):
    def __init__(self, x_dict, edge_index_dict, out_channels, pre_channels=None):
        super().__init__()
        num_nodes = {k: v[0] for k, v in x_dict.items()}

        if pre_channels== None:
            num_features = {k: v.size()[1] for k,v in x_dict.items()}
        else:
            num_features = {k: pre_channels for k in x_dict.keys()}

        self.linear = nn.ModuleDict({})
        for k in edge_index_dict.keys():
            self.linear['__'.join(k) + '__source'] = Linear(num_features[k[0]], out_channels, False, weight_initializer='glorot')
            self.linear['__'.join(k) + '__target'] = Linear(num_features[k[-1]], out_channels, False, weight_initializer='glorot')
        
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
            source_x_tmp = source_x[source_index] #(53850, 128)
            target_x_tmp = target_x[target_index] #(53850, 128)
            X = torch.cat([source_x_tmp, target_x_tmp], dim=1)

            a = self.attention['__'.join(k)]
            attention = torch.exp(self.LeakyReLU(torch.matmul(X, a)))
            '''
            div = torch.ones(target_x_tmp.size()[0],1).to(device)
            source_x_tmp = scatter(source_x_tmp, target_index, out=out, dim=0, reduce='sum')

            out_div = torch.zeros(target_x.size()[0], 1).to(device)
            div = scatter(div, target_index, out=out_div, dim=0, reduce='sum')
            div[div<1]=1
            '''

            source_x_tmp = source_x_tmp * attention
            source_x_tmp = scatter(source_x_tmp, target_index, out=out, dim=0, reduce='sum')

            out_att = torch.zeros(target_x.size()[0], 1).to(target_x.device)
            attention_div = scatter(attention, target_index, out=out_att,dim=0, reduce='sum')

            target_x = target_x + source_x_tmp/ (attention_div+1e-6)
            
            if x_dict_out.get(target)!=None:
                x_dict_out[target] += target_x
            else:
                x_dict_out[target] = target_x

        return x_dict_out

class MyReLU(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_dict):
        for k in x_dict.keys():
            x_dict[k] = x_dict[k].relu()
        return x_dict

class AttentionModule(torch.nn.Module):
    def __init__(self, input_dim, num_heads=4, split=1, out_dim=1000):
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.split = split
        self.out_dim = out_dim
        self.per_dim = out_dim//num_heads

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
        

class MyHetero(torch.nn.Module):
    def __init__(self, x_dict, edge_index_dict, hidden_channels, out_channels,out_dim,multi=False):
        super().__init__()
        if multi==True:
            self.att = AttentionModule(input_dim=out_dim, num_heads=4, split=5, out_dim=out_dim)
            #self.att = Transformer(input_dim=out_dim, num_heads=8, split=5,)
            #encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
            #self.att = nn.TransformerEncoder(encoder_layer, num_layers=1)
            self.conv1 = MyHeteroConv(x_dict, edge_index_dict, out_channels=hidden_channels, pre_channels=None, channel_dict={'spot':out_dim})
        else:
            self.conv1 = MyHeteroConv(x_dict, edge_index_dict, out_channels=hidden_channels, pre_channels=None)
        self.relu1 = MyReLU()
        self.conv2 = MyHeteroConv(x_dict, edge_index_dict, out_channels=hidden_channels, pre_channels= hidden_channels)
        self.relu2 = MyReLU()
        self.conv3 = MyHeteroConv(x_dict, edge_index_dict, out_channels=hidden_channels, pre_channels= hidden_channels)
        self.relu3 = MyReLU()
        self.conv4 = MyHeteroConv(x_dict, edge_index_dict, out_channels=hidden_channels, pre_channels= hidden_channels)
        self.relu4 = MyReLU()
        self.conv5 = MyHeteroConv(x_dict, edge_index_dict, out_channels=hidden_channels, pre_channels= hidden_channels)
        self.relu5 = MyReLU()
        self.conv6 = MyHeteroConv(x_dict, edge_index_dict, out_channels=hidden_channels, pre_channels= hidden_channels)
        self.relu6 = MyReLU()
        self.conv7 = MyHeteroConv(x_dict, edge_index_dict, out_channels=hidden_channels, pre_channels= hidden_channels)
        self.relu7 = MyReLU()
        #self.conv8 = MyHeteroConv(x_dict, edge_index_dict, out_channels=hidden_channels, pre_channels= hidden_channels)
        #self.relu8 = MyReLU()
        #self.conv9 = MyHeteroConv(x_dict, edge_index_dict, out_channels=hidden_channels, pre_channels= hidden_channels)
        #self.relu9 = MyReLU()
        #self.conv10 = MyHeteroConv(x_dict, edge_index_dict, out_channels=hidden_channels, pre_channels= hidden_channels)
        #self.relu10 = MyReLU()
        self.linears = torch.nn.ModuleDict()
        for k in x_dict.keys():
            self.linears[k] = Linear(hidden_channels, out_channels)
        self.multi = multi

    def forward(self, x_dict, edge_index_dict):
        if self.multi == True:
            x_dict['spot'] = self.att(x_dict['spot'])
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = self.relu1(x_dict)
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = self.relu2(x_dict)
        x_dict = self.conv3(x_dict, edge_index_dict)
        x_dict = self.relu3(x_dict)
        x_dict = self.conv4(x_dict, edge_index_dict)
        x_dict = self.relu4(x_dict)
        x_dict = self.conv5(x_dict, edge_index_dict)
        x_dict = self.relu5(x_dict)
        x_dict = self.conv6(x_dict, edge_index_dict)
        x_dict = self.relu6(x_dict)
        x_dict = self.conv7(x_dict, edge_index_dict)
        x_dict = self.relu7(x_dict)
        #x_dict = self.conv8(x_dict, edge_index_dict)
        #x_dict = self.relu8(x_dict)
        #x_dict = self.conv9(x_dict, edge_index_dict)
        #x_dict = self.relu9(x_dict)
        #x_dict = self.conv10(x_dict, edge_index_dict)
        #x_dict = self.relu10(x_dict)
        out_dict = {}
        for k in x_dict.keys():
            out_dict[k] = self.linears[k](x_dict[k])
        
        return x_dict, out_dict


class Trainer:
    def __init__(self, device):
        self.device = device
        self.neg_samples = np.load('./data/neg_samples.npy')
        self.df = pd.read_csv('/home/yamanishi/project/trip_recommend/data/jalan/spot/experience_light.csv')
        self.spot_names = self.df['spot_name'].values
        print('loaded neg samples')

    def train(self, model, optimizer, data, epoch):
        model.train()
        optimizer.zero_grad()
        #copy.deepcopy(model.state_dict())
        x_dict, out_dict = model(data.x_dict, data.edge_index_dict)
        mask = data['spot'].train_mask
        if isinstance(x_dict, dict):
            spot_mask = data['spot'].train_mask
            loss_spot = F.mse_loss(out_dict['spot'][spot_mask].flatten(), data['spot'].y[spot_mask])#/spot_mask.sum()
            city_mask = data['city'].train_mask
            loss_city = F.mse_loss(out_dict['city'][city_mask].flatten(), data['city'].y[city_mask])#/city_mask.sum()
            pref_mask = data['pref'].train_mask
            loss_pref = F.mse_loss(out_dict['pref'][pref_mask].flatten(), data['pref'].y[pref_mask])#/pref_mask.sum()

            user_emb = x_dict['user'][data['user'].user_pos]
            item_pos = x_dict['spot'][data['user'].item_pos]
            item_neg = x_dict['spot'][torch.from_numpy(self.neg_samples[epoch%1000]).to(self.device)]

            pos_scores = torch.sum(torch.mul(user_emb , item_pos, ), dim=1)
            neg_scores = torch.sum(torch.mul(user_emb, item_neg, ), dim=1)
            loss_rec = torch.mean(torch.nn.functional.softplus(neg_scores-pos_scores))
            alpha, beta = 0,0
            loss = alpha*loss_city+beta*loss_spot + loss_rec
            #loss = loss_rec
            loss = loss.float()
        else:
            loss = F.mse_loss(out[mask].flatten(), data['spot'].y[mask])/mask.sum()
        loss.backward()
        optimizer.step()
        return model,float(loss)

    @torch.no_grad()
    def test(self, model,data):
        model.eval()
        x_dict, out_dict = model(data.x_dict, data.edge_index_dict)
        alpha=1
        losses = []
        for split in ['train_mask', 'valid_mask', 'test_mask']:
            mask = data['spot'][split]
            if isinstance(x_dict, dict):
                spot_mask = data['spot'][split]
                loss_spot = F.mse_loss(out_dict['spot'][spot_mask].flatten(), data['spot'].y[spot_mask])/mask.sum()
                city_mask = data['city'][split]
                loss_city = F.mse_loss(out_dict['city'][city_mask].flatten(), data['city'].y[city_mask])/mask.sum()
                pref_mask = data['pref'][split]
                loss_pref = F.mse_loss(out_dict['pref'][pref_mask].flatten(), data['pref'].y[pref_mask])/mask.sum()
                loss = alpha*loss_city+loss_spot

                if split =='train_mask':
                    user_emb = x_dict['user'][data['user'].user_pos]
                    item_pos = x_dict['spot'][data['user'].item_pos]
                    item_neg = x_dict['spot'][data['user'].item_neg]

                    pos_scores = torch.sum(user_emb * item_pos, axis=1)
                    neg_scores = torch.sum(user_emb * item_neg, axis=1)
                    loss_rec = torch.mean(torch.nn.functional.softplus(neg_scores-pos_scores))

                loss+=loss_rec
            else:
                loss = F.mse_loss(out_dict[mask].flatten(), data['spot'].y[mask])/mask.sum()
            losses.append(float(loss))
        return losses

    def train_epoch(self, model, data, epoch_num):
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        for epoch in range(epoch_num):
            model, loss = self.train(model, optimizer, data, epoch)
            train_loss, val_loss, test_loss = self.test(model,data)
            print(f'Epoch: {epoch+1:03d}, Loss: {loss:.6f}, Train: {train_loss:.6f}, '
                f'Val: {val_loss:.6f}, Test: {test_loss:.6f}')

            if epoch%100==0:
                prec, recall = self.evaluate_rec(model, data, k=20)
                print(f'Precision: {prec:.6f}, Recall: {recall:.6f}')

        prec, recall=self.evaluate_rec(model, data, k=20)
        print(f'Precision: {prec:.6f}, Recall: {recall:.6f}')

        cor = self.calc_cor(model, data)
        print(f'cor: {cor:.4f}')

    @torch.no_grad()
    def evaluate_rec(self, model, data, k=20):
        with open('./data/test_item.pkl', 'rb') as f:
            test_item = pickle.load(f)
        x_dict, out_dict = model(data.x_dict, data.edge_index_dict)
        rating = torch.matmul(x_dict['user'], x_dict['spot'].T)
        #rating = torch.nn.functional.sigmoid(rating)
        rating[data['user'].user_pos, data['user'].item_pos]=-1e9
        _, rating_K = torch.topk(rating, k=k)
        print('rating_K',rating_K)
        recall, prec = 0, 0
        rs = []
        gts = []
        index = np.random.choice(40000, 10)
        for i, rk in enumerate(rating_K):
            gt = test_item[i]
            r = torch.isin(torch.tensor(gt).to(self.device), rk).sum()
            if len(gt)==0:continue
            
            prec+=r/k
            recall+=r/len(gt)
            if i in index:
                train_spots = data['user'].item_pos[data['user'].user_pos==i]
                print('train:', self.spot_names[train_spots.cpu().numpy()])
                print('gt:', self.spot_names[gt])
                print('pred:', self.spot_names[rk.cpu().numpy()])
            rs.append(r.item())
            gts.append(len(gt))
        prec/=len(rating_K)
        recall/=len(rating_K)

        return prec, recall

    @torch.no_grad()
    def calc_cor(self, model, data):
        model.eval()
        x_dict, out_dict = model(data.x_dict, data.edge_index_dict)
        for split in ['test_mask']:
            mask = data['spot'][split]
            gt = data['spot'].y[mask].cpu().numpy()
            if isinstance(out_dict, dict):
                pred = out_dict['spot'][mask].reshape(-1).cpu().numpy()
            else:
                pred = out_dict[mask].reshape(-1).cpu().numpy()
            cor = np.corrcoef(gt, pred)[0][1]
        return cor

    @torch.no_grad()
    def calc_cor_save(self, model, data):
        model.eval()
        gt_all_spot, pred_all_spot=[], []
        gt_all_city, pred_all_city = [], []
        out = model(data.x_dict, data.edge_index_dict)
        path = Path()
        #df = pd.read_csv(path.df_experience_path)
        #df['pred'] = pd.Series(out['spot'].flatten().cpu().detach().numpy())
        #df.to_csv(path.df_experience_path)

        for split in ['test_mask']:
            spot_mask = data['spot'][split]
            gt_spot = data['spot'].y[spot_mask]
            city_mask = data['city'][split]
            gt_city = data['city'].y[city_mask]

            if isinstance(out, dict):
                pred_spot = out['spot'][spot_mask].flatten()
                pred_city = out['city'][city_mask].flatten()
            else:
                mask = data['spot'][split]
                pred = out[mask]
            gt_all_spot.append(gt_spot.cpu().detach().numpy().copy())
            pred_all_spot.append(pred_spot.cpu().detach().numpy().copy())
            gt_all_city.append(gt_city.cpu().detach().numpy().copy())
            pred_all_city.append(pred_city.cpu().detach().numpy().copy())

        gt_all_spot = np.concatenate(gt_all_spot)
        pred_all_spot = np.concatenate(pred_all_spot).reshape(-1)
        gt_all_city = np.concatenate(gt_all_city)
        pred_all_city = np.concatenate(pred_all_city).reshape(-1)
        save_cor(gt_all_spot, pred_all_spot,'gt: log(review_count)','pred: log(review_count)','cor')
        return np.corrcoef(gt_all_spot, pred_all_spot)[0][1], np.corrcoef(gt_all_city, pred_all_city)[0][1]
            


if __name__=='__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    path = Path()
    df = pd.read_csv(path.df_experience_light_path)
    data = get_data(category=True, city=True, prefecture=True, multi=True)

    data.to(device)
    print(data)
    
    model = MyHetero(data.x_dict, data.edge_index_dict, hidden_channels=128, out_channels=1, out_dim=512,multi=True)
    model.to(device)
    
    #np.save('../data/embedding_new.npy', out)
    '''
    model = Sequential('x, edge_index', [
        (SAGEConv((-1, -1), 128), 'x, edge_index -> x'),
        ReLU(inplace=True),
        (SAGEConv((-1, -1), 128), 'x, edge_index -> x'),
        ReLU(inplace=True),
        (SAGEConv((-1, -1), 128), 'x, edge_index -> x'),
        ReLU(inplace=True),
        (Linear(-1, 1), 'x -> x'),
    ])
    model = to_hetero(model, data.metadata(), aggr='mean').to(device)
    '''
    #model.load_state_dict(torch.load('./model.pth'))
    trainer = Trainer(device)
    trainer.train_epoch(model, data, epoch_num=50)
    #torch.save(model.state_dict(), '../data/model/model_new.pth')
    #spot_cor, city_cor = trainer.calc_cor_save(model, data)
    #print(f'spot cor is {spot_cor:.5f}, city cor is {city_cor:.5f}')
        #    print("Early Stopping!")
        #    break

    '''
    module = nn.ModuleDict([[cat,nn.Identity()] for cat in data.metadata()[0]])
    model.module_6 = module
    out = model(data.x_dict, data.edge_index_dict)
    out = out['spot'].cpu().detach().numpy()
    np.save('../data/embedding.npy', out)
    '''
    #writer.add_embedding(out['spot'], metadata=df['観光地名'].values, global_step=15)
