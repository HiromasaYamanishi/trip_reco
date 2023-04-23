from __future__ import print_function, division
import matplotlib.pyplot as plt
import time
import copy
from collections import defaultdict
from typing import OrderedDict

import os
import sys
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
from torch_geometric.nn import GATConv,HGTConv, GCNConv, HANConv, SAGEConv, HeteroConv, GATv2Conv
from torch_geometric.nn import Linear, to_hetero, Sequential, GNNExplainer
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.loader import NeighborLoader, HGTLoader
from torch_geometric.nn.aggr import SumAggregation
from torch_scatter.scatter import scatter
from torch_scatter.utils import broadcast
import argparse
import yaml
sys.path.append('..')
from utils.path import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
def get_data(df, model_name='ResNet', multi=False, word=True, category=False, city=False, prefecture=False):
    data = HeteroData()
    path = Path()
    #df = pd.read_csv('/home/yamanishi/project/trip_recommend/data/df_experience.csv')
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
            img_emb_path = '/home/yamanishi/project/trip_recommend/data/graph/spot_img_emb_multi.npy'
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
        #data['word'].x = torch.from_numpy(np.load(path.word_embs_wiki_path)).float() #[num_spots, num_features]
        data['word'].x = torch.from_numpy(np.load(path.word_embs_finetune_path)).float()
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
        spot_city = torch.from_numpy(np.load(path.spot_city_path)).long()
        city_spot = torch.stack([spot_city[1], spot_city[0]]).long()
        city_city = torch.from_numpy(np.load(path.city_city_path)).long()
        data['spot','belongs','city'].edge_index = torch.from_numpy(np.load(path.spot_city_path)).long()
        data['city','revbelong','spot'].edge_index = city_spot
        data['city', 'cityadj','city'] = city_city

    if prefecture==True and city==True:
        data['pref'].x =  torch.from_numpy(np.load(path.pref_attr_path)).float()
        data['pref', 'prefadj', 'pref'].edge_index = torch.from_numpy(np.load(path.pref_pref_path)).long()
        city_pref = torch.from_numpy(np.load(path.city_pref_path))
        pref_city = torch.stack([city_pref[1], city_pref[0]]).long()
        data['city','belong','pref'].edge_index = city_pref
        data['pref', 'rebelong','city'].edge_index = pref_city
    
    
    return data


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

        
class MyHeteroConv(torch.nn.Module):
    def __init__(self, x_dict, edge_index_dict, out_channels, pre_channels=None, channel_dict=None):
        super().__init__()
        num_nodes = {k: v[0] for k, v in x_dict.items()}
        self.out_channels = out_channels

        if pre_channels== None:
            num_features = {k: v.size()[1] for k,v in x_dict.items()}
        else:
            num_features = {k: pre_channels for k in x_dict.keys()}

        if channel_dict is not None:
            for k,v in channel_dict.items():
                num_features[k] = v

        self.linear = nn.ModuleDict({})
        self.div = defaultdict(int)
        self.in_dim_dict = {}
        self.out_dim_dict = {}
        for k in edge_index_dict.keys():
            self.linear['__'.join(k) + '__source'] = Linear(num_features[k[0]], out_channels, False, weight_initializer='glorot')
            self.linear['__'.join(k) + '__target'] = Linear(num_features[k[-1]], out_channels, False, weight_initializer='glorot')

            self.in_dim_dict['__'.join(k)]=out_channels
            self.out_dim_dict['__'.join(k)]=out_channels
            self.div[k[-1]]+=1

        
        self.latest_source_embeddings = {}
        self.latest_target_embeddings = {}
        self.latest_messages = {}


    def count_unique(self, x):
        out, cnt = torch.unique(x, return_counts=True)
        x_ = torch.zeros_like(x)
        x_[out] = cnt
        return x_

    def get_latest_source_embeddings(self):
        return self.latest_source_embeddings
    
    def get_latest_target_embeddings(self):
        return self.latest_target_embeddings

    def get_latest_messages(self):
        return self.latest_messages

    def forward(self, x_dict, edge_index_dict,index, message_scale_dict=None, message_replacement_dict=None):
        x_dict_out = {}

        for k,v in edge_index_dict.items():
            edge_type = '__'.join(k)
            source, target = k[0], k[-1]
            source_x = self.linear[edge_type + '__source'](x_dict[source])
            target_x = self.linear[edge_type + '__target'](x_dict[target])

            self.in_dim_dict[edge_type] = self.out_channels
            self.out_dim_dict[edge_type] = self.out_channels

            source_index = v[0].reshape(-1)
            target_index = v[1].reshape(-1)
            out = torch.zeros_like(target_x).to(device)
            source_x = source_x[source_index]

            self.latest_source_embeddings[edge_type] = source_x
            self.latest_target_embeddings[edge_type] = target_x[target_index]
            #target_x_count = self.count_unique(target_index)
            #div = target_x_count[target_index].reshape(-1, 1)
            basis_messages = source_x

            if message_scale_dict is not None and message_scale_dict.get(edge_type):
                message_scale = message_scale_dict[edge_type][index]
                basis_messages = basis_messages * message_scale.unsqueeze(-1)
                #print('message scale',message_scale)
                

                if message_replacement_dict is not None:
                    #message_replacement = message_replacement_dict[edge_type+f'{index}']
                    message_replacement = message_replacement_dict[edge_type][index]
                    #print('message replacement', message_replacement)
                    if basis_messages.shape == message_replacement.shape:
                        basis_messages = basis_messages + (1 - message_scale).unsqueeze(-1) * message_replacement
                    else:
                        basis_messages = basis_messages + (1 - message_scale).unsqueeze(-1) * message_replacement.unsqueeze(
                            0)

                    print('attention', message_scale.requires_grad)
                    print('attention', message_replacement.requires_grad)
            
            self.latest_messages[edge_type] = basis_messages

            target_x = target_x + scatter(basis_messages, target_index, out=out, dim=0, reduce='mean')
            if x_dict_out.get(target):
                x_dict_out[target] += target_x
            else:
                x_dict_out[target] = target_x

        x_dict_out = {k: (v/self.div[k]).relu() for k,v in x_dict_out.items()}    

        return x_dict_out


    def l2_norm(self, x):
        return x/(torch.norm(x, dim=1) + 1e-6).view(-1, 1).expand(x.size())
        

class MyHetero(torch.nn.Module):
    def __init__(self, config, data):
        super().__init__()
        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict
        hidden_channels = config['model']['hidden_channels']
        out_channels = config['model']['out_channels']
        out_dim = config['model']['out_dim']
        multi = config['data']['multi']
        if multi==True:
            self.att = AttentionModule(input_dim=out_dim, num_heads=4, split=3, out_dim=out_dim)
            self.conv1 = MyHeteroConv(x_dict, edge_index_dict, out_channels=hidden_channels, pre_channels=None, channel_dict={'spot':out_dim})
        else:
            self.conv1 = MyHeteroConv(x_dict, edge_index_dict, out_channels=hidden_channels, pre_channels=None)

        self.conv2 = MyHeteroConv(x_dict, edge_index_dict, out_channels=hidden_channels, pre_channels= hidden_channels)
        self.conv3 = MyHeteroConv(x_dict, edge_index_dict, out_channels=hidden_channels, pre_channels= hidden_channels)
        self.linear = Linear(hidden_channels, out_channels)
        self.multi = multi

        self.gnn_layers = self.get_gnn()

        self.injected_message_scale_dict = None
        self.injected_message_replacement_dict = None
        self.prediction = None

    def forward(self, data, replacement):
        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict
        if self.multi == True:
            x_dict['spot'] = self.att(x_dict['spot'])
        if replacement:
            x_dict = self.conv1(x_dict, edge_index_dict, 0, self.injected_message_scale_dict, self.injected_message_replacement_dict)
            x_dict = self.conv2(x_dict, edge_index_dict, 1, self.injected_message_scale_dict, self.injected_message_replacement_dict)
            x_dict = self.conv3(x_dict, edge_index_dict, 2, self.injected_message_scale_dict, self.injected_message_replacement_dict)
        else:
            x_dict = self.conv1(x_dict, edge_index_dict, 0, None, None)
            x_dict = self.conv2(x_dict, edge_index_dict, 1, None, None)
            x_dict = self.conv3(x_dict, edge_index_dict, 2, None, None)
        out = self.linear(x_dict['spot'])
        if self.prediction==None:
            loss = None
        else:
            loss = torch.abs(out-self.prediction).sum()/len(out)
        return loss, out

    def get_gnn(self):
        return [self.conv1, self.conv2, self.conv3]
        

    #TODO: vertex_embeddingsからx_dictに変更
    #TODO: message_scaleからmessage_scale_dictに変更
    #TODO: これらを会うように実装
    def reset_injected_messages(self):
        self.injected_message_replacement_dict = None
        self.injected_message_scale_dict = None

    def is_adj_mat(self):
        return False

    def get_initial_layer_input(self, vertex_embeddings):
        return vertex_embeddings

    def inject_message_scale(self, message_scale, edge_type):
        self.injected_message_scale_dict = {}
        self.injected_message_scale_dict[edge_type] =message_scale

    def inject_message_replacement(self, message_replacement, edge_type):
        self.injected_message_replacement_dict = {}
        self.injected_message_replacement_dict[edge_type] = message_replacement
        '''
        self.injected_message_replacement_dict = nn.ParameterDict()
        #print(message_replacement
        for i,mr in enumerate(message_replacement):
            self.injected_message_replacement_dict[edge_type+f'{i}'] = mr
        '''
        #self.injected_message_replacement_dict[edge_type] = message_replacement # Have to store it in a list to prevent the pytorch module from thinking it is a parameter

    def get_vertex_embedding_dims_dict(self):
        return [layer.in_dim_dict for layer in self.gnn_layers]
    
    def get_vertex_embedding_dims(self, edge_type):
        return [layer.in_dim_dict[edge_type] for layer in self.gnn_layers]

    def get_message_dims_dict(self):
        return [layer.out_dim_dict for layer in self.gnn_layers]

    def get_message_dims(self, edge_type):
        return [layer.out_dim_dict[edge_type] for layer in self.gnn_layers]

    def get_latest_source_embeddings_dict(self):
        return [layer.get_latest_source_embeddings() for layer in self.gnn_layers]

    def get_latest_source_embeddings(self, edge_type):
        return [layer.get_latest_source_embeddings()[edge_type] for layer in self.gnn_layers]

    def get_latest_target_embeddings_dict(self):
        return [layer.get_latest_target_embeddings() for layer in self.gnn_layers]

    def get_latest_target_embeddings(self, edge_type):
        return [layer.get_latest_target_embeddings()[edge_type] for layer in self.gnn_layers]

    def get_latest_messages_dict(self):
        return [layer.get_latest_messages() for layer in self.gnn_layers]

    def get_latest_messages(self, edge_type):
        return [layer.get_latest_messages()[edge_type] for layer in self.gnn_layers]

    def count_latest_messages(self):
        return sum([layer_messages.numel()/layer_messages.shape[-1] for layer_messages in self.get_latest_messages()])

    def overwrite_label(self, prediction):
        self.prediction = prediction

    def count_layers(self):
        return self.n_layers

    def process_layer(self, x_dict, edge_index_dict, gnn_layer, message_scale_dict, message_replcement_dict):
        return gnn_layer(x_dict, edge_index_dict, message_scale_dict, message_replacement_dict)

    

    def forward_(self, vertex_embeddings, edges, edge_types, edge_direction_cutoff=None):
        layer_input = self.get_initial_layer_input(vertex_embeddings)

        for i, gnn_layer in enumerate(self.gnn_layers):
            if self.injected_message_scale is not None:
                message_scale_dict = self.injected_message_scale_dict[i]
            else:
                message_scale_dict = None

            if self.injected_message_replacement is not None:
                message_replacement_dict = self.injected_message_replacement_dict[0][i]
            else:
                message_replacement_dict = None

            layer_input = self.process_layer(x_dict=x_dict,
                                             edge_index_dict=edge_index_dict,
                                             gnn_layer=gnn_layer,
                                             message_scale_dict=message_scale_dict,
                                             message_replacement_dict=message_replacement_dict)

        output = layer_input

        if self.injected_message_scale is not None:
            self.injected_message_scale = None

        if self.injected_message_replacement is not None:
            self.injected_message_replacement = None

        return output

if __name__=='__main__':
    with open('/home/yamanishi/project/trip_recommend/analyzer/configuration/trip_popularity.yaml') as file:
        config = yaml.safe_load(file)
    data = get_data(config)
    model = MyHetero(config, data)
    data.to(device)
    model.to(device)
    print(model.get_gnn_layer())
    print(model(data))