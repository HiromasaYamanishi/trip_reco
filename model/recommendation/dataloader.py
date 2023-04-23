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


def get_data(config, model_name='ResNet', multi=True,):
    word = config['data']['word']
    category = config['data']['category']
    city = config['data']['city']
    prefecture = config['data']['pref']
    data = HeteroData()
    path = Path()
    df = pd.read_csv('/home/yamanishi/project/trip_recommend/data/jalan/spot/experience_light.csv')
    #df['y']=(df['page_view']/df['page_view'].max())*100
    #df['y'] = np.log10(df['page_view']+1)
    #df['y'] = df['jalan_review_count']
    df['y'] = np.log10(df['review_count'])
    #df['y'] = df['jalan_review_rate']
    mask = np.load(path.valid_idx_path)
    print('mask is', mask) 
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

        data['spot'].spot_pref = torch.from_numpy(np.load(path.spot_pref_path))

    data['user'].x = torch.load('./data/user_x.pt')
    torch.nn.init.normal_(data['user'].x, 0, config['data']['init_std'])
    user_spot = torch.from_numpy(np.load('/home/yamanishi/project/trip_recommend/model/recommendation/data/train_edge.npy'))
    spot_user = torch.stack([user_spot[1], user_spot[0]]).long()
    data['user', 'visit', 'spot'].edge_index = user_spot
    data['spot', 'revvisit', 'user'].edge_index = spot_user
    data['user'].user_pos = torch.load('./data/user_pos.pt')
    data['user'].user_neg = torch.load('./data/user_neg.pt')
    data['user'].item_pos = torch.load('./data/item_pos.pt')
    data['user'].item_neg = torch.load('./data/item_neg.pt')
    if config['model']['model_type'] =='lgcn':
        data['user'].x = torch.nn.Embedding(num_embeddings=len(data['user'].x), embedding_dim=config['model']['hidden_channels'])
        data['spot'].x = torch.nn.Embedding(num_embeddings=len(data['spot'].x), embedding_dim=config['model']['hidden_channels'])
        nn.init.normal_(data['user'].x.weight, std=0.1)
        nn.init.normal_(data['spot'].x.weight, std=0.1)
    return data

if __name__=='__main__':
    data = get_data()
    print(data)