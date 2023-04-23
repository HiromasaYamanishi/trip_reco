from __future__ import print_function, division
import matplotlib.pyplot as plt
import time
import copy
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
from torch_geometric.loader import NeighborLoader, HGTLoader
sys.path.append('..')
from collect_data.preprocessing.preprocess_refactor import Path
from utils import save_plot, save_cor, EarlyStopping


def get_data(model_name='ResNet', multi=False, word=True, category=False, city=False, prefecture=False):
    data = HeteroData()
    path = Path()
    #df = pd.read_csv('/home/yamanishi/project/trip_recommend/data/df_experience.csv')
    df = pd.read_csv(path.df_experience_path)
    #df['y']=(df['page_view']/df['page_view'].max())*100
    #df['y'] = np.log10(df['page_view']+1)
    #df['y'] = df['jalan_review_count']
    df['y'] = np.log10(df['jalan_review_count']+1)
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
        data['spot'].x = torch.from_numpy(np.load(path.spot_img_emb_path)).float()
    else:
        if model_name=='ResNet':
            img_emb_path = '/home/yamanishi/project/trip_recommend/data/graph/spot_img_emb_multi.npy'
            data['spot'].x = torch.from_numpy(np.load(img_emb_path))
        else:
            img_emb_path = os.path.join(path.data_graph_dir, f'spot_img_emb_multi_{model_name}.npy')
            data['spot'].x = torch.from_numpy(np.load(img_emb_path))

    #data['spot'].x = torch.from_numpy(np.load(path.spot_img_emb_clip_path)).float()

    num_words = np.load(path.word_embs_path).shape[0]
    #data['word'].x = torch.rand((num_words, 300))
    #data['word'].x = torch.from_numpy(np.load(path.word_emb_clip_path)).float()
    category_size = len(df['category'].unique())
    city_size = len(df['city'].unique())
    pref_size = len(df['都道府県'].unique())
    if word==True:    
    #data['spot','near','spot'].edge_index = torch.from_numpy(np.load(path.spot_spot_path)).long()
        data['word'].x = torch.from_numpy(np.load(path.word_embs_finetune_path)).float() #[num_spots, num_features]
        spot_word = torch.from_numpy(np.load(path.spot_word_path)).long()
        word_spot = torch.stack([spot_word[1], spot_word[0]]).long()
        data["spot", "relate", "word"].edge_index = torch.from_numpy(np.load(path.spot_word_path)).long() #[2, num_edges_describe]
        data['word', 'revrelate', 'spot'].edge_index = word_spot
    
    if category==True:
        data['category'].x =torch.rand(category_size,10)#torch.from_numpy(np.load(path.category_img_emb_path)).float()#torch.rand(category_size, 5)#torch.rand(category_size,10)
        spot_category = torch.from_numpy(np.load(path.spot_category_path)).long()
        category_spot = torch.stack([spot_category[1], spot_category[0]]).long()
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
        spot_pref = torch.from_numpy(np.load(path.spot_pref_path)).long()
        pref_spot = torch.stack([spot_pref[1], spot_pref[0]]).long()
        data['pref', 'prefadj', 'pref'].edge_index = torch.from_numpy(np.load(path.pref_pref_path)).long()
        city_pref = torch.from_numpy(np.load(path.city_pref_path))
        pref_city = torch.stack([city_pref[1], city_pref[0]]).long()
        data['city','belong','pref'].edge_index = city_pref
        data['pref', 'rebelong','city'].edge_index = pref_city
    
    
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

class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv((-1, -1), hidden_channels, add_self_loops = False)
        self.lin1 = Linear(-1, hidden_channels)
        self.conv2 = GATConv((-1, -1), hidden_channels, add_self_loops = False)
        self.lin2 = Linear(-1, hidden_channels)
        self.conv3 = GATConv((-1, -1), out_channels, add_self_loops = False)
        self.lin3 = Linear(-1, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index) + self.lin1(x)
        x = x.relu()
        x = self.conv2(x, edge_index) + self.lin2(x)
        x = x.relu()
        x = self.conv3(x, edge_index) + self.lin3(x)
        return x

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(128, 16)
        self.conv2 = GCNConv(16, 3)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

class HGT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers):
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, data.metadata(),
                           num_heads, group='sum')
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x).relu_()

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        return self.lin(x_dict['author'])

class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('spot', 'belong', 'city'): SAGEConv((-1,-1), hidden_channels),
                ('city','reblong','city'): SAGEConv((-1,-1), hidden_channels),
                ('spot', 'has', 'category'): SAGEConv((-1,-1), hidden_channels),
                ('category', 'rev_has', 'spot'): SAGEConv((-1,-1), hidden_channels),
                ('spot', 'near', 'spot') : GCNConv((-1), hidden_channels),
                ('word', 'revrelate', 'spot') : SAGEConv((-1, -1), hidden_channels), 
                ('spot', 'relate', 'word') : SAGEConv((-1, -1), hidden_channels),               
            }, aggr='sum')
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        return self.lin(x_dict['spot'])


class HAN(torch.nn.Module):
    def __init__(self,data, in_channels, out_channels, hidden_channels=64,heads=4,):
        super().__init__()
        self.han_conv1 = HANConv(in_channels, hidden_channels, heads=heads,
                                dropout=0.2, metadata=data.metadata())
        self.han_conv2 = HANConv(hidden_channels, hidden_channels, heads=heads,
                               dropout=0.2, metadata=data.metadata())
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        out = self.han_conv1(x_dict, edge_index_dict)
        out = out.relu()
        out = self.han_conv2(out, edge_index_dict)
        out = self.lin(out['spot'])
        return out


def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    #copy.deepcopy(model.state_dict())
    out = model(data.x_dict, data.edge_index_dict)
    mask = data['spot'].train_mask
    if isinstance(out, dict):
        loss = F.mse_loss(out['spot'][mask].flatten(), data["spot"].y[mask])/mask.sum()
    else:
        loss = F.mse_loss(out[mask].flatten(), data['spot'].y[mask])/mask.sum()
    loss.backward()
    optimizer.step()
    return model,float(loss)

@torch.no_grad()
def test(model,data):
    model.eval()
    pred = model(data.x_dict, data.edge_index_dict)

    losses = []
    for split in ['train_mask', 'valid_mask', 'test_mask']:
        mask = data['spot'][split]
        if isinstance(pred, dict):
            loss = F.mse_loss(pred['spot'][mask].flatten(), data['spot'].y[mask])/mask.sum()
        else:
            loss = F.mse_loss(pred[mask].flatten(), data['spot'].y[mask])/mask.sum()
        losses.append(float(loss))
    return losses

def train_epoch(model, data, epoch_num):
    optimizer = optim.Adam(model.parameters())
    for epoch in range(epoch_num):
        model, loss = train(model, optimizer, data)
        train_loss, val_loss, test_loss = test(model,data)
        print(f'Epoch: {epoch+1:03d}, Loss: {loss:.6f}, Train: {train_loss:.6f}, '
            f'Val: {val_loss:.6f}, Test: {test_loss:.6f}')

@torch.no_grad()
def calc_cor(model, data):
    model.eval()
    gt_all = []
    pred_all = []
    out = model(data.x_dict, data.edge_index_dict)
    path = Path()
    #df = pd.read_csv(path.df_experience_path)
    #df['pred'] = pd.Series(out['spot'].flatten().cpu().detach().numpy())
    #df.to_csv(path.df_experience_path)
    spot_name = pd.read_csv('/home/yamanishi/project/trip_recommend/data/spot_name.csv')
    spot_names = []
    for split in ['test_mask']:
        mask = data['spot'][split]
        gt = data['spot'].y[mask]
        if isinstance(out, dict):
            pred = out['spot'][mask].flatten()
        else:
            pred = out[mask]
        mask = mask.cpu().numpy()
        spot_names.append(spot_name['観光地名'][mask].values)
        gt_all.append(gt.cpu().detach().numpy().copy())
        pred_all.append(pred.cpu().detach().numpy().copy())

    gt_all = np.concatenate(gt_all)
    pred_all = np.concatenate(pred_all).reshape(-1)
    spot_names = np.concatenate(spot_names)
    save_cor(gt_all, pred_all,'gt: log(review_count)','pred: log(review_count)','cor',spot_names)
    return np.corrcoef(gt_all, pred_all)[0][1]

if __name__=='__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    print(device)
    path = Path()
    df = pd.read_csv(path.df_experience_path)
    data = get_data()
    print(data)
    data = data.to(device)
    dataloaders = get_dataloaders(data)

    model = Sequential('x, edge_index', [
        (SAGEConv((-1, -1), 128), 'x, edge_index -> x'),
        ReLU(inplace=True),
        (SAGEConv((-1, -1), 128), 'x, edge_index -> x'),
        ReLU(inplace=True),
        (SAGEConv((-1, -1), 128), 'x, edge_index -> x'),
        ReLU(inplace=True),
        (Linear(-1, 1), 'x -> x'),
    ])

    #model = GAT(hidden_channels=128, out_channels=1)
    model = to_hetero(model, data.metadata(), aggr='mean').to(device)
    print(model)
    copy.copy(model.state_dict())
    optimizer = optim.Adam(model.parameters())

    writer = SummaryWriter(log_dir='../data/log/graph')
    losses = {'train': [], 'val':[], 'test':[]}
    earlystopping = EarlyStopping(patience=10, verbose=True)
    epoch_num=200
    for epoch in range(epoch_num):
        #print(data.x_dict)
        #print(data.edge_index_dict)
        model, loss = train(model,optimizer, data)
        train_loss, val_loss, test_loss = test(model,data)
        #earlystopping(loss, model)
        losses['train'].append(train_loss)
        losses['val'].append(val_loss)
        losses['test'].append(test_loss)
        print(f'Epoch: {epoch+1:03d}, Loss: {loss:.6f}, Train: {train_loss:.6f}, '
            f'Val: {val_loss:.6f}, Test: {test_loss:.6f}')
        writer.add_scalar('loss', loss, epoch)
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('valid/loss', val_loss, epoch)
        writer.add_scalar('test/loss', test_loss, epoch)
        #if earlystopping.early_stop:
        #    print("Early Stopping!")
        #    break

    save_plot(epoch_num, 'loss', **losses)
    cor = calc_cor(df, model,data)
    with open('result.txt','a') as f:
        f.write(str(cor))
        f.write('\n')
    print(cor)
    '''
    explainer = GNNExplainer(model, epochs=200, return_type='log_prob')
    node_idx = 10
    x = data.x_dict
    edge_index = data.edge_index_dict
    node_feat_mask, edge_mask = explainer.explain_node(node_idx, x, edge_index)
    '''
    module = nn.ModuleDict([[cat,nn.Identity()] for cat in data.metadata()[0]])
    model.module_6 = module
    out = model(data.x_dict, data.edge_index_dict)
    out = out['spot'].cpu().detach().numpy()
    np.save('../data/embedding.npy', out)
    #writer.add_embedding(out['spot'], metadata=df['観光地名'].values, global_step=15)
