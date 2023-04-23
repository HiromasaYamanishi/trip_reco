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
from graph_test import get_data
from utils import save_plot, save_cor, EarlyStopping


os.environ['CUDA_VISIBLE_DEVICES'] = '5'
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
print(device)

class Encoder(torch.nn.Module):
    def __init__(self, data, hidden_channels, out_channels):
        super().__init__()
        self.model = Sequential('x, edge_index', [
            (SAGEConv((-1, -1), hidden_channels), 'x, edge_index -> x'),
            ReLU(inplace=True),
            (SAGEConv((-1, -1), hidden_channels), 'x, edge_index -> x'),
            ReLU(inplace=True),
            (SAGEConv((-1, -1), hidden_channels), 'x, edge_index -> x'),
            ReLU(inplace=True),
            (Linear(-1, out_channels), 'x -> x'),
        ])
        self.model = to_hetero(self.model, data.metadata(), aggr='mean')

    def forward(self, data):
        out = self.model(data.x_dict, data.edge_index_dict)
        return out

class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_dict, edge_pairs):
        predicted_edges = {}
        for edge_pair in edge_pairs:
            type1, type2 = edge_pair
            predicted_edges[edge_pair] = torch.sigmoid(torch.matmul(x_dict[type1], x_dict[type2].T))
        
        return predicted_edges
        
class LinkPrediction(torch.nn.Module):
    def __init__(self, data, edge_pairs, hidden_channels=128, out_channels=32):
        super().__init__()
        self.encoder = Encoder(data, hidden_channels, out_channels)
        self.decoder = Decoder()
        self.edge_pairs = edge_pairs

    def forward(self, data):
        x_dict_out = self.encoder(data)
        predicted_edges = self.decoder(x_dict_out, self.edge_pairs)
        return predicted_edges


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


if __name__=='__main__':
    data = get_data(word=True, category=True, city=True, prefecture=True)
    edge_pairs = [('city','category')]
    model = LinkPrediction(data, edge_pairs)
    predicted_edges=model(data)
    print(predicted_edges[('city', 'category')].size())
