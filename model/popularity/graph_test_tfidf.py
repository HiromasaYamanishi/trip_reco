from __future__ import print_function, division
import matplotlib.pyplot as plt
import time
import copy
from typing import OrderedDict

import os
import pickle
import sys
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
from torch_geometric.nn import GATConv,HGTConv, GCNConv, HANConv, SAGEConv
from torch_geometric.nn import Linear, to_hetero, Sequential
from torch_geometric.loader import NeighborLoader, HGTLoader
sys.path.append('..')
from collect_data.preprocessing.preprocess_refactor import Path, PreProcessing
from graph_test import get_data, train, test, HAN

os.environ['CUDA_VISIBLE_DEVICES'] = '7'
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
print(device)


def test_model(experience_name):
    data = get_data()
    print(data)
    data = data.to(device)
    model = HAN(data, in_channels=-1, hidden_channels=64, out_channels=1)
    model = model.to(device)
    print(model)
    #print(model.state_dict())
    
    optimizer = optim.Adam(model.parameters())

    writer = SummaryWriter(log_dir=os.path.join('../data/log/graph/tfidf', experience_name))

    train_losses, val_losses, test_losses = [],[],[]
    for epoch in range(500):

        model, loss = train(model,optimizer, data)
        train_loss, val_loss, test_loss = test(model,data)
        print(f'Epoch: {epoch+1:03d}, Loss: {loss:.6f}, Train: {train_loss:.6f}, '
            f'Val: {val_loss:.6f}, Test: {test_loss:.6f}')
        writer.add_scalar('loss', loss, epoch)
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('valid/loss', val_loss, epoch)
        writer.add_scalar('test/loss', test_loss, epoch)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        test_losses.append(test_loss)

    losses = {'train': train_losses, 'val': val_losses, 'test': test_losses}
    return losses


if __name__=='__main__':
    PP = PreProcessing()
    path = Path()
    df = pd.read_csv(path.df_experience_path)
    loss_all = {}
    for column in ['jalan_review', 'wiki_text', 'text_joined']:
        for hinshi in [['名詞'],['名詞','形容詞']]:
            print(column+''.join(hinshi))
            df = PP.Tokenize.tokenize_text(df, column = column, hinshi = hinshi)
            for k in [5,10, 20]:
                hinshi_=''.join(hinshi)
                experience_name = f'{column}_{k}_{hinshi_}'
                print(experience_name)
                df = PP.TFIDF.tfidf_topk(df, k=k)
                df = PP.ConstructGraph.construct_after_tokenize(df)
                df.to_csv(path.df_experience_path)
                hinshi_=''.join(hinshi)
                losses=test_model(experience_name)
                loss_all[experience_name] = losses

    with open('../data/graph_tfidf_loss.pkl','wb') as f:
        pickle.dump(loss_all, f)






