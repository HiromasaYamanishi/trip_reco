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
from dataloader import get_data
import wandb
from torch.autograd import Variable
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import RocCurveDisplay, roc_curve, auc

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

class Trainer:
    def __init__(self, config,):
        self.config = config
        self.device = config['device']
        #self.neg_samples = np.load('./data/neg_samples.npy')
        self.df = pd.read_csv('/home/yamanishi/project/trip_recommend/data/jalan/spot/experience_light.csv')
        self.spot_names = self.df['spot_name'].values
        self.max_cor = 0
        self.neg_spots = np.load('./data/unappear_spots.npy')
        self.W = torch.nn.parameter.Parameter(torch.rand(128, 128))
        torch.nn.init.normal_(self.W, 0.1)
        self.W=self.W.to(self.device)
        self.class_label = np.load('/home/yamanishi/project/trip_recommend/model/popularity_final/data/label.npy')
        self.mask = np.load('/home/yamanishi/project/trip_recommend/model/popularity_final/data/mask.npy')
        self.split = np.load('/home/yamanishi/project/trip_recommend/model/popularity_final/data/split.npy')
        self.train_mask = np.load('/home/yamanishi/project/trip_recommend/model/popularity_final/data/train_mask.npy')
        self.valid_mask = np.load('/home/yamanishi/project/trip_recommend/model/popularity_final/data/valid_mask.npy')
        self.test_mask = np.load('/home/yamanishi/project/trip_recommend/model/popularity_final/data/test_mask.npy')                
        self.train_label = self.class_label[self.train_mask]
        self.valid_label = self.class_label[self.valid_mask]
        self.test_label = self.class_label[self.test_mask]
        local_binarizer = LabelBinarizer().fit(self.train_label)
        y_onehot_test = local_binarizer.transform(self.test_label)
        self.test_label_onehot = y_onehot_test
        print(self.test_label_onehot)
        print(self.W.device)
        print('loaded neg samples')
        #wandb.init('popularity', config=config)
        self.loss = torch.nn.CrossEntropyLoss()
        #self.loss = FocalLoss(gamma=2)
        self.max_auc = 0
        self.early_count = 0
        self.earyl_stop=False
        self.valid_loss=1e6

    def train_minibatch(self, model, optimizer, data, epoch):
        class_label = torch.tensor(self.class_label).to(self.device) #num_spot
        train_mask = torch.tensor(self.train_mask).to(self.device) #num_cand*0.9
        model.train()
        #mask = data['spot'].train_mask.nonzero().reshape(-1)
        shuffle_indices = np.arange(len(train_mask))
        np.random.shuffle(shuffle_indices)
        train_mask = train_mask[shuffle_indices]
        batch_size = 4096
        total_loss = 0
        for i in range(0, len(train_mask), batch_size):
            mask_tmp = train_mask[i:min(i+batch_size, len(train_mask))]
            x_dict, out_dict = model(data.x_dict, data.edge_index_dict)
            loss = self.loss(out_dict['spot'][mask_tmp], class_label[mask_tmp])
            #loss = F.cross_entropy(out_dict['spot'][mask_tmp], class_label[mask_tmp])
            #loss = F.mse_loss(out_dict['spot'][mask_tmp].flatten(), data['spot'].y[mask_tmp].to(out_dict['spot'].device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss+=loss*batch_size
        total_loss/=len(train_mask)
        return model, total_loss

    @torch.no_grad()
    def test(self, model,data):
        class_label = torch.tensor(self.class_label).to(self.device) #num_spot
        model.eval()
        x_dict, out_dict = model(data.x_dict, data.edge_index_dict)
        alpha=1
        losses = []
        accuracies = []
        train_mask = torch.tensor(self.train_mask).to(self.device)
        valid_mask = torch.tensor(self.valid_mask).to(self.device)
        test_mask = torch.tensor(self.test_mask).to(self.device)
        for i,mask_tmp in enumerate([train_mask, valid_mask, test_mask]):
            if isinstance(x_dict, dict):
                logit = out_dict['spot'][mask_tmp]
                loss = F.cross_entropy(logit, class_label[mask_tmp])
                if i==1 and loss>self.valid_loss:
                    self.early_count+=1
                elif i==1 and loss<self.valid_loss:
                    self.valid_loss=loss
                    self.best_model_state = model.state_dict().copy()
                    self.early_count=0
                pred_label = torch.argmax(logit, dim=1).reshape(-1)
                accuracy = sum(pred_label==class_label[mask_tmp])/len(pred_label)
                accuracies.append(accuracy)

            else:
                loss = F.mse_loss(out_dict[mask].flatten(), data['spot'].y[mask])/mask.sum()
            losses.append(float(loss))
        return losses, accuracies

    @torch.no_grad()
    def evaluate(self, model, data):
        class_label = torch.tensor(self.class_label).to(self.device) #num_spot
        mask = torch.tensor(self.mask).nonzero().reshape(-1).to(self.device) #num_cand
        split = torch.tensor(self.split).to(self.device) #num_cand
        model.eval()
        model.load_state_dict(self.best_model_state)
        x_dict, out_dict = model(data.x_dict, data.edge_index_dict)
        y_onehot_test = self.test_label_onehot
        y_score = F.softmax(out_dict['spot'][self.test_mask], dim=-1).cpu().numpy()
        pred_label = torch.argmax(torch.tensor(y_score).to(self.device),axis=1).reshape(-1)
        accuracy = sum(pred_label==class_label[self.test_mask])/len(self.test_mask)
        fpr, tpr, _= roc_curve(y_onehot_test.ravel(), y_score.ravel())
        auc_score = auc(fpr, tpr)
        config = self.config
        config_path = config['model']['model_type'] + '_' + str(config['data']['word']) + '_' +str(config['data']['category']) +'_' + str(config['data']['city'])+'_' + str(config['data']['prefecture'])+'.pkl'
        d = {'fpr': fpr, 'tpr': tpr}
        if auc_score> self.max_auc:
            self.max_auc = auc_score
        save_path = os.path.join('/home/yamanishi/project/trip_recommend/model/popularity/data/classification', config_path)
        with open(save_path, 'wb') as f:
            pickle.dump(d, f)
        return accuracy, auc_score



    def train_epoch(self, model, data, epoch_num):
        optimizer = optim.Adam(model.parameters(), lr=self.config['trainer']['lr'])
        for epoch in range(epoch_num):
            #model, loss = self.train(model, optimizer, data, epoch)
            model, loss = self.train_minibatch(model, optimizer, data, epoch)
            losses, accuracies = self.test(model,data)
            if self.early_count==5:
                test_accuracy, auc_score = self.evaluate(model, data)
                print('test_accuracy:', test_accuracy, 'auc_score', auc_score)
                break
            train_loss, valid_loss, test_loss = losses
            train_acc, valid_acc, test_acc = accuracies
            print(f'Epoch: {epoch+1:03d}, Loss: {loss:.5f}, Train: {train_loss:.5f}, '
                f'Val: {valid_loss:.5f}, Test: {test_loss:.5f}, Train acc: {train_acc:.5f}, '
                f'Valid acc: {valid_acc:.5f}, Test acc: {test_acc:.5f}')

if __name__=='__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    data = get_data(category=True, city=True, prefecture=True, multi=True)

    data.to(device)
    print(data)
    
    model = MyHetero(data.x_dict, data.edge_index_dict, num_layers=4, hidden_channels=128, out_channels=1, out_dim=512,multi=True)
    model.to(device)
    
    trainer = Trainer(device)
    trainer.train_epoch(model, data, epoch_num=1500)