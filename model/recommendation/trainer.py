from __future__ import print_function, division
import matplotlib.pyplot as plt
import time
import copy
import math
from typing import OrderedDict
import yaml
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
from conv.lgcn import LGCN

class Trainer:
    def __init__(self, config,):
        self.config = config
        self.device = config['device']
        self.neg_samples = np.load('./data/neg_samples.npy')
        self.df = pd.read_csv('/home/yamanishi/project/trip_recommend/data/jalan/spot/experience_light.csv')
        self.train_weights = np.load('/home/yamanishi/project/trip_recommend/model/popularity/data/train_weights.npy')
        self.spot_names = self.df['spot_name'].values
        self.max_recall = 0
        self.max_precision = 0
        self.neg_spots = np.load('./data/unappear_spots.npy')
        self.W = torch.nn.parameter.Parameter(torch.rand(128, 128))
        torch.nn.init.normal_(self.W, 0.1)
        self.W=self.W.to(self.device)
        print(self.W.device)
        print('loaded neg samples')
        wandb.init('recommendation', config=config)

    def sampling(self, data):
        user_num = len(data['user'].user_pos)
        negs=np.load('/home/yamanishi/project/trip_recommend/model/recommendation/data/unappear_spots.npy')
        n_users = data['user'].x.size(0)
        m_items = data['spot'].x.size(0)
        all = set(list(range(m_items)))-set(negs)
        allPos = data['user'].item_pos
        users = np.random.randint(0, n_users, user_num)
        users_list, pos_items, neg_items = [], [], []
        with open('./data/train_item.pkl', 'rb') as f:
            train_items = pickle.load(f)
        for i, user in enumerate(users):
            posForUser = np.array(train_items[user])
            if len(posForUser)==0:
                continue
            posindex = np.random.randint(0, len(posForUser))
            positem = posForUser[posindex]
            users_list.append(user)
            pos_items.append(positem)
            while True:
                negitem = np.random.randint(0, m_items)
                if (negitem not in posForUser) and (negitem in all):
                    neg_items.append(negitem)
                    break
        return torch.tensor(users_list), torch.tensor(pos_items), torch.tensor(neg_items)



    def train(self, model, optimizer, data, epoch):
        model.train()
        optimizer.zero_grad()
        #copy.deepcopy(model.state_dict())
        x_dict, out_dict = model(data.x_dict, data.edge_index_dict)
        mask = data['spot'].train_mask
        if isinstance(x_dict, dict):
            spot_mask = data['spot'].train_mask
            loss_spot = 0
            if self.config['trainer']['spot_pop_weight']>0:
                loss_spot = F.mse_loss(out_dict['spot'][spot_mask].flatten(), data['spot'].y[spot_mask])#/spot_mask.sum()
            loss_city=0
            if self.config['trainer']['city_pop_weight']>0 and self.config['data']['city']:
                city_mask = data['city'].train_mask
                loss_city = F.mse_loss(out_dict['city'][city_mask].flatten(), data['city'].y[city_mask])#/city_mask.sum()
            #pref_mask = data['pref'].train_mask
            #loss_pref = F.mse_loss(out_dict['pref'][pref_mask].flatten(), data['pref'].y[pref_mask])#/pref_mask.sum()

            if self.config['trainer']['sampling']=='lgcn':
                users, item_pos, item_neg=self.sampling(data)
            else:
                users = data['user'].user_pos
                item_pos = data['user'].item_pos
                item_neg = torch.from_numpy(self.neg_samples[epoch%1000])
            users = users.long()
            item_pos = item_pos.long()
            item_neg = item_neg.long()
            user_emb = x_dict['user'][users.to(x_dict['user'].device)]
            pos_emb = x_dict['spot'][item_pos.to(x_dict['spot'].device)]
            neg_emb = x_dict['spot'][item_neg.to(x_dict['spot'].device)]

            pos_scores = torch.sum(torch.mul(user_emb , pos_emb, ), dim=1)
            neg_scores = torch.sum(torch.mul(user_emb, neg_emb, ), dim=1)
            if self.config['trainer']['loss_word_weight']>0:
                spot_word = data['spot', 'relate', 'word'].edge_index
                word_emb = torch.mean(x_dict['word'][spot_word[1]].reshape(-1, 15, self.config['model']['hidden_channels']), dim=1)
                word_pos_emb = word_emb[item_pos]
                word_neg_emb = word_emb[item_neg]
                word_pos_scores = torch.sum(torch.mul(user_emb, word_pos_emb), dim=1)
                word_neg_scores = torch.sum(torch.mul(user_emb, word_neg_emb), dim=1)
                pos_scores += self.config['trainer']['loss_word_weight']*word_pos_scores
                neg_scores += self.config['trainer']['loss_word_weight']*word_neg_scores
            if self.config['trainer']['loss_city_weight']>0:
                spot_city = data['spot', 'belongs', 'city'].edge_index
                city_pos_index = spot_city[1][item_pos]
                city_neg_index = spot_city[1][item_neg]
                city_pos = x_dict['city'][city_pos_index]
                city_neg = x_dict['city'][city_neg_index]
                city_pos_scores = torch.sum(torch.mul(user_emb, city_pos), dim=1)
                city_neg_scores = torch.sum(torch.mul(user_emb, city_neg), dim=1)
                pos_scores += self.config['trainer']['loss_city_weight']*city_pos_scores
                neg_scores += self.config['trainer']['loss_city_weight']*city_neg_scores
            if self.config['trainer']['loss_category_weight']>0:
                spot_category = data['spot', 'has', 'category'].edge_index
                category_pos_index = spot_category[1][item_pos]
                category_neg_index = spot_category[1][item_neg]
                category_pos = x_dict['category'][category_pos_index]
                category_neg = x_dict['category'][category_neg_index]
                category_pos_scores = torch.sum(torch.mul(user_emb, category_pos), dim=1)
                category_neg_scores = torch.sum(torch.mul(user_emb, category_neg),dim=1)
                pos_scores += self.config['trainer']['loss_category_weight']*category_pos_scores
                neg_scores += self.config['trainer']['loss_category_weight']*category_neg_scores
            if self.config['trainer']['loss_pref_weight']>0:
                spot_pref = data['spot'].spot_pref
                pref_pos_index = spot_pref[1][item_pos]
                pref_neg_index = spot_pref[1][item_neg]
                pref_pos = x_dict['pref'][pref_pos_index]
                pref_neg = x_dict['pref'][pref_neg_index]
                pref_pos_scores = torch.sum(torch.mul(user_emb, pref_pos), dim=1)
                pref_neg_scores = torch.sum(torch.mul(user_emb, pref_neg),dim=1)
                pos_scores += self.config['trainer']['loss_pref_weight']*pref_pos_scores
                neg_scores += self.config['trainer']['loss_pref_weight']*pref_neg_scores
            loss_rec = torch.mean(torch.nn.functional.softplus(neg_scores-pos_scores))
            if self.config['trainer']['loss_weight']:
                loss_rec = loss_rec * torch.from_numpy(self.train_weights).to(loss_rec.device)
            loss_rec = torch.mean(loss_rec)
            reg_loss = (1/2)*(user_emb.norm(2).pow(2)+
                                pos_emb.norm(2).pow(2)+
                                neg_emb.norm(2).pow(2))

            weight_decay = 1e-4
            loss = self.config['trainer']['city_pop_weight']*loss_city \
                    +self.config['trainer']['spot_pop_weight']*loss_spot \
                    +loss_rec #+ weight_decay * reg_loss 
            if self.config['model']['model_type'] == 'lgcn':
                loss+=weight_decay * reg_loss

            #loss = loss_rec
            loss = loss.float()
        wandb.log({'loss':loss})
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
                loss_spot = 0
                if self.config['trainer']['spot_pop_weight']>0:
                    loss_spot = F.mse_loss(out_dict['spot'][mask].flatten(), data['spot'].y[city_mask])/mask.sum()
                loss_city=0
                if self.config['data']['city'] and self.config['trainer']['city_pop_weight']>0:
                    city_mask = data['city'][split]
                    loss_city = F.mse_loss(out_dict['city'][city_mask].flatten(), data['city'].y[city_mask])/mask.sum()
                #pref_mask = data['pref'][split]
                #loss_pref = F.mse_loss(out_dict['pref'][pref_mask].flatten(), data['pref'].y[pref_mask])/mask.sum()
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
        optimizer = optim.Adam(model.parameters(), lr=self.config['trainer']['lr'])
        for epoch in range(epoch_num):
            model, loss = self.train(model, optimizer, data, epoch)
            train_loss, val_loss, test_loss = self.test(model,data)
            print(f'Epoch: {epoch+1:03d}, Loss: {loss:.6f}, Train: {train_loss:.6f}, '
                f'Val: {val_loss:.6f}, Test: {test_loss:.6f}')

            if epoch%self.config['trainer']['explain_span']==0:
                prec, recall = self.evaluate_rec(model, data)
                if prec>self.max_precision:
                    self.max_precision = prec.cpu().item()
                    self.best_epoch = epoch
                if recall>self.max_recall:
                    self.max_recall = recall.cpu().item()
                print(f'Precision: {prec:.6f}, Recall: {recall:.6f}')

        prec, recall=self.evaluate_rec(model, data,)
        wandb.log({'prec': prec, 'recall':recall})
        print(f'Precision: {prec:.6f}, Recall: {recall:.6f}')

        cor = self.calc_cor(model, data)
        self.cor = cor
        print(f'cor: {cor:.4f}')

    @torch.no_grad()
    def evaluate_rec(self, model, data,):
        k = self.config['k']
        with open('./data/test_item.pkl', 'rb') as f:
            test_item = pickle.load(f)
        x_dict, out_dict = model(data.x_dict, data.edge_index_dict)
        rating = torch.matmul(x_dict['user'], x_dict['spot'].T)
        #rating = torch.nn.functional.sigmoid(rating)
        rating[data['user'].user_pos, data['user'].item_pos]=-1e9
        rating[:,torch.from_numpy(self.neg_spots).to(rating.device)]=-1e9
        _, rating_K = torch.topk(rating, k=k)
        print('rating_K',rating_K)
        recall, prec = 0, 0
        rs = []
        gts = []
        index = np.arange(0, 26000, 1000)
        for i, rk in enumerate(rating_K):
            gt = test_item[i]
            r = torch.isin(torch.tensor(gt).to(self.device), rk).sum()
            if len(gt)==0:continue
            
            prec+=r/k
            recall+=r/len(gt)
            if self.config['log']:
                if i in index:
                    train_spots = data['user'].item_pos[data['user'].user_pos==i]
                    with open('log_tmp.txt', 'a') as f:
                        f.write('train:')
                        for s in self.spot_names[train_spots.cpu().numpy()]:
                            f.write(s)
                            f.write(' ')
                        f.write('\n')
                        f.write('gt:')
                        for s in self.spot_names[gt]:
                            f.write(s)
                            f.write(' ')
                        f.write('\n')
                        f.write('pred:')
                        for s in self.spot_names[rk.cpu().numpy()]:
                            f.write(s)
                            f.write(' ')
                        f.write('\n')

                        print('train:', self.spot_names[train_spots.cpu().numpy()])
                        print('gt:', self.spot_names[gt])
                        print('pred:', self.spot_names[rk.cpu().numpy()])
            rs.append(r.item())
            gts.append(len(gt))
        prec/=len(rating_K)
        recall/=len(rating_K)
        if self.config['log']:
            with open('log_tmp.txt', 'a') as f:
                f.write(str(prec.cpu().numpy().item()))
                f.write(' ')
                f.write(str(recall.cpu().numpy().item()))
                f.write('\n')
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
        self.save_cor(gt_all_spot, pred_all_spot,'gt: log(review_count)','pred: log(review_count)','cor')
        return np.corrcoef(gt_all_spot, pred_all_spot)[0][1]

class LGCNTrainer:
    def __init__(self, data, config):
        self.lgcn = LGCN(data, config)
        self.data = data
        self.device = config['device']
        #self.data['spot'].x = torch.nn.Embedding(num_embeddings=len(data['spot'].x), embedding_dim=config['model']['hidden_channels'])
        #self.data['user'].x = torch.nn.Embedding(num_embeddings=len(data['user'].x), embedding_dim=config['model']['hidden_channels'])
        #torch.nn.init.normal_(self.data['user'].x.weight, std=0.1)
        #torch.nn.init.normal_(self.data['spot'].x.weight, std=0.1)
        print('data', self.data)
        self.lgcn = LGCN(self.data, config)
        self.config = config
        self.data.to(self.device)
        self.lgcn.to(self.device)
        self.optim = optim.Adam(self.lgcn.parameters(), lr=1e-3)
        self.neg_samples = np.load('./data/neg_samples.npy')
        self.neg_spots = np.load('./data/unappear_spots.npy')
        self.max_recall = 0
        self.max_precision = 0

    def train(self, epoch):
        user_out, spot_out = self.lgcn()
        user_pos = user_out[self.data['user'].user_pos]
        spot_pos = spot_out[self.data['user'].item_pos]
        spot_neg = spot_out[torch.from_numpy(self.neg_samples[epoch%1000])]
        pos_scores = torch.sum(torch.mul(user_pos, spot_pos), dim=1)
        neg_scores = torch.sum(torch.mul(user_pos, spot_neg), dim=1)
        loss = torch.mean(torch.nn.functional.softplus(neg_scores-pos_scores))
        print('lgcn user emb',self.lgcn.user_emb, 'lgcn spot emb',self.lgcn.spot_emb)
        time.sleep(5)
        #reg_loss = (1/2)* (self.lgcn.user_emb.norm(2).pow(2) + self.lgcn.spot_emb.norm(2).pow(2))/float(self.lgcn.user_emb.size(0))
        reg_loss = (1/2) * (user_pos.norm(2).pow(2)+
                            spot_pos.norm(2).pow(2)+
                            spot_neg.norm(2).pow(2))/float(self.lgcn.user_emb.size(0))
        weight_decay = 1e-4
        print(loss.item(), reg_loss.item())
        loss+=weight_decay*reg_loss
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return loss

    def train_epoch(self, epoch):
        for i in range(epoch):
            loss=self.train(epoch)
            print(f'epoch: {i}, loss is {loss:.4f}')
            if (i+1)%50==0:
                precision, recall = self.evaluate_rec()
                print(f'Precision is {precision:.4f}, Recall is {recall:.4f}')
                if precision > self.max_precision:
                    self.max_precision = precision
                if recall > self.max_recall:
                    self.max_recall =  recall
                time.sleep(1)


    @torch.no_grad()
    def evaluate_rec(self):
        k = self.config['k']
        with open('./data/test_item.pkl', 'rb') as f:
            test_item = pickle.load(f)

        user_out, spot_out = self.lgcn()
        rating = torch.matmul(user_out, spot_out.T)
        rating[data['user'].user_pos, data['user'].item_pos] = -1e9
        rating[:, torch.from_numpy(self.neg_spots).to(rating.device)] = -1e9
        _, rating_K = torch.topk(rating, k=k)
        print('rating_K', rating_K)
        recall, precision = 0, 0
        rs, gts = [], []
        for i, rk in enumerate(rating_K):
            gt = test_item[i]
            r = torch.isin(torch.tensor(gt).to(self.device), rk).sum()
            if len(gt)==0:continue

            precision+=r/k
            recall+=r/len(gt)
        precision/=len(rating_K)
        recall/=len(rating_K)
        return precision, recall


if __name__=='__main__':
    with open('./config.yaml') as f:
        config = yaml.safe_load(f)

    config['k'] = 20
    config['device'] = 'cuda:1'
    config['explain_num'] = 10
    config['epoch_num'] = 3000
    config['model']['model_type'] = 'lgcn'
    config['model']['num_layers'] = 3
    config['model']['hidden_channels'] = 128
    config['model']['concat'] = True
    config['model']['ReLU'] = True
    config['trainer']['explain_span'] = 50
    config['trainer']['lr'] = 0.0003
    config['trainer']['loss_city_weight'] = 0
    config['trainer']['loss_category_weight'] = 0
    config['trainer']['loss_word_weight'] = 0
    config['trainer']['loss_pref_weight'] = 0
    config['trainer']['city_pop_weight']=0
    config['trainer']['spot_pop_weight']=0
    config['data']['word'] = False
    config['data']['city'] = False
    config['data']['category'] = False
    config['data']['pref'] = False
    config['data']['init_std'] = 0.1
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    data = get_data(config)
    trainer = LGCNTrainer(data, config)
    trainer.train_epoch(3000)