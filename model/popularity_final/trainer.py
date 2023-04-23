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
import numpy as np

import pandas as pd
import torch.nn.functional as F
 
import csv
import wandb
from model import DeepTour
import yaml
from conv.ggnn import HeteroGGNN
from conv.hgt import HGT
from conv.han import HAN
from conv.sage import HeteroSAGE
from conv.sageattn import HeteroSAGEAttention
from conv.ggnnv2 import HeteroGGNNV2
from conv.ggnnv3 import HeteroGGNNV3
from conv.ggnnv4 import HeteroGGNNV4
from sklearn.svm import SVR, SVC

class Trainer:
    def __init__(self, config,):
        self.config = config
        self.device = config['device']
        #self.neg_samples = np.load('./data/neg_samples.npy')
        self.df = pd.read_csv('/home/yamanishi/project/trip_recommend/data/jalan/spot/experience_light.csv')
        self.spot_names = self.df['spot_name'].values
        self.max_cor = 0
        self.max_last_cor = 0
        self.neg_spots = np.load('./data/unappear_spots.npy')
        self.W = torch.nn.parameter.Parameter(torch.rand(128, 128))
        torch.nn.init.normal_(self.W, 0.1)
        self.W=self.W.to(self.device)
        self.city_word = torch.load('/home/yamanishi/project/trip_recommend/data/jalan/graph/city_words.pt')
        self.pref_word = torch.load('/home/yamanishi/project/trip_recommend/data/jalan/graph/pref_words.pt')
        self.best_model = None
        #wandb.init('popularity', config=config)

    def train(self, model, optimizer, epoch):
        model.train()
        data = model.train_graph
        optimizer.zero_grad()
        #copy.deepcopy(model.state_dict())
        data = model.train_graph
        mask = data['spot'].train_mask
        loss = F.mse_loss(out[mask].flatten(), data['spot'].y[mask].to(out.device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return model,float(loss)
    

    def train_minibatch(self, model, optimizer, epoch):
        model.train()
        data = model.train_graph
        mask = data['spot'].train_mask.nonzero().reshape(-1)
        shuffle_indices = np.arange(len(mask))
        np.random.shuffle(shuffle_indices)
        mask = mask[shuffle_indices]
        batch_size = 8192
        total_loss = 0
        for i in range(0, len(mask), batch_size):
            mask_tmp = mask[i:min(i+batch_size, len(mask))]
            out, x_dict = model()
            loss = F.mse_loss(out[mask_tmp].flatten(), data['spot'].y[mask_tmp].to(out.device))
            if config['ssl']:
                city_word = self.city_word
                cities, city_words = city_word[0], city_word[1]
                lc = len(cities)
                index = torch.tensor(np.random.choice(np.arange(lc), int(lc*0.1), replace=False))
                city = cities[index]
                city_word_pos = city_words[index]
                city_word_neg = torch.randint(0, 68337, size=(len(city_word_pos),))
                city_pos_scores = torch.sum(torch.mul(x_dict['city'][city], x_dict['word'][city_word_pos]), dim=1)
                city_neg_scores = torch.sum(torch.mul(x_dict['city'][city], x_dict['word'][city_word_neg]), dim=1)
                city_loss = torch.mean(torch.nn.functional.softplus((city_neg_scores-city_pos_scores)))
                
                pref_word = self.pref_word
                prefs, pref_words = pref_word[0], pref_word[1]
                lp = len(prefs)
                index = torch.tensor(np.random.choice(np.arange(lp), int(lp*0.1), replace=False))
                pref = prefs[index]
                pref_word_pos = city_words[index]
                pref_word_neg = torch.randint(0, 68337, size=(len(pref_word_pos),))
                pref_pos_scores = torch.sum(torch.mul(x_dict['pref'][pref], x_dict['word'][pref_word_pos]), dim=1)
                pref_neg_scores = torch.sum(torch.mul(x_dict['pref'][pref], x_dict['word'][pref_word_neg]), dim=1)
                pref_loss = torch.mean(torch.nn.functional.softplus((pref_neg_scores-pref_pos_scores)))
                loss = loss+0.1*(city_loss+pref_loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss+=loss*batch_size
        total_loss/=len(mask)
        return model, total_loss

        

    @torch.no_grad()
    def test(self, model):
        data = model.train_graph
        model.eval()
        out, x_dict = model()
        alpha=1
        losses = []
        new_spot_y = torch.load('./new_spot/new_spot_y.pt').to(self.device)
        y = torch.cat([data['spot'].y.to(self.device), new_spot_y])
        for split in ['train_mask', 'valid_mask', 'test_mask']:
            all = 60
            train=0
            if split!='test_mask':
                mask = torch.cat([data['spot'][split] , torch.tensor([True]*train+[False]*(all-train))])
            else:
                mask = torch.cat([data['spot'][split] , torch.tensor([False]*train+[True]*(all-train))])
            loss = F.mse_loss(out[mask].flatten(), y[mask].to(out.device))

            losses.append(float(loss))
        return losses

    def train_epoch(self, model,epoch_num):
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=self.config['trainer']['lr'])
        for epoch in range(epoch_num):
            #model, loss = self.train_minibatch(model, optimizer, epoch)
            model, loss = self.train_minibatch(model, optimizer, epoch)
            train_loss, val_loss, test_loss = self.test(model)
            print(f'Epoch: {epoch+1:03d}, Loss: {loss:.6f}, Train: {train_loss:.6f}, '
                f'Val: {val_loss:.6f}, Test: {test_loss:.6f}')
            if epoch%1==0:
                #cor = self.calc_cor(model)
                cor, last_cor = self.calc_cor(model)
                print('cor is ', cor)
                print('last cor', last_cor)
                if last_cor>self.max_last_cor:
                    self.max_last_cor = last_cor
                    self.save_result_last(model)
                    torch.save(model.state_dict(), '/home/yamanishi/project/trip_recommend/model/popularity_final/data/deeptour.pth')
                if cor>self.max_cor:
                    self.max_cor = cor
                    self.best_epoch = epoch
                    #self.best_model = model.state_dict().copy()
                    #model.eval()
                    #out, x_dict = model()
                    #self.max_x = x_dict['spot']
                    #self.save_result(model)
                    #model.eval()
                    #out, x_dict = model()
                    #torch.save(x_dict['spot'],'/home/yamanishi/project/trip_recommend/model/popularity_final/data/spot_emb.pt')
                    #print('saved emb')
                    #model.eval()
                    #torch.save(model.state_dict(), '/home/yamanishi/project/trip_recommend/model/popularity_final/data/deeptour.pth')
                    #print('saved model')
                    #self.calc_cor_save(model, data)
        cor = self.calc_cor(model)
        self.last_cor = cor
        
        self.save_experiment(config)
        print(f'max cor:{self.max_cor:.4f}')
        print(f'final cor: {cor:.4f}')

    def save_experiment(self, config):
        isempty = os.stat('/home/yamanishi/project/trip_recommend/model/recommendation/result/result.csv').st_size == 0
        with open('result/result.csv', 'a') as f:
            writer = csv.writer(f)
            if isempty:
                writer.writerow(['cor', 'best_epoch', 'model_type', 'num_layers','hidden_channels', 
                        'word', 'category', 'city', 'station', 'pref', 'spot'])
        
            writer.writerow([round(self.max_cor, 5),
                            self.best_epoch,
                            config['model']['model_type'],
                            config['model']['num_layers'],
                            config['model']['hidden_channels'],
                            config['data']['word'],
                            config['data']['category'],
                            config['data']['city'],
                            config['data']['station'],
                            config['data']['prefecture'],
                            config['data']['spot'],
                            config['data']['neighbor_ratio'],
                            config['model']['conv'],])

    @torch.no_grad()
    def save_result_last(self, model):
        data = model.train_graph
        model.eval()
        out, x_dict = model()
        new_spot_y = torch.load('./new_spot/new_spot_y.pt').to(self.device)
        y = torch.cat([data['spot'].y.to(self.device), new_spot_y])
        for split in ['test_mask']:
            all = 60
            train=0
            mask = torch.cat([data['spot'][split] , torch.tensor([False]*train+[True]*(all-train))])
            #mask = data['spot'][split] + 
            #gt = data['spot'].y[mask].cpu().numpy()
            gt = y.cpu().numpy()[-60:]
            if isinstance(out, dict):
                pred = out['spot'][mask].reshape(-1).cpu().numpy()[-60:]
            else:
                pred = out[mask].reshape(-1).cpu().numpy()[-60:]
            cor = np.corrcoef(gt, pred)[0][1]

        plt.rcParams["font.size"] = 18
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1, aspect='equal')
        ax.set_xlim(0,4)
        ax.set_ylim(0,4)
        ax.set_xlabel('Ground Truth', fontsize=18)
        ax.set_ylabel('Predicted', fontsize=18)
        ax.set_title(f'cor: {round(cor,4)}', fontsize=18)
        ax.scatter(gt, pred)
        fig.subplots_adjust(bottom = 0.15)
        plt.savefig('/home/yamanishi/project/trip_recommend/model/popularity_final/data/scatter_last.png')
        #f = open('/home/yamanishi/project/trip_recommend/model/popularity_final/data/out_spot.txt', 'w')
        #for i, arg in enumerate(spot_names):
        #    if abs(gt[i]-pred[i])>1:
        #        ax.annotate(arg, (gt[i],pred[i]), fontname='Noto Serif CJK JP')
        #        f.write(arg)
        #        f.write(f' gt:{gt[i]}')
        #        f.write(f' pred:{pred[i]}')
        #        f.write('\n')
        #f.close()
        #plt.savefig('/home/yamanishi/project/trip_recommend/model/popularity_final/data/scatter_with_name.png')
        return cor

    @torch.no_grad()
    def save_result(self, model):
        data = model.train_graph
        model.eval()
        out, x_dict = model()
        for split in ['test_mask']:
            mask = data['spot'][split]
            gt = data['spot'].y[mask].cpu().numpy()
            if isinstance(out, dict):
                pred = out['spot'][mask].reshape(-1).cpu().numpy()
            else:
                pred = out[mask].reshape(-1).cpu().numpy()
            cor = np.corrcoef(gt, pred)[0][1]

        spot_names = self.df['spot_name'].values[data['spot'].test_mask.cpu().numpy()]
        df = pd.DataFrame({'spot_names':spot_names, 'gt':gt, 'pred': pred})
        df.to_csv('/home/yamanishi/project/trip_recommend/model/popularity_final/data/test_result.csv')

        plt.rcParams["font.size"] = 18
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1, aspect='equal')
        ax.set_xlim(0,4)
        ax.set_ylim(0,4)
        ax.set_xlabel('Ground Truth', fontsize=18)
        ax.set_ylabel('Predicted', fontsize=18)
        ax.set_title(f'cor: {round(cor,4)}', fontsize=18)
        ax.scatter(gt, pred)
        fig.subplots_adjust(bottom = 0.15)
        plt.savefig('/home/yamanishi/project/trip_recommend/model/popularity_final/data/scatter.png')
        #f = open('/home/yamanishi/project/trip_recommend/model/popularity_final/data/out_spot.txt', 'w')
        #for i, arg in enumerate(spot_names):
        #    if abs(gt[i]-pred[i])>1:
        #        ax.annotate(arg, (gt[i],pred[i]), fontname='Noto Serif CJK JP')
        #        f.write(arg)
        #        f.write(f' gt:{gt[i]}')
        #        f.write(f' pred:{pred[i]}')
        #        f.write('\n')
        #f.close()
        #plt.savefig('/home/yamanishi/project/trip_recommend/model/popularity_final/data/scatter_with_name.png')
        return cor


    @torch.no_grad()
    def calc_cor(self, model):
        data = model.train_graph
        model.eval()
        out, x_dict = model()
        new_spot_y = torch.load('./new_spot/new_spot_y.pt').to(self.device)
        y = torch.cat([data['spot'].y.to(self.device), new_spot_y])
        for split in ['test_mask']:
            all = 60
            train=0
            mask = torch.cat([data['spot'][split], torch.tensor([False]*train+[True]*(all-train))]) 
            gt = y[mask].cpu().numpy()
            if isinstance(out, dict):
                pred = out['spot'][mask].reshape(-1).cpu().numpy()
            else:
                pred = out[mask].reshape(-1).cpu().numpy()
            cor = np.corrcoef(gt, pred)[0][1]
            last_cor = np.corrcoef(gt[-(all-train):], pred[-(all-train):])[0][1]
            print(gt[-all:])
            print(pred[-all:])
        return cor, last_cor


    @torch.no_grad()
    def calc_cor_save(self, model, data):
        print('saving cor')
        model.eval()
        gt_all_spot, pred_all_spot=[], []
        gt_all_city, pred_all_city = [], []
        x_dict, out_dict = model(data.x_dict, data.edge_index_dict)
        path = Path()
        #df = pd.read_csv(path.df_experience_path)
        #df['pred'] = pd.Series(out['spot'].flatten().cpu().detach().numpy())
        #df.to_csv(path.df_experience_path)

        for split in ['test_mask']:
            spot_mask = data['spot'][split]
            gt_spot = data['spot'].y[spot_mask]
            #city_mask = data['city'][split]
            #gt_city = data['city'].y[city_mask]

            if isinstance(out_dict, dict):
                pred_spot = out_dict['spot'][spot_mask].flatten()
                #pred_city = out_dict['city'][city_mask].flatten()
            else:
                mask = data['spot'][split]
                pred = out[mask]
            gt_all_spot.append(gt_spot.cpu().detach().numpy().copy())
            pred_all_spot.append(pred_spot.cpu().detach().numpy().copy())
            #gt_all_city.append(gt_city.cpu().detach().numpy().copy())
            #pred_all_city.append(pred_city.cpu().detach().numpy().copy())

        gt_all_spot = np.concatenate(gt_all_spot)
        pred_all_spot = np.concatenate(pred_all_spot).reshape(-1)
        spot_names = self.df['spot_name'].values[data['spot'].test_mask.cpu().numpy()]
        df = pd.DataFrame({'spot_names':spot_names, 'gt':gt_all_spot, 'pred': pred_all_spot})
        df.to_csv('/home/yamanishi/project/trip_recommend/model/popularity/data/test_result.csv')
        #gt_all_city = np.concatenate(gt_all_city)
        #pred_all_city = np.concatenate(pred_all_city).reshape(-1)
        self.save_cor(gt_all_spot, pred_all_spot,'gt: log(review_count)','pred: log(review_count)','cor')
        return np.corrcoef(gt_all_spot, pred_all_spot)[0][1], None#np.corrcoef(gt_all_city, pred_all_city)[0][1]

    def save_cor(self, x, y, x_name, y_name,save_name,*args):
        plt.rcParams["font.size"] = 18
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1, aspect='equal')
        ax.set_xlim(0,4)
        ax.set_ylim(0,4)
        ax.set_xlabel(x_name, fontsize=18)
        ax.set_ylabel(y_name, fontsize=18)
        cor=np.corrcoef(x, y)[0][1]
        ax.set_title(f'cor: {round(cor,5)}', fontsize=20)
        ax.scatter(x, y)
        fig.subplots_adjust(bottom = 0.15)
        plt.savefig(f'{save_name}.png')
        if len(args)>0:
            f = open('out_spot.txt', 'w')
            for i, arg in enumerate(args[0]):
                if abs(x[i]-y[i])>1.5:
                    ax.annotate(arg, (x[i],y[i]), fontname='Noto Serif CJK JP')
                    f.write(arg)
                    f.write(f' gt:{x[i]}')
                    f.write(f' pred:{y[i]}')
                    f.write('\n')
            f.close()
            plt.savefig('cor_with_name.png')

if __name__=='__main__':
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    import sys
    args = sys.argv
    gpu_num = args[1]
    config['device'] = f'cuda:{gpu_num}'
    config['ssl'] = False
    config['data']['word'] = True
    config['data']['category'] = False
    config['data']['city'] = True
    config['data']['prefecture'] = False
    config['data']['station'] = False
    config['data']['spot'] = False
    config['data']['neighbor_ratio']=0
    config['model']['model_type'] = 'deeptour'
    config['model']['ReLU'] = True
    config['model']['num_layers'] = 3
    config['model']['tpgnn_layers'] = 2
    config['model']['spgnn_layers'] = 2
    config['model']['hidden_channels'] = 256
    config['trainer']['lr'] = 5e-4
    config['data']['spot'] = False
    print(config)
    gnn = DeepTour(config)
    trainer = Trainer(config)
    trainer.train_epoch(gnn, epoch_num=70)
    #for i in range(3):
    #    trainer.train_epoch(gnn, epoch_num=100)
    exit()
    for layer in [1,2,3,4,5,6]:
        for dim in [16,32,64,128,256,512, 1024]:
            for i in range(3):
                config['model']['hidden_channels']=dim
                config['model']['num_layers']=layer
                gnn = DeepTour(config)
                trainer.train_epoch(gnn, epoch_num=100)
    exit()
    ''''
    for i in range(3):
        for dist in [50, 100, 200, 500, 1000, 2000, 5000]:
            config['data']['spot'] = dist
            trainer = Trainer(config)
            gnn = DeepTour(config)
            print(gnn)
            trainer.train_epoch(gnn, epoch_num=100)  
    exit()
    '''
    import sys
    args = sys.argv
    gpu_num = args[1]
    config['device'] = f'cuda:{gpu_num}'
    for i in range(3):
        trainer = Trainer(config)
        gnn = DeepTour(config)
        print(gnn)
        trainer.train_epoch(gnn, epoch_num=100)  
    exit()   
    '''
    print(config)
    for i in range(3):
        trainer = Trainer(config)
        gnn = DeepTour(config)
        print(gnn)
        trainer.train_epoch(gnn, epoch_num=100)
    exit()
    '''
    for i in range(3):
        for dist in [50, 100, 200, 500, 1000, 2000, 5000]:
            config['data']['spot'] = dist
            trainer = Trainer(config)
            gnn = DeepTour(config)
            print(gnn)
            trainer.train_epoch(gnn, epoch_num=100)          
    exit()
    
    for w in [True, False]:
        for ca in [True, False]:
            for ci in [True, False]:
                for pr in [True, False]:
                    for st in [True, False]:
                        if (st==True and ci==False) or (ci==False and pr==True):continue
                        for i in range(3):
                            config['data']['word'] = w
                            config['data']['category'] = ca
                            config['data']['city'] = ci
                            config['data']['prefecture'] = pr
                            config['data']['station'] = st
                            trainer = Trainer(config)
                            gnn = DeepTour(config)
                            print(gnn)
                            trainer.train_epoch(gnn, epoch_num=100)
                            print(config)
    exit()
    trainer = Trainer(config)
    gnn = DeepTour(config)
    print(gnn)
    trainer.train_epoch(gnn, epoch_num=35)
    exit()
    #gnn = DeepTour(config)
    #print(gnn)
    for din in [768, 1024]:
        config['model']['hidden_channels'] = din
        trainer = Trainer(config)
        gnn = DeepTour(config)
        print(gnn)
        trainer.train_epoch(gnn, epoch_num=50)
    exit()
    '''
    for num_layer in range():
        config['model']['num_layers'] = num_layer
        trainer = Trainer(config)
        trainer.train_epoch(gnn, epoch_num=100)
    exit()
    '''
    #trainer = Trainer(config)
    #trainer.train_epoch(gnn, epoch_num=50)
    #exit()
    '''
    model_dict = {'sage': HeteroSAGE,
                'sageattn': HeteroSAGEAttention,
                'han': HAN,
                'hgt': HGT,
                'ggnn': HeteroGGNN,
                'ggnnv2': HeteroGGNNV2,
                'ggnnv3': HeteroGGNNV3,
                'ggnnv4': HeteroGGNNV4}
    '''
    for num_layer in range(1,6):
        config['model']['num_layers'] = num_layer
        trainer = Trainer(config)
        trainer.train_epoch(gnn, epoch_num=5)
    #trainer = Trainer(config)
    #trainer.train_epoch(gnn, epoch_num=5)