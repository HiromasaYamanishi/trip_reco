from dataloader import get_data
from model_rec import Model
from trainer import Trainer 
import torch
import os
import csv
import yaml

class ExpRunner:
    def __init__(self, config):
        self.config = config
        self.data = get_data(config)
        self.model =  Model(self.data, config).model
        self.trainer = Trainer(config)
        self.device = config['device']
        self.data.to(self.device)
        self.model.to(self.device)
        print(config)

    def run_experiment(self):
        self.trainer.train_epoch(self.model, self.data, epoch_num=self.config['epoch_num'])

        isempty = os.stat('/home/yamanishi/project/trip_recommend/model/recommendation/result/result.csv').st_size == 0
        with open('result/result.csv', 'a') as f:
            writer = csv.writer(f)
            if isempty:
                writer.writerow(['recall', 'accuracy', 'best_epoch', 'model_type', 'num_layers','lr','hidden_channels', 'concat', 
                        'loss_city_weight', 'loss_pref_weight','loss_category_weight','loss_word_weight','city_pop_weight', 'spot_pop_weight',
                        'data_word' , 'data_category', 'data_city', 'data_pref' 'cor'])
        
            writer.writerow([round(self.trainer.max_recall, 5),
                            round(self.trainer.max_precision,5),
                            self.trainer.best_epoch,
                            self.config['model']['model_type'],
                            self.config['model']['num_layers'],
                            self.config['trainer']['lr'],
                            self.config['model']['hidden_channels'],
                            self.config['model']['concat'],
                            self.config['model']['ReLU'],
                            self.config['model']['pool'],
                            self.config['trainer']['loss_city_weight'],
                            self.config['trainer']['loss_pref_weight'],
                            self.config['trainer']['loss_category_weight'],
                            self.config['trainer']['loss_word_weight'],
                            self.config['trainer']['city_pop_weight'],
                            self.config['trainer']['spot_pop_weight'],
                            self.config['trainer']['sampling'],
                            self.config['trainer']['loss_weight'],
                            self.config['data']['word'],
                            self.config['data']['category'],
                            self.config['data']['city'],
                            self.config['data']['pref'],
                            self.config['data']['init_std'],
                            round(self.trainer.cor, 5)])
    
    
if __name__=='__main__':
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    config['log']=False
    config['k'] = 20
    config['device'] = 'cuda:2'
    config['explain_num'] = 3
    config['epoch_num'] = 2000
    config['model']['model_type'] = 'lgcn'
    config['model']['num_layers'] = 4
    config['model']['hidden_channels'] = 128
    config['model']['concat'] = False
    config['model']['ReLU'] = True
    config['model']['pool'] = 'mean'
    config['trainer']['explain_span'] = 50
    config['trainer']['lr'] = 0.0005
    config['trainer']['loss_city_weight'] = 0
    config['trainer']['loss_category_weight'] = 0
    config['trainer']['loss_word_weight'] = 0
    config['trainer']['loss_pref_weight'] = 0
    config['trainer']['city_pop_weight']=0
    config['trainer']['spot_pop_weight']=0.2
    config['trainer']['sampling'] = None
    config['trainer']['loss_weight'] = False
    config['data']['word'] = False
    config['data']['city'] = True
    config['data']['category'] = True
    config['data']['pref'] = True
    config['data']['init_std'] = 0.1
    
if __name__=='__main__':
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    config['log']=False
    config['k'] = 20
    config['device'] = 'cuda:1'
    config['explain_num'] = 5
    config['epoch_num'] = 2500
    config['model']['model_type'] = 'lgcn'
    config['model']['num_layers'] = 3
    config['model']['hidden_channels'] = 128
    config['model']['concat'] = False
    config['model']['ReLU'] = False
    config['model']['pool'] ='mean'
    config['trainer']['explain_span'] = 50
    config['trainer']['lr'] = 0.0005
    config['trainer']['loss_city_weight'] = 0
    config['trainer']['loss_category_weight'] = 0
    config['trainer']['loss_pref_weight'] = 0
    config['trainer']['loss_word_weight'] = 0
    config['trainer']['city_pop_weight']=0
    config['trainer']['spot_pop_weight']=0
    config['trainer']['sampling'] = None
    config['trainer']['loss_weight'] = False
    config['data']['word'] = False
    config['data']['city'] = False
    config['data']['category'] = False
    config['data']['pref'] = False
    config['data']['init_std'] = 0.1

    exp = ExpRunner(config)
    exp.run_experiment()
    '''
    for lc in [0, 0.1]:
        for la in [0, 0.1]:
            for lp in [0, 0.1]:
                for lw in [0, 0.1]:
                    for cp in [0, 0.1]:
                        for sp in [0, 0.1]:
                            config['trainer']['loss_city_weight'] = lc
                            config['trainer']['loss_category_weight'] = la
                            config['trainer']['loss_pref_weight'] = lp
                            config['trainer']['loss_word_weight'] = lw
                            config['trainer']['city_pop_weight']=cp
                            config['trainer']['spot_pop_weight']=sp
                            exp = ExpRunner(config)
                            try:
                                exp.run_experiment()   
                            except RuntimeError:
                                pass
                            except AttributeError:
                                pass
    '''
    '''
    for pool in ['mean', 'max', 'sum']:
        config['model']['pool'] = pool
        exp = ExpRunner(config)
        exp.run_experiment()
    '''
    '''
    config['model']['hidden_channels'] = 128
    for layer in [2,3,4,5]:
        for arch in ['hgt', 'han', 'sage', 'ggnn']:
            config['model']['num_layers']=layer
            config['model']['model_type']=arch
            exp = ExpRunner(config)
            try:
                exp.run_experiment()   
            except RuntimeError:
                pass          
    #tmux rec_data
    '''
    '''
    for word in [True, False]:
        for cat in [True, False]:
            for cit in [False]:
                for pre in [False]:
                    config['data']['word'] = word
                    config['data']['city'] = cit
                    config['data']['category'] = cat
                    config['data']['pref'] = pre           
                    if not (cit==False and pre==True):
                        exp = ExpRunner(config)
                        try:
                            exp.run_experiment()   
                        except RuntimeError:
                            pass
                        except AttributeError:
                            pass
    '''
    '''
    config['model']['hidden_channels'] = 128
    #tmux rec_relu
    for concat in [True, False]:
        for lr in [0.0005]:
            for relu in [False, True]:
                config['model']['concat']=concat
                config['trainer']['lr']=lr
                config['model']['ReLU']=relu
                exp = ExpRunner(config)
                try:
                    exp.run_experiment()
                except RuntimeError:
                    pass 
    '''
    '''
    #tmux rec_arch
    for arch in ['ggnn', 'ggnnv2', 'ggnnv3', 'ggnnv4']:
        config['model']['model_type']=arch
        exp = ExpRunner(config)
        try:
            exp.run_experiment()
        except RuntimeError:
            pass 
    '''

    