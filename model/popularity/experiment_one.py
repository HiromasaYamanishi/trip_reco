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
                writer.writerow(['cor', 'best_epoch', 'model_type', 'num_layers','lr','hidden_channels', 'concat', 
                        'city_pop_weight','data_word' , 'data_category', 'data_city', 'data_pref'])
        
            writer.writerow([round(self.trainer.max_cor, 5),
                            self.trainer.best_epoch,
                            self.config['model']['model_type'],
                            self.config['model']['num_layers'],
                            self.config['trainer']['lr'],
                            self.config['model']['hidden_channels'],
                            self.config['model']['concat'],
                            self.config['trainer']['city_pop_weight'],
                            self.config['data']['word'],
                            self.config['data']['category'],
                            self.config['data']['city'],
                            self.config['data']['pref'],])

    
    
if __name__=='__main__':
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    config['log']=False
    config['k'] = 20
    config['device'] = 'cuda:0'
    config['explain_num'] = 3
    config['epoch_num'] = 700
    config['model']['model_type'] = 'ggnnv4'
    config['model']['num_layers'] = 4
    config['model']['hidden_channels'] = 384
    config['model']['concat'] = False
    config['model']['ReLU'] = True
    config['trainer']['lr'] = 1e-4
    config['trainer']['city_pop_weight']=0
    config['data']['word'] = True
    config['data']['city'] = True
    config['data']['category'] = True
    config['data']['pref'] = True

    for layer in [3,4]:
        for hidden_dim in [256, 384]:
            for arch in ['ggnn', 'ggnnv4','han', 'hgt', 'sage']:
                config['model']['model_type'] = arch
                config['model']['hidden_channels'] = hidden_dim
                config['model']['num_layers'] = layer 
                exp = ExpRunner(config)
                try:
                    exp.run_experiment()
                except RuntimeError:
                    continue
    '''
    for dw in [True, False]:
        for dc in [True, False]:
            for dca in [True, False]:
                for dp in [True, False]:
                    if dc==False and dp==True:continue
                    config['data']['word'] = dw
                    config['data']['city'] = dc
                    config['data']['category'] = dca
                    config['data']['pref'] = dp
                    exp = ExpRunner(config)
                    exp.run_experiment()
    '''
    '''
    for layer in [2,3,4,5]:
        for hidden_dim in [128, 256, 384]:
            for arch in ['ggnnv2', 'ggnnv3', 'ggnnv4']:
                config['model']['model_type'] = arch
                config['model']['hidden_channels'] = hidden_dim
                config['model']['num_layers'] = layer 
                exp = ExpRunner(config)
                try:
                    exp.run_experiment()
                except RuntimeError:
                    continue
    '''
    