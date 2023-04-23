from dataloader import get_data
from model_rec import Model
from trainer import Trainer 
from trainer_classification import Trainer
from model_classification import Model
import torch
import os
import csv
import yaml

class ExpRunner:
    def __init__(self, config):
        self.config = config
        self.data = get_data(config)
        print(self.data)
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
    import sys
    args = sys.argv
    with open('./config.yaml') as f:
        config = yaml.safe_load(f)
    gpu_num = args[1]
    device = f'cuda:{gpu_num}'
    config['log']=False
    config['k'] = 20
    config['device'] = device
    config['explain_num'] = 3
    config['epoch_num'] = 200
    config['model']['model_type'] = 'sage'
    config['model']['num_layers'] = 3
    config['model']['hidden_channels'] = 128
    config['model']['concat'] = False
    config['model']['ReLU'] = True
    config['trainer']['lr'] = 5e-4
    config['trainer']['city_pop_weight']=0
    config['data']['word'] = True
    config['data']['city'] = True
    config['data']['category'] = True
    config['data']['pref'] = True
    config['data']['spot'] = False
    config['data']['station'] = False

    exp = ExpRunner(config)
    exp.run_experiment()
    exit()

    for arc in ['ggnn', 'hgt', 'sage', 'gcn']:
        for layer in [1,2,3,4,5]:
            for dim in [32, 64, 128, 256, 512]:
                config['model']['model_type'] = arc
                config['model']['num_layers'] = layer
                config['model']['hidden_channels'] = dim
                exp = ExpRunner(config)
                exp.run_experiment()
    exit()
    config['model']['model_type'] = 'ggnn'
    config['model']['num_layers'] = 3
    config['model']['hidden_channels'] = 256
    config['data']['word'] = False
    config['data']['city'] = True
    config['data']['category'] = True
    config['data']['pref'] = False
    exp = ExpRunner(config)
    exp.run_experiment()
    exit()
    config['data']['word'] = True
    config['data']['city'] = False
    config['data']['category'] = True
    config['data']['pref'] = False
    exp = ExpRunner(config)
    exp.run_experiment()
    config['data']['word'] = True
    config['data']['city'] = True
    config['data']['category'] = True
    config['data']['pref'] = False
    exp = ExpRunner(config)
    exp.run_experiment()
    config['data']['word'] = True
    config['data']['city'] = True
    config['data']['category'] = True
    config['data']['pref'] = True
    exp = ExpRunner(config)
    exp.run_experiment()

    config['model']['model_type'] = 'han'
    config['model']['num_layers'] = 2
    config['model']['hidden_channels'] = 256
    config['data']['word'] = True
    config['data']['city'] = False
    config['data']['category'] = False
    config['data']['pref'] = False
    exp = ExpRunner(config)
    exp.run_experiment()
    config['data']['word'] = True
    config['data']['city'] = False
    config['data']['category'] = True
    config['data']['pref'] = False
    exp = ExpRunner(config)
    exp.run_experiment()
    config['data']['word'] = True
    config['data']['city'] = True
    config['data']['category'] = True
    config['data']['pref'] = False
    exp = ExpRunner(config)
    exp.run_experiment()
    config['data']['word'] = True
    config['data']['city'] = True
    config['data']['category'] = True
    config['data']['pref'] = True
    exp = ExpRunner(config)
    exp.run_experiment()

    config['model']['model_type'] = 'hgt'
    config['model']['num_layers'] = 3
    config['model']['hidden_channels'] = 256
    config['data']['word'] = True
    config['data']['city'] = False
    config['data']['category'] = False
    config['data']['pref'] = False
    exp = ExpRunner(config)
    exp.run_experiment()

    config['data']['word'] = True
    config['data']['city'] = False
    config['data']['category'] = True
    config['data']['pref'] = False
    exp = ExpRunner(config)
    exp.run_experiment()

    config['data']['word'] = True
    config['data']['city'] = True
    config['data']['category'] = True
    config['data']['pref'] = False
    exp = ExpRunner(config)
    exp.run_experiment()

    config['data']['word'] = True
    config['data']['city'] = True
    config['data']['category'] = True
    config['data']['pref'] = True
    exp = ExpRunner(config)
    exp.run_experiment()

    config['model']['model_type'] = 'sage'
    config['model']['num_layers'] = 3
    config['model']['hidden_channels'] = 256
    config['data']['word'] = True
    config['data']['city'] = False
    config['data']['category'] = False
    config['data']['pref'] = False
    exp = ExpRunner(config)
    exp.run_experiment()

    config['data']['word'] = True
    config['data']['city'] = False
    config['data']['category'] = True
    config['data']['pref'] = False
    exp = ExpRunner(config)
    exp.run_experiment()

    config['data']['word'] = True
    config['data']['city'] = True
    config['data']['category'] = True
    config['data']['pref'] = False
    exp = ExpRunner(config)
    exp.run_experiment()

    config['data']['word'] = True
    config['data']['city'] = True
    config['data']['category'] = True
    config['data']['pref'] = True
    exp = ExpRunner(config)
    exp.run_experiment()

    config['model']['model_type'] = 'gcn'
    config['model']['num_layers'] = 3
    config['model']['hidden_channels'] = 256
    config['data']['word'] = True
    config['data']['city'] = False
    config['data']['category'] = False
    config['data']['pref'] = False
    exp = ExpRunner(config)
    exp.run_experiment()

    config['data']['word'] = True
    config['data']['city'] = False
    config['data']['category'] = True
    config['data']['pref'] = False
    exp = ExpRunner(config)
    exp.run_experiment()

    config['data']['word'] = True
    config['data']['city'] = True
    config['data']['category'] = True
    config['data']['pref'] = False
    exp = ExpRunner(config)
    exp.run_experiment()

    config['data']['word'] = True
    config['data']['city'] = True
    config['data']['category'] = True
    config['data']['pref'] = True
    exp = ExpRunner(config)
    exp.run_experiment()

    config['model']['model_type'] = 'gcn'
    config['model']['num_layers'] = 2
    config['model']['hidden_channels'] = 256
    config['data']['word'] = True
    config['data']['city'] = False
    config['data']['category'] = False
    config['data']['pref'] = False
    exp = ExpRunner(config)
    exp.run_experiment()

    config['data']['word'] = True
    config['data']['city'] = False
    config['data']['category'] = True
    config['data']['pref'] = False
    exp = ExpRunner(config)
    exp.run_experiment()

    config['data']['word'] = True
    config['data']['city'] = True
    config['data']['category'] = True
    config['data']['pref'] = False
    exp = ExpRunner(config)
    exp.run_experiment()

    config['data']['word'] = True
    config['data']['city'] = True
    config['data']['category'] = True
    config['data']['pref'] = True
    exp = ExpRunner(config)
    exp.run_experiment()
    '''
    exp = ExpRunner(config)
    exp.run_experiment()
    for layer in [2,3,4]:
        for hidden_dim in [256, 384]:
            for arch in ['ggnn','han', 'hgt', 'sage']:
                config['model']['model_type'] = arch
                config['model']['hidden_channels'] = hidden_dim
                config['model']['num_layers'] = layer 
                exp = ExpRunner(config)
                try:
                    exp.run_experiment()
                except RuntimeError:
                    continue
    
    
    config['data']['word'] = True
    config['data']['city'] = False
    config['data']['category'] = False
    config['data']['pref'] = False
    exp = ExpRunner(config)
    exp.run_experiment()

    config['data']['word'] = True
    config['data']['city'] = False
    config['data']['category'] = False
    config['data']['pref'] = False
    exp = ExpRunner(config)
    exp.run_experiment()

    config['data']['word'] = False
    config['data']['city'] = False
    config['data']['category'] = True
    config['data']['pref'] = False
    exp = ExpRunner(config)
    exp.run_experiment()

    config['data']['word'] = False
    config['data']['city'] = True
    config['data']['category'] = False
    config['data']['pref'] = True
    exp = ExpRunner(config)
    exp.run_experiment()
    '''
    '''
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
    