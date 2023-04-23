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
                        'city_pop_weight','data_word' , 'data_category', 'data_city', 'data_pref' 'cor'])
        
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
                            self.config['data']['pref'],
                            round(self.trainer.cor, 5)])
    
    
if __name__=='__main__':
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    config['log']=False
    config['k'] = 20
    config['device'] = 'cuda:2'
    config['explain_num'] = 3
    config['epoch_num'] = 4000
    config['model']['model_type'] = 'ggnn'
    config['model']['num_layers'] = 4
    config['model']['hidden_channels'] = 320
    config['model']['concat'] = False
    config['model']['ReLU'] = True
    config['trainer']['lr'] = 0.0005
    config['trainer']['city_pop_weight']=0
    config['data']['word'] = True
    config['data']['city'] = True
    config['data']['category'] = True
    config['data']['pref'] = True

    exp = ExpRunner(config)
    with torch.autograd.set_detect_anomaly(True):
        exp.run_experiment()


    