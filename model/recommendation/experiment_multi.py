from trainer import  MultiTrainer
import torch
import os
import csv
import numpy as np
import yaml

class ExpRunner:
    def __init__(self, config):
        self.config = config
        self.trainer = MultiTrainer(config)
        #if 'lgcn'==config['model']['model_type']:
        #    self.trainer = LGCNTrainer(self.data, config)
        self.device = config['device']

    def run_experiment(self):
        self.trainer.train_epoch(epoch_num=self.config['epoch_num'])

        isempty = os.stat('/home/yamanishi/project/trip_recommend/model/recommendation/result/result_multi.csv').st_size == 0
        with open('result/result_multi.csv', 'a') as f:
            writer = csv.writer(f)
            if isempty:
                writer.writerow(['recall', 'accuracy', 'best_epoch', 'batch_size', 'sampling',
                        'hidden_channels', 'num_layers', 'pre', 'mid', 'post', 'cat'])
        
            writer.writerow([round(self.trainer.max_recall.item(), 5),
                            round(self.trainer.max_precision.item(),5),
                            self.trainer.best_epoch,
                            self.config['batch_size'],
                            self.config['trainer']['sampling'],
                            self.config['model']['hidden_channels'],
                            self.config['model']['num_layers'],
                            self.config['model']['pre'],
                            self.config['model']['mid'],
                            self.config['model']['post'],
                            self.config['model']['cat']])
    
    
if __name__=='__main__':
    import sys
    args = sys.argv
    with open('./config.yaml') as f:
        config = yaml.safe_load(f)
    gpu_num = args[1]
    device = f'cuda:{gpu_num}'
    config['device'] = device
    config['epoch_num'] = 1000
    config['batch_size'] = 2048
    config['trainer']['sampling']= 'lgcn' #lgcn, mix, 
    config['model']['model_type'] = 'lightgat'
    config['model']['mix'] = None
    config['model']['hidden_channels'] = 128
    config['model']['num_layers'] = 3
    config['model']['pre'] = False
    config['model']['mid'] = False
    config['model']['post'] = False
    config['model']['cat'] = False

    exp = ExpRunner(config)
    exp.run_experiment()
    