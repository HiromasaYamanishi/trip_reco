import numpy as np
import torch
import torch.optim as optim
from gru import GRU
import pandas as pd
from dataloader import get_dataloaders, get_predict_dataloaders
import torch.nn.functional as F
from tqdm import tqdm
import wandb
import os
os.environ['CUDA_VISIVLE_DEVICES']='7,8,9'
torch.backends.cudnn.enabled=False
class Trainer:
    def __init__(self, device, batch_size):
        self.train_loader, self.test_loader = get_dataloaders(batch_size=batch_size)
        self.model = GRU()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-5)
        self.device = device
        print(device)

    def train(self, epoch):
        wandb.init()
        for i in range(epoch):
            loss=self.train_one()
            print(f'epoch: {i+1}/{epoch}, loss: {loss}')
            wandb.log({'epoch':i, 'loss':loss})

        torch.save(self.model.gru.state_dict(), '/home/yamanishi/project/trip_recommend/data/GRU_itenary/model.pth')
    
    def train_one(self):
        self.model.train()
        self.model.to(self.device)
        loss_all, count= 0,0
        for batch in tqdm(self.train_loader):
            pre_x = batch[0].to(self.device)
            post_x = batch[1].to(self.device)
            h_n_pre = self.model(pre_x).squeeze(0)
            h_n_post = self.model(post_x).squeeze(0)
            h_n_pre = h_n_pre/h_n_pre.norm(dim=1).unsqueeze(1)
            h_n_post = h_n_post/h_n_post.norm(dim=1).unsqueeze(1)
            logit = h_n_pre @ h_n_post.T
            temp = 0.3
            logit = torch.exp(logit/temp)
            label = torch.arange(len(logit)).to(self.device)
            loss_pre = F.cross_entropy(logit, label)
            loss_post = F.cross_entropy(logit.T, label)
            loss = loss_pre + loss_post
            loss_all+=loss
            count+=len(batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss_all/count

class PredictTrainer:
    def __init__(self, device, batch_size):
        self.model = GRU(is_fc=True)
        self.train_loader, self.test_loader = get_predict_dataloaders(batch_size=batch_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-5)
        self.device = device
        self.mask = torch.from_numpy(np.load('/home/yamanishi/project/trip_recommend/data/GRU_itenary/neighbors.npy'))
        self.df = pd.read_csv('/home/yamanishi/project/trip_recommend/data/jalan/spot/experience_light.csv')
        self.spot_size = len(self.df)
        self.W = torch.load('/home/yamanishi/project/trip_recommend/data/spot_embedding.pt')
        self.embedding = torch.nn.Embedding(self.W.size(0), self.W.size(1))
        self.embedding.weight= torch.nn.Parameter(self.W)

    def train(self, epoch):
        wandb.init('gru_itenary')
        #self.model.gru.load_state_dict(torch.load('/home/yamanishi/project/trip_recommend/data/GRU_itenary/gru.pth'))
        self.model.load_state_dict(torch.load('/home/yamanishi/project/trip_recommend/data/GRU_itenary/model_predict.pth'))
        print('loaded model')
        for i in range(epoch):
            loss=self.train_one()
            print(f'epoch:{i+1}/{epoch}, loss:{loss}')
            wandb.log({'epoch':epoch, 'loss':loss})
        torch.save(self.model.state_dict(), '/home/yamanishi/project/trip_recommend/data/GRU_itenary/model_predict.pth')

    def train_one(self):
        self.model.train()
        self.model.to(self.device)

        for batch in tqdm(self.train_loader):
            pre_embs, post_embs, pre_idxs, post_idxs, prefs = batch
            total_loss=0
            for i in range(len(batch)):
                pre_id = pre_idxs[-1]
                emb = pre_embs[i].unsqueeze(0)
                for id in post_idxs[i]:
                    emb = emb.to(self.device)
                    logit = self.model(emb).squeeze(0).squeeze(0)
                    mask = torch.tensor([True]*self.spot_size)
                    mask[self.mask[pre_id]]=False
                    mask.to(device)
                    logit[mask] = -1e9
                    self.optimizer.zero_grad()
                    loss = torch.nn.functional.cross_entropy(logit.unsqueeze(0), torch.tensor([id]).to(self.device))
                    if loss<1e8:
                        total_loss+=loss
                        loss.backward(retain_graph=True)
                        self.optimizer.step()

                    pre_id = id
                    emb = self.embedding(id).unsqueeze(0).unsqueeze(0).detach()
        return total_loss

    def test(self):
        self.model.load_state_dict(torch.load('/home/yamanishi/project/trip_recommend/data/GRU_itenary/model_predict.pth'))
        self.model.to(self.device)
        for batch in tqdm(self.test_loader):
            pre_embs, post_embs, pre_idxs, post_idxs, prefs = batch
            for i in range(post_idxs):
                continue
            
        
if __name__=='__main__':
    device='cuda'
    #trainer = Trainer(device, batch_size=64)
    #trainer.train(100)
    trainer = PredictTrainer(device, batch_size=64)
    trainer.train(100)
    #trainer.train(10)
        