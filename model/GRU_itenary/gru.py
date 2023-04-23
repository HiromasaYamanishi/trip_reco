import torch
from dataloader import get_dataloaders
import torch.nn.functional as F

class GRU(torch.nn.Module):
    def __init__(self, is_fc=False):
        super().__init__()
        self.W = torch.load('/home/yamanishi/project/trip_recommend/data/spot_embedding.pt')
        self.spot_size = self.W.size(0)
        self.emb_size = self.W.size(1)
        self.gru = torch.nn.GRU(input_size=self.emb_size, hidden_size=32, batch_first=True)
        self.is_fc=is_fc
        if is_fc==True:
            self.fc = torch.nn.Linear(32, self.spot_size)

    def forward(self, x):
        output,h_n = self.gru(x)
        if not self.is_fc:
            return h_n
        if self.is_fc:
            prob = self.fc(h_n)
            return prob
        

if __name__=='__main__':
    gru = GRU(is_fc = True)
    gru.gru.load_state_dict(torch.load('/home/yamanishi/project/trip_recommend/data/GRU_itenary/gru.pth'))
    #torch.save(gru.gru.state_dict(),'/home/yamanishi/project/trip_recommend/data/GRU_itenary/gru.pth')
    '''
    train_loader, test_loader = get_dataloaders()

    for batch in train_loader:'
        print(batch[0].size())
        print(batch[1].size())
        h_n_0 = gru(batch[0]).squeeze(0)
        h_n_1 = gru(batch[1]).squeeze(0)
        #print(h_n_0)
        #print(h_n_1)
        h_n_0 = h_n_0/h_n_0.norm(dim=1).unsqueeze(1)
        h_n_1 = h_n_1/h_n_1.norm(dim=1).unsqueeze(1)
        logit = h_n_0 @ h_n_1.T
        temp=0.3
        logit = torch.exp(logit/temp)
        label=torch.arange(len(logit))
        loss_pre = F.cross_entropy(logit, label)
        loss_post = F.cross_entropy(logit.T, label)
        print(loss_pre, loss_post)

        print(logit)
        break
    '''
    


