import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

df_info = pd.read_csv('/home/yamanishi/project/trip_recommend/data/jalan/spot/experience_light.csv')
spot_index = {spot:i for i,spot in enumerate(df_info['spot_name'])}

class ItenaryDataset(Dataset):
    def __init__(self,):
        super().__init__()
        self.df_info = pd.read_csv('/home/yamanishi/project/trip_recommend/data/jalan/spot/experience_light.csv')
        self.spot_index = {spot:i for i,spot in enumerate(df_info['spot_name'])}
        self.index_spot = {i:spot for i,spot in enumerate(df_info['spot_name'])}
        self.spot_pref = {spot:pref for spot, pref in zip(df_info['spot_name'], df_info['prefecture'])}
        self.W = torch.load('/home/yamanishi/project/trip_recommend/data/spot_embedding.pt')
        self.embedding = torch.nn.Embedding(self.W.size(0), self.W.size(1))
        self.embedding.weight= torch.nn.Parameter(self.W)
        self.itenaries = np.load('/home/yamanishi/project/trip_recommend/data/jalan/itenary_all.npy', allow_pickle=True)
        self.itenary_ids = self.itenary2id()
        self.mode='train'

    def itenary2id(self):
        itenaries = []
        for itenary in self.itenaries:
            i = []
            for spot in itenary:
                if self.spot_index.get(spot):
                    i.append(spot_index[spot])
            if len(i)>1:
                itenaries.append(i)
        return itenaries

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, mode):
        self._mode = mode

    def __getitem__(self, index):
        itenary = torch.tensor(self.itenary_ids[index])
        if self.mode!='test':
            ratio = np.random.uniform(0.3, 0.8)
        else:
            ratio = 0.5
        itenary_size = int(len(itenary)*ratio)
        itenary_size=max(1, itenary_size)
        itenary_size=min(itenary_size, len(itenary)-1)
        #print(len(itenary), itenary_size)
        return self.embedding(itenary[:itenary_size]), self.embedding(itenary[itenary_size:]), self.spot_pref[self.index_spot[itenary[0].item()]]

    def __len__(self):
        return len(self.itenary_ids)

class ItenaryPredictDataset(ItenaryDataset):
    def __init__(self):
        super().__init__()
        self.masks = np.load('/home/yamanishi/project/trip_recommend/data/GRU_itenary/neighbors.npy')

    def __getitem__(self, index):
        itenary = torch.tensor(self.itenary_ids[index])
        if self.mode=='train':
            ratio = np.random.uniform(0.3, 0.8)
        else:
            ratio = 0.5
        itenary_size = int(len(itenary)*ratio)
        itenary_size=max(1, itenary_size)
        itenary_size=min(itenary_size, len(itenary)-1)
        #print(len(itenary), itenary_size)
        return self.embedding(itenary[:itenary_size]), self.embedding(itenary[itenary_size:]),itenary[:itenary_size], itenary[itenary_size:], self.spot_pref[self.index_spot[itenary[0].item()]]       
    
    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, mode):
        self._mode = mode

def collate_fn(batch):
    pre = [b[0] for b in batch]
    post = [b[1] for b in batch]
    pre_x = torch.nn.utils.rnn.pad_sequence(pre, batch_first=True)
    post_x = torch.nn.utils.rnn.pad_sequence(post, batch_first=True)
    return pre_x, post_x

def collate_fn_predict(batch):
    pre = [b[0] for b in batch]
    post = [b[1] for b in batch]
    pre_id = [b[2] for b in batch]
    post_id = [b[3] for b in batch]
    pref = [b[4] for b in batch]
    pre_x = torch.nn.utils.rnn.pad_sequence(pre, batch_first=True)
    post_x = torch.nn.utils.rnn.pad_sequence(post, batch_first=True)
    return pre_x, post_x, pre_id, post_id, pref

def create_batch_sampler(data, batch_size):
    indices = torch.arange(len(data)).tolist()
    sorted_indices = sorted(indices, key=lambda idx:(data[idx][2],len(data[idx][0])+len(data[idx][1])))

    batch_indices = []
    start = 0
    end = min(start + batch_size, len(data))
    while True:
        batch_indices.append(sorted_indices[start: end])

        if end >= len(data):
            break

        start = end
        end = min(start + batch_size, len(data))
    return batch_indices

def create_predict_batch_sampler(data, batch_size):
    indices = torch.arange(len(data)).tolist()
    sorted_indices = sorted(indices, key=lambda idx:(data[idx][4],len(data[idx][0])+len(data[idx][1])))

    batch_indices = []
    start = 0
    end = min(start + batch_size, len(data))
    while True:
        batch_indices.append(sorted_indices[start: end])

        if end >= len(data):
            break

        start = end
        end = min(start + batch_size, len(data))
    return batch_indices
    

def get_dataloaders(batch_size=32, ratio=0.7):
    dataset = ItenaryDataset()
    all_data_size = len(dataset)
    train_size = int(all_data_size*ratio)
    test_size = all_data_size - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_dataset.mode = 'train'
    test_dataset.mode = 'test'
    train_sampler = create_batch_sampler(train_dataset, batch_size)
    test_sampler = create_batch_sampler(test_dataset, batch_size)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_sampler=test_sampler,
        collate_fn=collate_fn
    )
    return train_loader, test_loader   

def get_predict_dataloaders(batch_size=32, ratio=0.7):
    dataset = ItenaryPredictDataset()
    all_data_size = len(dataset)
    train_size = int(all_data_size*ratio)
    test_size = all_data_size - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_dataset.mode = 'train'
    test_dataset.mode = 'test'
    train_sampler = create_predict_batch_sampler(train_dataset, batch_size)
    test_sampler = create_predict_batch_sampler(test_dataset, batch_size)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collate_fn_predict
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_sampler=test_sampler,
        collate_fn=collate_fn_predict
    )
    return train_loader, test_loader   

if __name__=='__main__':
    '''
    dataset = ItenaryDataset()
    all_data_size = len(dataset)
    train_size = int(all_data_size*0.7)
    test_size = all_data_size - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    it0, it1 = train_dataset[4]

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn
    )
    '''
    train_loader, test_loader = get_predict_dataloaders()
    for batch in train_loader:
        pre_x, post_x,pre_id, post_id, pref = batch
        print(len(pre_id), len(post_id))
    #for batch in test_loader:
    #    pre_x, post_x,pre_id, post_id, pref = batch
    #    print(len(pre_id), len(post_id))
    #for batch in train_loader:
    #    pre_x, post_x,pre_id, post_id, pref = batch
    #    print(batch)
        #print(len(pre_id))
        #print(len(post_id))
        #break

    


