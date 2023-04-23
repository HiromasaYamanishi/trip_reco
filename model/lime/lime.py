import sys
sys.path.append('..')
from collect_data.preprocessing.preprocess_refactor import Path
from graph_test import get_data
from my_hetero import MyHetero
import numpy as np
import random
import itertools
import torch
from sklearn.linear_model import Lasso
import csv
from tqdm import tqdm

class LIME:
    def __init__(self, data, node_idx, target_edge_type, num_sample=1024, sample=False):
        self.node_idx = node_idx
        self.target_edge_type = target_edge_type
        self.rev_target_edge_type = ('spot', 'relate', 'word')
        self.data = data
        self.target_edge = self.data.edge_index_dict[target_edge_type]
        self.rev_target_edge = self.data.edge_index_dict[self.rev_target_edge_type]
        self.num_sample = num_sample
        self.sample=False
        self.sigma=10
        self.alpha=1e-2
        

    def get_onehop_neighor(self, node_idx):
        return self.target_edge[0][self.target_edge[1]==node_idx]

    def get_onehop_edges(self, node_idx):
        perturb_edge = self.target_edge[:, self.target_edge[1]==node_idx]
        not_perturb_edge = self.target_edge[:, self.target_edge[1]!=node_idx]

        perturb_rev_edge = self.rev_target_edge[:, self.rev_target_edge[0]==node_idx]
        not_perturb_rev_edge = self.rev_target_edge[:, self.rev_target_edge[0]!=node_idx]

        return perturb_edge, not_perturb_edge, perturb_rev_edge, not_perturb_rev_edge


    def explain_node(self, model, data, node_idx, device):
        perturb_edge, not_perturb_edge, perturb_rev_edge, not_perturb_rev_edge = self.get_onehop_edges(node_idx)

        vocab_all = np.load('/home/yamanishi/project/trip_recommend/data/graph/tfidf_words.npy')
        vocab_tmp = vocab_all[perturb_edge[0].cpu().numpy()]
        spot_names = np.load('/home/yamanishi/project/trip_recommend/data/graph/spot_names.npy', allow_pickle=True)
        spot_name = spot_names[node_idx] 
        print(spot_name, vocab_tmp)

        b = [True, False]
        bool_all = np.array(list(itertools.product(b, repeat=perturb_edge.size()[1])))

        if self.sample:
            bool_sample = np.random.sample(bool_all, self.num_sample)
        else:
            bool_sample = bool_all

        model.eval()
        X, y, weights = [],[],[]


        n = perturb_edge.size()[1]
    
        for bool_ in bool_sample:
            l1_norm = sum(bool_.astype(int))
            weight = np.exp(-(n-l1_norm)**2/(self.sigma**2))
            edge_index = torch.cat([torch.tensor(perturb_edge[:, bool_]), torch.tensor(not_perturb_edge)], axis=1).to(device)
            rev_edge_index = torch.cat([torch.tensor(perturb_rev_edge[:,bool_]), torch.tensor(not_perturb_rev_edge)], axis=1).to(device)
            data['word','revrelate','spot'].edge_index = edge_index
            data['spot','relate','word'].edge_index = rev_edge_index

            out = model(data.x_dict, data.edge_index_dict).cpu().detach().numpy()
            target_out = out[node_idx]

            X.append(list(bool_.astype(int)))
            y.append(target_out.item())
            weights.append(weight)

        X = np.array(X)
        y = np.array(y)
        weights = np.array(weights)
        X = X*(weights.reshape(-1,1))
        y = y*weights

        lasso_model = Lasso(alpha = self.alpha)
        lasso_model.fit(X, y)
        coef = lasso_model.coef_
        return spot_name, vocab_tmp, coef

    def explain_all(self, model, data, device):
        fieldnames=['spot_name','gt','pred']
        for i in range(10):
            fieldnames.append(f'vocab{i}')
            fieldnames.append(f'coef{i}')

        with open('/home/yamanishi/project/trip_recommend/data/lime/lime_explain.csv','w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            writer.writeheader()
            spot_names = np.load('/home/yamanishi/project/trip_recommend/data/graph/spot_names.npy', allow_pickle=True)
            ground_truth_all = np.load('/home/yamanishi/project/trip_recommend/data/graph/review_counts.npy')
            model.eval()
            predicted_all = model(data.x_dict, data.edge_index_dict).cpu().detach().numpy()


            limit=len(spot_names)
            for node_idx in tqdm(range(limit)):
                spot_name, vocab, coef = self.explain_node(model, data, node_idx=node_idx, device=device)
                
                d = {'spot_name':spot_name, 'gt':ground_truth_all[node_idx], 'pred':predicted_all[node_idx][0]}
                for i in range(10):
                    d[f'vocab{i}'] = vocab[i]
                    d[f'coef{i}'] = coef[i]

                writer.writerow(d)


if __name__=='__main__':
    data = get_data(multi=True)
    model = MyHetero(data.x_dict, data.edge_index_dict, hidden_channels=128, out_channels=1, out_dim=512,multi=True)
    
    model.load_state_dict(torch.load('../data/model/model.pth'))
    device = 'cuda:5'
    test_idx = torch.where(data['spot'].test_mask==True)
    node_idx = random.sample(list(test_idx[0].numpy()), 1)[0]
    print(node_idx)
    data.to(device)
    model.to(device)
    lime = LIME(node_idx=node_idx, data=data, target_edge_type=('word', 'revrelate', 'spot'), sample=False)
    lime.explain_all(model, data, device=device)
    #print(spot_name, vocab, coef)

    #print(y)
    #lime.explain_node(model, data, explain_node_id)