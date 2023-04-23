import torch
import torch.nn as nn
from torch_geometric.nn.conv import GCNConv
from sklearn.metrics.pairwise
from torch_geometric.nn import Linear, to_hetero, Sequential, GNNExplainer
from torch_geometric.nn.dense.linear import Linear
import torch.nn.functional as F


class MRFasGCN(torch.nn.Module):
    def __init__(self, X, S, clsss_num, hidden_dims=32, beta=0.5):
        super().__init__()
        self.num_nodes = X.size()[-0]
        self.attribute_num = X.size()[1]
        self.class_num = class_num
        self.hidden_dims = hidden_dims
        self.X = X
        self.A = A
        self.d = torch.sum(A, axis=0)
        self.e = A.sum()

        self.conv1 = GCNConv(in_channels=self.attribue_num, out_channels=hidden_dims)
        self.conv2 = GCNConv(in_channels=hidden_dims, out_channels=class_num)

        self.h = torch.nn.Parameter(torch.zeros(class_num, class_num))
        nn.init.xavier_uniform_(self.G)

        self.guzai = torch.matmul(self.deg.reshape(-1,1), self.def.reshape(1,-1))/(2*self.e) - A
        self.zeta = torch.tensor(cosine_similarity(X.cpu().numpy(), X.cpu().numpy()))
        self.R_zeta = self.zeta/torch.sum(self.zeta, dim=1)

        self.beta = beta
        self.tau = self.beta*self.guzai + (1-beta) * self.R_zeta

        self.H = Linear(class_num, class_num, False, weight_initializer='glorot')


    def MR(self, x, A):
        x_ = torch.matmul(self.tau, x)
        x_=self.H(x_)
        return F.softmax(x-x_)

    def forward(self, x):
        x = self.conv1(x)
        x.relu()
        x = self.conv2(x)
        x = self.MR(x)
        return x

class Decoder(torch.nn.Module):
    def __


class USLGCNCommunity(torch.nn.Module):
    def __init__(self, X, A, clsss_num, hidden_dims,beta=0.5):
        self.encoder = MRFasGCN(X, A, clsss_num, hidden_dims, beta=0.5)

    
        
index = torch.meshgrid(c, c, indexing='ij')
Y = X[index]