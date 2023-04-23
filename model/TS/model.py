import torch
from typing import Dict, List, Tuple
import torch.nn as nn
from torch.nn.parameter import Parameter
import utils as u
import math
import torch.nn.functional as F
from torch.optim import Adam

class GCNConv(torch.nn.Module):
    def __init__(self, A, args:dict):
        super().__init__()
        self.A = A
        self.args = args
        self.emb_dim = args.emb_dim
        if isinstance(A, torch.sparse.FloatTensor):
            dense = A.to_dense()
            self.n = dense.size()[0]
            D = torch.sum(dense, dim=1).float()
            D[D==0.]=1
            self.D = D
            D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
            dense = dense/D_sqrt
            dense = dense/D_sqrt.t()
            index = dense.nonzero()
            data = dense[dense>1e-9]
            self.G = torch.sparse.FloatTensor(index.t(), data, torch.Size([self.n, self.n]))
        else:
            self.A = self.A.float()
            D = torch.sum(A, dim=1).float()
            D[D==0.]==1
            self.D = D
            D_sqrt = torch.sqrt(D)
            D_sqrt = torch.diag(D_sqrt)
            self.D_sqrt = D_sqrt
        self.W = torch.nn.Parameter(torch.Tensor(self.emb_dim, self.emb_dim))
        self.reset_param(self.W)

    def reset_param(self, t):
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv, stdv)

    def forward(self, X):
        if isinstance(self.A, torch.sparse.FloatTensor):
            X = torch.sparse.mm(self.G, X)
            X = torch.sparse.mm(X, self.W)
        else:
            X = self.D_sqrt @ self.A @ self.D_sqrt @ X @ self.W
        return X

class GCN(torch.nn.Module):
    def __init__(self, A: torch.sparse.IntTensor, args:dict):
        super().__init__()
        self.conv_num = args.conv_num
        self.convs = torch.nn.ModuleList([])
        for i in range(self.conv_num):
            self.convs.append(GCNConv(A, args))

    def forward(self, X):
        for i in range(self.conv_num):
            X = self.convs[i](X).relu()
        Z = X
        return Z 

class mat_GRU_gate(torch.nn.Module):
    def __init__(self, rows:int, cols:int,activation:torch.nn):
        super().__init__()
        self.activation = activation
        self.W = Parameter(torch.Tensor(rows, rows))
        self.reset_param(self.W)

        self.U = Parameter(torch.Tensor(rows, rows))
        self.reset_param(self.U)

        self.bias = Parameter(torch.zeros(rows, cols))

    def reset_param(self, t):
        stdv = 1./math.sqrt(t.size(1))
        t.data.uniform_(-stdv, stdv)

    def forward(self, x, hidden):
        #x: (rows, cols)
        out = self.activation(self.W.matmul(x) + self.U.matmul(hidden) + self.bias)
        #out: (rows, cols)
        return out

class mat_GRU_cell(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        rows, cols = args.n, args.emb_dim
        self.update = mat_GRU_gate(rows, cols, torch.nn.Sigmoid())
        self.reset = mat_GRU_gate(rows, cols, torch.nn.Sigmoid())
        self.htilda = mat_GRU_gate(rows, cols, torch.nn.Tanh())
        self.choose_topk = TopK(rows, cols)

    def forward(self, prev_H, Z):
        #prev_Q: (rows, cols) prev_Z: (rows, cols)
        #z_topk = self.chooose_topk(prev_Z, mask)
        update = self.update(Z, prev_H) #(rows, cols)
        reset = self.reset(Z, prev_H) #(rows, cols)

        h_cap = reset*prev_H #(rows, cols)
        h_cap = self.htilda(Z, h_cap) #(rows, cols)
        new_H = (1-update) * prev_H + update*h_cap

        return new_H

class GRU(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.gru_num = args.gru_num
        self.grus = torch.nn.ModuleList([])
        for i in range(self.gru_num):
            self.grus.append(mat_GRU_cell(args))
        print(self.grus)

    def forward(self, H, X):
        H_out = []
        for i in range(self.gru_num):
            H_ = X
            H_ = self.grus[i](H[i], H_)
            H_out.append(H_)
        H_out = torch.stack(H_out)
        return H_out

class TopK(torch.nn.Module):
    def __init__(self, feats, k):
        super().__init__()
        self.scorer = Parameter(torch.Tensor(feats, 1))
        self.reset_param(self.scorer)

        self.k = k
    
    def reset_param(self, t):
        stdv = 1./math.sqrt(t.size(0))
        t.data.uniform_(-stdv, stdv)

    def forward(self, node_embs, mask):
        scores = node_embs.matmul(self.scorer)/self.scorer.sum()
        scores = scores + mask

        vals, topk_indices = scores.view(-1).topk(self.k)
        topk_indices = topk_indices[vals>-float('Inf')]

        tanh = torch.nn.Tanh()

        if isinstance(node_embs, torch.sparse.FloatTensor) or isinstance(node_embs, torch.cuda.sparse.FloatTensor):
            node_embs = node_embs.to_dense()
        
        out = node_embs[topk_indices]

        return out.t()


class GCNGRU(torch.nn.Module):
    def __init__(self, A, args):
        super().__init__()
        self.GCN = GCN(A, args)
        self.GRU = GRU(args)
        self.fc = nn.Linear(args.emb_dim, 1)
    
    def forward(self, X, H):
        #Z: (args.n, args.emb_dim)
        #H: (args.n, args.emb_im)
        Z = self.GCN(X)
        H = self.GRU(H, Z)
        out = self.fc(H[-1])
        return H, out

if __name__ == '__main__':
    args = u.Namespace({})
    args.conv_num = 2
    args.gru_num = 2
    args.t = 20
    args.n = 100
    args.emb_dim = 10
    args.feature_dim = 10
    A = torch.zeros((args.n, args.n))
    print(A.size())
    for _ in range(200):
        i = torch.randint(low=0, high=args.n,size=(1,))
        j = torch.randint(low=0, high=args.n,size=(1,))
        A[i,j] = 1
    X = torch.rand(args.n, args.feature_dim)
    X_all =  []
    for i in range(args.t):
        X_all.append(torch.rand(args.n, args.feature_dim))
    X_all = torch.stack(X_all)
    #data_size = args.n*2
    #rows = torch.randint(low=0, high=args.n, size=(data_size,))
    #cols = torch.randint(low=0, high=args.n, size=(data_size,))
    #index = torch.stack([rows, cols])
    #data = torch.ones(data_size)
    #A = torch.sparse.IntTensor(index, data, torch.Size([args.n, args.n]))
    y_all = []
    for i in range(args.t):
        y_all.append(torch.rand(args.n))
    y_all = torch.stack(y_all)
    
    H_zero = []
    for i in range(2):
        H_zero.append(torch.zeros(args.n, args.emb_dim))
    H_zero = torch.stack(H_zero)
    H_zero.requires_grad_ = False

    gcngru = GCNGRU(A, args)
    optimizer = Adam(gcngru.parameters())
    print(gcngru)
    H_ = H_zero
    total_loss = 0
    for i in range(args.t):
        H_, out = gcngru(X, H_)
        y = torch.rand(args.n)
        loss = F.mse_loss(out.squeeze(1), y)
        total_loss +=loss
        print(i, loss)
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    '''
    loss = F.mse_loss(out.squeeze(1), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    '''

        