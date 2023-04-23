import torch
import torch.nn as nn
import math
import torch.nn.functional as F
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, input_dim, split, num_heads):
        super().__init__()
        self.input_dim = input_dim
        self.split = split
        self.num_heads = num_heads
        self.Wq = torch.nn.ParameterList()
        for i in range(num_heads):
            wq = torch.nn.Parameter(torch.rand(input_dim, input_dim//self.num_heads))
            self.reset_param(wq)
            self.Wq.append(wq)
        self.Wk = torch.nn.ParameterList()
        for i in range(num_heads):
            wk = torch.nn.Parameter(torch.rand(input_dim, input_dim//self.num_heads))
            self.reset_param(wk)
            self.Wk.append(wk)

        self.Wv = torch.nn.ParameterList()
        for i in range(num_heads):
            wv = torch.nn.Parameter(torch.rand(input_dim, input_dim//self.num_heads))
            self.reset_param(wv)
            self.Wv.append(wv)

    def reset_param(self,t):
        #Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv,stdv)

    def forward(self, x):
        out = []
        for i in range(self.num_heads):
            z = x.reshape(-1, self.split, self.input_dim)
            q = torch.matmul(z, self.Wq[i])
            k = torch.matmul(z, self.Wk[i])
            v = torch.matmul(z, self.Wv[i])
            k = torch.permute(k, (0,2,1))
            qk = torch.bmm(q,k)
            attn = F.softmax(qk/torch.sqrt(torch.tensor(self.input_dim)),dim=2)
            z = torch.matmul(attn, v)
            out.append(z)
        return torch.cat(out, dim=2)
        
class FFN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(F.relu(self.linear1(x)))

class Transformer(torch.nn.Module):
    def __init__(self, input_dim, num_heads, split,):
        super().__init__()
        self.input_dim = input_dim
        self.num_head = num_heads
        self.split = split
        self.att = MultiHeadAttention(input_dim=input_dim, split=split, num_heads=num_heads)
        self.ln1 = nn.LayerNorm(input_dim)
        self.do1 = nn.Dropout(0.2)
        self.ffn = FFN(input_dim * self.split, input_dim * self.split)
        self.ln2 = nn.LayerNorm(input_dim)
        self.do2 = nn.Dropout(0.2)

    def forward(self, x):
        y = self.att(x)
        y = y.reshape(-1,self.split*self.input_dim)
        y = self.do1(y)
        x = x + y
        y = self.ffn(x)
        y = self.do2(y)
        x = x + y
        x = x.reshape(-1, self.split, self.input_dim)
        x = torch.mean(x, dim=1)
        return x