import torch

def get_itenary_embedding(sequence):
    itenary_embedding = []
    for spot_name in sequence:
        embedding=get_embedding(spot_name)
        itenary_embedding.append(embedding)
    return torch.cat(itenary_embedding, dim=1)

class AttentionModule(torch.nn.Module):
    def __init__(self, input_dim, num_heads=4, split=1, out_dim=512):
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.split = split
        self.out_dim = out_dim
        self.per_dim = out_dim//num_heads

        self.W = torch.nn.ModuleList([Linear(input_dim, self.per_dim, False, weight_initializer='glorot') for _ in range(num_heads)])
        self.q = torch.nn.ParameterList([])
        for _ in range(num_heads):
            q_ =torch.nn.Parameter(torch.zeros(size=(self.per_dim, 1)))
            nn.init.xavier_uniform_(q_.data, gain=1.414)
            self.q.append(q_)
        
        self.LeakyReLU = torch.nn.LeakyReLU(0.2)

    def forward(self, x):
        out = []
        x = x.resize(x.size()[0],self.split, self.input_dim)
        for i in range(self.num_heads):
            W = self.W[i]
            q = self.q[i]
            x_ = W(x)
            att = self.LeakyReLU(torch.matmul(x_, q))
            att = torch.nn.functional.softmax(att, dim=1)
            att = torch.broadcast_to(att, x_.size())
            x_= (x_*att).sum(dim=1)
            out.append(x_)
        return torch.cat(out, dim=1)

class PredictItenary(torch.nn.Module):
    def __init__(self, num_heads=8):
        super().__init__()
        self.attention = AttentionModule()

    def forward(self, x):
        x = self.attention(x)

class GGR
        