import torch
import torch.nn as nn
from torch_geometric.nn import Linear
from torch_geometric.data import HeteroData

class HeteroLinear(torch.nn.Module):
    def __init__(self, in_channels_dict, out_channels):
        super().__init__()
        self.linears = nn.ModuleDict()
        for node_type, in_channels in in_channels_dict.items():
            self.linears[node_type] = Linear(in_channels, out_channels, weight_initializer='glorot')

    def forward(self, x_dict):
        x_dict_out = {}
        for node_type, x in x_dict.items():
            x = self.linears[node_type](x)
            x_dict_out[node_type] = x
        return x_dict_out

if __name__=='__main__':
    data = HeteroData()
    data['a'].x = torch.rand(10, 5)
    data['b'].x = torch.rand(20, 10)
    in_channels_dict = {'a':5, 'b':10}
    linear = HeteroLinear(in_channels_dict, 128)
    out=linear(data.x_dict)
    print(out)



