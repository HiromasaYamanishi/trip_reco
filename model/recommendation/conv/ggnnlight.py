import torch
import torch.nn as nn
from torch_geometric.nn import Linear
from collections import defaultdict
from torch_scatter.scatter import scatter
#from attention import AttentionModule
#from heterolinear import HeteroLinear
import sys
import yaml
#from conv.attention import AttentionModule
#from get_data import get_data

class AttentionModule(torch.nn.Module):
    def __init__(self, input_dim, num_heads=4, split=1,):
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.split = split
        self.out_dim = input_dim
        self.per_dim = input_dim//num_heads

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

class HeteroGGNNConv(torch.nn.Module):
    def __init__(self, in_channels_dict, edge_index_dict, out_channels, ReLU, pool):
        super().__init__()

        self.linear = nn.ModuleDict({})
        self.div = defaultdict(int)
        for k in edge_index_dict.keys():
            self.linear['__'.join(k) + '__source'] = Linear(in_channels_dict[k[0]], out_channels, False, weight_initializer='glorot')
            self.linear['__'.join(k) + '__target'] = Linear(in_channels_dict[k[-1]], out_channels, False, weight_initializer='glorot')
            self.div[k[-1]]+=1

        self.gru = nn.ModuleDict({})
        for k in edge_index_dict.keys():
            self.gru['__'.join(k)] = nn.GRUCell(out_channels, out_channels)

        self.ReLU = ReLU
        self.pool = pool

    def forward(self, x_dict, edge_index_dict):
        x_dict_out = {}

        for k,v in edge_index_dict.items():
            source, target = k[0], k[-1]
            source_x = self.linear['__'.join(k) + '__source'](x_dict[source])
            target_x = self.linear['__'.join(k) + '__target'](x_dict[target])
            source_index = v[0].reshape(-1)
            target_index = v[1].reshape(-1)
            out = torch.zeros_like(target_x).to(target_x.device)
            source_x = source_x[source_index]

            #target_x = target_x + scatter(source_x, target_index, out=out, dim=0, reduce='mean')
            aggregated = scatter(source_x, target_index, out=out, dim=0, reduce=self.pool)
            target_x =  self.gru['__'.join(k)](target_x, aggregated)
            if x_dict_out.get(target)!=None:
                x_dict_out[target] += target_x
            else:
                x_dict_out[target] = target_x

        #x_dict_out = {k: self.l2_norm(v) for k,v in x_dict_out.items()}    

        x_dict_out = {k: v/self.div[k] for k,v in x_dict_out.items()}   
        if self.ReLU:
            x_dict_out = {k: v.relu() for k,v in x_dict_out.items()} 
        return x_dict_out

class HeteroGGNNLightConv(torch.nn.Module):
    def __init__(self, in_channels_dict, edge_index_dict, out_channels, ReLU, pool):
        super().__init__()

        self.linear = nn.ModuleDict({})
        self.div = defaultdict(int)

        self.gru = nn.ModuleDict({})
        for k in edge_index_dict.keys():
            self.div[k[-1]]+=1
        for k in edge_index_dict.keys():
            self.gru['__'.join(k)] = nn.GRUCell(out_channels, out_channels)

        self.ReLU = ReLU
        self.pool = pool

    def forward(self, x_dict, edge_index_dict):
        x_dict_out = {}

        for k,v in edge_index_dict.items():
            source, target = k[0], k[-1]
            source_x = x_dict[source]
            target_x = x_dict[target]
            source_index = v[0].reshape(-1)
            target_index = v[1].reshape(-1)
            out = torch.zeros_like(target_x).to(target_x.device)
            source_x = source_x[source_index]

            #target_x = target_x + scatter(source_x, target_index, out=out, dim=0, reduce='mean')
            aggregated = scatter(source_x, target_index, out=out, dim=0, reduce=self.pool)
            target_x =  self.gru['__'.join(k)](target_x, aggregated)
            if x_dict_out.get(target)!=None:
                x_dict_out[target] += target_x
            else:
                x_dict_out[target] = target_x

        #x_dict_out = {k: self.l2_norm(v) for k,v in x_dict_out.items()}    

        x_dict_out = {k: v/self.div[k] for k,v in x_dict_out.items()}   
        if self.ReLU:
            x_dict_out = {k: v.relu() for k,v in x_dict_out.items()} 
        return x_dict_out


class HeteroGGNNLight(torch.nn.Module):
    def __init__(self, data, config, out_channels=1,multi=True):
        super().__init__()
        self.hidden_channels = config['model']['hidden_channels']
        self.num_layers = config['model']['num_layers']
        self.concat = config['model']['concat']
        self.ReLU = config['model']['ReLU']
        self.pool = config['model']['pool']

        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict
        self.layers = torch.nn.ModuleList()
        self.multi = multi
        self.first_in_channels_dict = {node_type: x.size(1) for node_type, x in x_dict.items()}
        self.mid_in_channels_dict = {node_type: self.hidden_channels for node_type in x_dict.keys()}
        if multi==True:
            self.att = AttentionModule(input_dim=512, split=5)
            self.first_in_channels_dict['spot'] = 512
       
        self.layers.append(HeteroGGNNConv(self.first_in_channels_dict, edge_index_dict, self.hidden_channels, self.ReLU, self.pool))

        for i in range(self.num_layers-1):
            self.layers.append(HeteroGGNNLightConv(self.mid_in_channels_dict, edge_index_dict, self.hidden_channels, self.ReLU, self.pool))
        self.linears = HeteroLinear(self.mid_in_channels_dict, out_channels)
        self.multi = multi

    def forward(self, x_dict, edge_index_dict):
        x_dict_all = {node_type: [] for node_type in x_dict.keys()}
        if self.multi==True:
            x_dict['spot'] = self.att(x_dict['spot'])

        for l in self.layers:
            x_dict = l(x_dict, edge_index_dict)
            if not self.concat:continue
            for node_type in x_dict.keys():
                x_dict_all[node_type].append(x_dict[node_type])
        
        out_dict = self.linears(x_dict)
        if self.concat==True:
            #x_dict = {node_type: torch.cat(x, dim=1) for node_type, x in x_dict_all.items()}
            x_dict = {node_type: torch.mean(torch.stack(x, dim=1), dim=1) for node_type, x in x_dict_all.items()}
        return x_dict, out_dict

class HeteroGGNNLightEmb(torch.nn.Module):
    def __init__(self, data, config, out_channels=1,multi=False):
        super().__init__()
        self.hidden_channels = config['model']['hidden_channels']
        self.num_layers = config['model']['num_layers']
        self.concat = config['model']['concat']
        self.ReLU = config['model']['ReLU']
        self.pool = config['model']['pool']
        self.device = config['device']
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict
        self.layers = torch.nn.ModuleList()
        self.multi = multi
        self.first_in_channels_dict = {node_type: x.size(1) for node_type, x in x_dict.items()}
        self.mid_in_channels_dict = {node_type: self.hidden_channels for node_type in x_dict.keys()}
        self.layers.append(HeteroGGNNConv(self.first_in_channels_dict, edge_index_dict, self.hidden_channels, self.ReLU, self.pool))

        for i in range(self.num_layers-1):
            self.layers.append(HeteroGGNNLightConv(self.mid_in_channels_dict, edge_index_dict, self.hidden_channels, self.ReLU, self.pool))
        self.linears = HeteroLinear(self.mid_in_channels_dict, out_channels)
        self.multi = multi
        self.spot_emb = torch.nn.Embedding(num_embeddings=data['spot'].x.size(0), embedding_dim=data['spot'].x.size(1))
        self.user_emb = torch.nn.Embedding(num_embeddings=data['user'].x.size(0), embedding_dim=data['spot'].x.size(1))
        self.spot_emb.weight = torch.nn.Parameter(data['spot'].x)
        self.user_emb.weight = torch.nn.Parameter(data['user'].x)
        '''
        if 'city' in x_dict:
            self.city_emb = torch.nn.Embedding(num_embeddings=data['city'].x.size(0), embedding_dim=data['city'].x.size(1))
            self.city_emb.weight = torch.nn.Parameter(data['city'].x)
        if 'pref' in x_dict:
            self.pref_emb = torch.nn.Embedding(num_embeddings=data['pref'].x.size(0), embedding_dim=data['pref'].x.size(1))
            self.pref_emb.weight = torch.nn.Parameter(data['pref'].x)
        if 'category' in x_dict:
            self.category_emb = torch.nn.Embedding(num_embeddings=data['category'].x.size(0), embedding_dim=data['category'].x.size(1))
            self.category_emb.weight = torch.nn.Parameter(data['category'].x)
        '''
    def forward(self, x_dict, edge_index_dict):
        x_dict['user'] = self.user_emb.weight
        x_dict['spot'] = self.spot_emb.weight
        '''
        if 'city' in x_dict:
            x_dict['city'] = self.city_emb.weight
        if 'pref' in x_dict:
            x_dict['pref'] = self.pref_emb.weight
        if 'category' in x_dict:
            x_dict['category'] = self.category_emb.weight
        '''
        #x_dict_all = {node_type: [] for node_type in x_dict.keys()}
        #x_dict_all['spot'].append(self.spot_emb.weight)
        #x_dict_all['user'].append(self.user_emb.weight)

        for l in self.layers:
            x_dict = l(x_dict, edge_index_dict)
            if not self.concat:continue
            #for node_type in x_dict.keys():
            #    x_dict_all[node_type].append(x_dict[node_type])
        
        out_dict = self.linears(x_dict)
        #if self.concat:
        #    x_dict = {node_type: torch.cat(x, dim=1) for node_type, x in x_dict_all.items()}
            #x_dict = {node_type: torch.mean(torch.stack(x, dim=1), dim=1) for node_type, x in x_dict_all.items()}
        return x_dict, out_dict

    def bpr_loss(self, x_dict, users, pos, neg):
        users_emb = x_dict['user'][users]
        pos_emb = x_dict['spot'][pos]
        neg_emb = x_dict['spot'][neg]
        users_emb_ego = self.user_emb(users.to(self.device))
        pos_emb_ego = self.spot_emb(pos.to(self.device))
        neg_emb_ego = self.spot_emb(neg.to(self.device))
        pos_scores = torch.sum(torch.mul(users_emb, pos_emb), dim=1)
        neg_scores = torch.sum(torch.mul(users_emb, neg_emb), dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores-pos_scores))
        reg_loss = (1/2)*(users_emb_ego.norm(2).pow(2) + 
                            pos_emb_ego.norm(2).pow(2) +
                            neg_emb_ego.norm(2).pow(2))/len(users)
        del users_emb
        del pos_emb
        del neg_emb
        del users_emb_ego,pos_emb_ego, neg_emb_ego, pos_scores, neg_scores
        del x_dict, users, pos, neg

        return loss, reg_loss

if __name__=='__main__':
    with open('../config.yaml') as f:
        config = yaml.safe_load(f)

    config['k'] = 20
    config['device'] = 'cuda:1'
    config['explain_num'] = 10
    config['epoch_num'] = 2500
    config['model']['model_type'] = 'ggnn'
    config['model']['num_layers'] = 4
    config['model']['hidden_channels'] = 256
    config['model']['concat'] = True
    config['model']['ReLU'] = True
    config['trainer']['explain_span'] = 50
    config['trainer']['lr'] = 0.0003
    config['trainer']['loss_city_weight'] = 0
    config['trainer']['loss_category_weight'] = 0
    config['trainer']['loss_word_weight'] = 0.2
    config['trainer']['loss_pref_weight'] = 0
    config['trainer']['city_pop_weight']=0
    config['trainer']['spot_pop_weight']=0
    config['data']['word'] = True
    config['data']['city'] = True
    config['data']['category'] = True
    config['data']['pref'] = True
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    data = get_data(category=True, city=True, prefecture=True, multi=True)
    
    model = HeteroGGNN(data, config)
    data.to(device)
    model.to(device)
    print(model)
    x_dict, out_dict= model(data.x_dict, data.edge_index_dict)
    print(x_dict)