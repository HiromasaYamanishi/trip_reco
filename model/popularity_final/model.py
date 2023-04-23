import torch
from conv.ggnn import HeteroGGNN
#from dataloader import get_geo_graph, get_content_graph, get_citystation_graph

import torch
import torch.nn as nn
from torch_geometric.nn import Linear
import yaml
import os
from tqdm import tqdm
import yaml
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import HeteroData
from torch_geometric.nn.conv import SAGEConv, GATConv, GCNConv, GatedGraphConv
from conv.attention import AttentionModule
from dataloader import get_content_graph, get_citystation_graph, get_geo_graph, get_train_graph
from conv.spgnn import SPGNN
from conv.tpgnn import TPGNN
from conv.contextgnn import ContextGNN
from conv.deeptour import DeepTourModel
from conv.ggnn import HeteroGGNN
from torch_geometric.utils import softmax
import sys


import torch
from torch.nn.parameter import Parameter
import numpy as np
from torch import sigmoid
#sys.path.append('../../')
#from data.jalan.preprocessing import Path

class Path:
    def __init__(self):
        self.df_experience_path = '/home/yamanishi/project/trip_recommend/data/jalan/spot/experience.csv'
        self.df_experience_spare_path = '/home/yamanishi/project/trip_recommend/data/jalan/spot/experience_spare.csv'
        self.df_experience_light_path = '/home/yamanishi/project/trip_recommend/data/jalan/spot/experience_light.csv'
        self.df_review_path = '/home/yamanishi/project/trip_recommend/data/jalan/review/review_all.csv'
        #self.df_review = pd.read_csv(self.df_review_path)

        self.data_graph_dir = '/home/yamanishi/project/trip_recommend/data/jalan/graph'
        self.data_dir = '/home/yamanishi/project/trip_recommend/data/jalan/data'
        self.flickr_image_dir = '/home/yamanishi/project/trip_recommend/data/flickr_image'
        self.jalan_image_dir = '/home/yamanishi/project/trip_recommend/data/jalan_image'
        self.category_image_dir = '/home/yamanishi/project/trip_recommend/data/category_image'

        self.valid_idx_path = os.path.join(self.data_graph_dir, 'valid_idx.npy')
        self.spot_index_path = os.path.join(self.data_graph_dir,'spot_index.pkl')
        self.index_spot_path = os.path.join(self.data_graph_dir,'index_spot.pkl')
        self.index_word_path = os.path.join(self.data_graph_dir,'index_word.pkl')
        self.word_index_path = os.path.join(self.data_graph_dir,'word_index.pkl')
        self.city_index_path = os.path.join(self.data_graph_dir,'city_index.pkl')
        self.index_city_path = os.path.join(self.data_graph_dir,'index_city.pkl')
        self.pref_index_path = os.path.join(self.data_graph_dir,'pref_index.pkl')
        self.index_pref_path = os.path.join(self.data_graph_dir,'index_pref.pkl')
        self.category_index_path = os.path.join(self.data_graph_dir, 'category_index.pkl')
        self.index_category_path = os.path.join(self.data_graph_dir, 'index_category.pkl')
        self.tfidf_topk_index_path = os.path.join(self.data_graph_dir, 'tfidf_topk_index.npy')
        self.tfidf_top_words_path = os.path.join(self.data_graph_dir, 'tfidf_top_words.npy')
        self.tfidf_word_path = os.path.join(self.data_graph_dir,'tfidf_words.npy')
        self.tfidf_topk_index_spare_path = os.path.join(self.data_graph_dir, 'tfidf_topk_index_spare.npy')
        self.tfidf_top_words_spare_path = os.path.join(self.data_graph_dir, 'tfidf_top_words_spare.npy')
        self.tfidf_word_spare_path = os.path.join(self.data_graph_dir,'tfidf_words_spare.npy')
        self.tfidf_word_th_path = os.path.join(self.data_graph_dir,'tfidf_words_th.npy')
        self.word_popularity_path = os.path.join(self.data_graph_dir, 'word_popularity.npy')
        self.word_embs_path = os.path.join(self.data_graph_dir,'word_embs.npy')
        self.word_embs_th_path = os.path.join(self.data_graph_dir, 'word_embs_th.npy')
        self.word_embs_wiki_path = os.path.join(self.data_graph_dir,'word_embs_wiki.npy')
        self.word_embs_finetune_path = os.path.join(self.data_graph_dir,'word_embs_finetune.npy')
        self.word_embs_ensemble_path = os.path.join(self.data_graph_dir,'word_embs_ensemble.npy')
        self.word_emb_clip_path = os.path.join(self.data_graph_dir,'word_emb_clip.npy')
        self.spot_word_path = os.path.join(self.data_graph_dir,'spot_word.npy')
        self.spot_word_th_path = os.path.join(self.data_graph_dir,'spot_word_th.npy')
        self.spot_category_path = os.path.join(self.data_graph_dir, 'spot_category.npy')
        self.spot_popularity_path = os.path.join(self.data_graph_dir, 'spot_popularity.npy')
        self.spot_city_path = os.path.join(self.data_graph_dir, 'spot_city.npy')
        self.city_pref_path = os.path.join(self.data_graph_dir, 'city_pref.npy')
        self.city_adj_path = os.path.join(self.data_graph_dir, 'city_adj.pkl')
        self.city_city_path = os.path.join(self.data_graph_dir, 'city_city.npy')
        self.city_popularity_path = os.path.join(self.data_graph_dir, 'city_popularity.npy')
        self.pref_popularity_path = os.path.join(self.data_graph_dir, 'pref_popularity.npy')
        self.pref_pref_path = os.path.join(self.data_graph_dir, 'pref_pref.npy')
        self.pref_attr_cat_path = os.path.join(self.data_graph_dir, 'pref_attr_cat.npy')
        self.spot_pref_path = os.path.join(self.data_graph_dir, 'spot_pref.npy')
        self.spot_spot_path = os.path.join(self.data_graph_dir, 'spot_spot.npy')
        self.pref_attr_path = os.path.join(self.data_graph_dir, 'pref_attr.npy')
        self.city_attr_path = os.path.join(self.data_graph_dir, 'city_attr.npy')
        self.city_attr_cat_path = os.path.join(self.data_graph_dir, 'city_attr_cat.npy')
        self.spot_img_emb_path = os.path.join(self.data_graph_dir, 'spot_img_emb_ResNet.npy')
        self.category_img_emb_path = os.path.join(self.data_graph_dir, 'category_img_emb.npy')
        self.category_emb_path = os.path.join(self.data_graph_dir, 'category_emb.npy')
        self.spot_img_emb_multi_path = os.path.join(self.data_graph_dir,'spot_img_emb_multi.npy')
        self.spot_img_emb_clip_path = os.path.join(self.data_graph_dir, 'spot_img_emb_clip.npy')

def get_wordvec():
    path = Path()
    word_embs=np.load(path.word_embs_finetune_path)
    word_indexs = np.load(path.tfidf_topk_index_path)
    top_words = np.load(path.tfidf_top_words_path)
    tfidf_words = np.load(path.tfidf_word_path)
    word_vec_all= []
    for ind in word_indexs:
        word_vec_all.append(np.concatenate(word_embs[ind]))
    word_vec_all = np.array(word_vec_all)
    return word_vec_all

class DistanceGNN(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_channels = config['model']['hidden_channels']
        self.device = config['device']
        self.linear = torch.nn.Linear(self.hidden_channels,self.hidden_channels)
        self.query_w = torch.nn.Linear(self.hidden_channels+512+self.hidden_channels, self.hidden_channels)
        self.value_w = torch.nn.Linear(self.hidden_channels, self.hidden_channels)
        self.dist_unit = 50
        self.dist_emb = torch.nn.Embedding(num_embeddings=self.config['data']['spot']//self.dist_unit, embedding_dim=self.hidden_channels).float()
        torch.nn.init.normal_(self.dist_emb.weight)
        self.softmax = nn.Softmax(dim=0)
        if self.config['model']['conv']=='sage':
            self.conv = SAGEConv(self.hidden_channels, self.hidden_channels)
        elif self.config['model']['conv']=='gat':
            self.conv = GATConv(self.hidden_channels, self.hidden_channels)
        elif self.config['model']['conv']=='gcn':
            self.conv = GCNConv(self.hidden_channels, self.hidden_channels)
        elif self.config['model']['conv']=='ggnn':
            self.conv = GatedGraphConv(self.hidden_channels, self.hidden_channels)
        self.neighbor_ratio = config['data']['neighbor_ratio']

        
    def forward(self, x, edge_index, distances, aerial_emb): #x: [spot_num, 812] edge_index[2, edge_num], distances[edge_num], aerial_emb[spot_num, hidden_channel]
        x = self.linear(x)
        source_x = x[edge_index[0]].to(self.device)
        target_x = x[edge_index[1]].to(self.device)
        source_aerial = aerial_emb[edge_index[0]].to(self.device)
        distances = torch.div(distances, self.dist_unit, rounding_mode='floor')
        dist_x = self.dist_emb(distances)
        query = torch.cat([source_x, source_aerial, dist_x], dim=1)
        query = self.query_w(query)
        value = self.value_w(target_x)
        prob = -torch.norm(query-value, dim=1)
        prob = self.softmax(prob)
        edge_selected = torch.multinomial(prob, int(edge_index.shape[1]*self.neighbor_ratio))
        edge_selected = edge_index[:, edge_selected]
        #edge_selected = torch.cat([edge_index[:, edge_selected], torch.tensor([np.arange(42852), np.arange(42852)]).to(self.device)], dim=1)
        aggregated = self.conv(x, edge_selected)
        return aggregated

class HardConcrete(torch.nn.Module):

    def __init__(self, beta=1 / 3, gamma=-0.2, zeta=1.0, fix_temp=True, loc_bias=3):
        super(HardConcrete, self).__init__()

        self.temp = beta if fix_temp else Parameter(torch.zeros(1).fill_(beta))
        self.gamma = gamma
        self.zeta = zeta
        self.gamma_zeta_ratio = np.math.log(-gamma / zeta)

        self.loc_bias = loc_bias

    def forward(self, input_element, summarize_penalty=True):
        input_element = input_element + self.loc_bias

        if self.training:
            u = torch.empty_like(input_element).uniform_(1e-6, 1.0-1e-6)

            s = sigmoid((torch.log(u) - torch.log(1 - u) + input_element) / self.temp)

            penalty = sigmoid(input_element - self.temp * self.gamma_zeta_ratio)
            penalty = penalty
        else:
            s = sigmoid(input_element)
            penalty = torch.zeros_like(input_element)

        if summarize_penalty:
            penalty = penalty.mean()

        s = s * (self.zeta - self.gamma) + self.gamma

        clipped_s = self.clip(s)

        if True:
            hard_concrete = (clipped_s > 0.5).float()
            clipped_s = clipped_s + (hard_concrete - clipped_s).detach()

        return clipped_s, penalty

    def clip(self, x, min_val=0, max_val=1):
        return x.clamp(min_val, max_val)
    
class DistGNN(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_channels = config['model']['hidden_channels']
        self.device = config['device']
        self.linear = torch.nn.Linear(self.hidden_channels,self.hidden_channels)
        self.query_w = torch.nn.Linear(self.hidden_channels+512+self.hidden_channels, self.hidden_channels)
        self.value_w = torch.nn.Linear(self.hidden_channels, self.hidden_channels)
        self.dist_unit = 50
        self.dist_emb = torch.nn.Embedding(num_embeddings=self.config['data']['spot']//self.dist_unit, embedding_dim=self.hidden_channels).float()
        torch.nn.init.normal_(self.dist_emb.weight)
        
        self.softmax = nn.Softmax(dim=0)
        if self.config['model']['conv']=='sage':
            self.conv = SAGEConv(self.hidden_channels, self.hidden_channels)
        elif self.config['model']['conv']=='gat':
            self.conv = GATConv(self.hidden_channels, self.hidden_channels)
        elif self.config['model']['conv']=='gcn':
            self.conv = GCNConv(self.hidden_channels, self.hidden_channels)
        elif self.config['model']['conv']=='ggnn':
            self.conv = GatedGraphConv(self.hidden_channels, self.hidden_channels)
        self.neighbor_ratio = config['data']['neighbor_ratio']
        
        
    def forward(self, x, edge_index, distances, aerial_emb): #x: [spot_num, 812] edge_index[2, edge_num], distances[edge_num], aerial_emb[spot_num, hidden_channel]
        x = self.linear(x)
        source_x = x[edge_index[0]].to(self.device)
        target_x = x[edge_index[1]].to(self.device)
        source_aerial = aerial_emb[edge_index[0]].to(self.device)
        distances = torch.div(distances, self.dist_unit, rounding_mode='floor')
        dist_x = self.dist_emb(distances)
        query = torch.cat([source_x, source_aerial, dist_x], dim=1)
        query = self.query_w(query)
        value = self.value_w(target_x)
        prob = -torch.norm(query-value, dim=1)
        prob = self.softmax(prob)
        edge_selected = torch.multinomial(prob, int(edge_index.shape[1]*self.neighbor_ratio))
        edge_selected = edge_index[:, edge_selected]
        #edge_selected = torch.cat([edge_index[:, edge_selected], torch.tensor([np.arange(42852), np.arange(42852)]).to(self.device)], dim=1)
        aggregated = self.conv(x, edge_selected)
        return aggregated
        

    
class DeepTour(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config['device']
        self.hidden_channels = config['model']['hidden_channels']
        self.train_graph = get_train_graph(config)
        self.content_graph = torch.load('./new_spot/new_spot.pt').to(self.device)#get_content_graph(config).to(self.device)
        print(self.content_graph)
        #self.content_gnn = DeepTourModel(self.content_graph, config).to(self.device)
        self.content_gnn = HeteroGGNN(self.content_graph, config, multi=False).to(self.device)
        #self.fc = Linear(config['model']['hidden_channels']*1, 1).to(self.device)
        self.fc = Linear(self.hidden_channels*1, 1).to(self.device)
        self.word_emb = torch.tensor(get_wordvec().reshape(-1, 15, 300)).to(self.device).float()
        image_emb = np.load('/home/yamanishi/project/trip_recommend/data/jalan/graph/spot_img_emb_multi_ResNet.npy')
        self.image_emb = torch.tensor(image_emb.reshape(-1, 5, 512)).to(self.device).float()
        self.word_emb = torch.mean(self.word_emb, dim=1)
        self.image_emb = torch.mean(self.image_emb, dim=1)
        self.spot_emb = torch.cat([self.image_emb, self.word_emb], dim=1)
        self.spot_embedding = torch.nn.Embedding(num_embeddings=self.spot_emb.shape[0], embedding_dim=self.spot_emb.shape[1]).to(self.device)
        self.spot_embedding.weight = torch.nn.Parameter(self.spot_emb)
        self.word_linear = torch.nn.Linear(300, self.hidden_channels).to(self.device)
        self.image_linear = torch.nn.Linear(512, self.hidden_channels).to(self.device)
        self.w1 = torch.nn.Parameter(torch.normal(0, 0.1, size=(self.hidden_channels,))).to(self.device)
        self.w2 = torch.nn.Parameter(torch.normal(0, 0.1, size=(self.hidden_channels,))).to(self.device)
        self.w3 = torch.nn.Parameter(torch.normal(0, 0.1, size=(self.hidden_channels,))).to(self.device)
        self.jalan_graph_dir = '/home/yamanishi/project/trip_recommend/data/jalan/graph/'
        if config['data']['spot']:
            spot = config['data']['spot']
            self.v = torch.from_numpy(np.load(os.path.join(self.jalan_graph_dir, f'spot_spot_edge_{spot}.npy'))).to(self.device)
            self.distances = torch.from_numpy(np.load(os.path.join(self.jalan_graph_dir, f'spot_spot_distance_{spot}.npy'))).to(self.device)
            mask = self.v[0]!=self.v[1]
            self.v = self.v[:,mask]
            self.distances = self.distances[mask]
            self.aerial_emb = torch.from_numpy(np.load(os.path.join(self.jalan_graph_dir, 'aerial_img_emb_ResNet.npy'))).float().to(self.device)
        #self.distance_gnn = DistanceGNN(config).to(self.device)
        #self.spatial_heatmap = torch.tensor(np.load('/home/yamanishi/project/trip_recommend/data/jalan/graph/spatial_heatmap.npy')).to(self.device)
        #self.conv = torch.nn.Conv2d(290, 10, kernel_size=3, stride=1).to(self.device)
        #self.pool = torch.nn.MaxPool2d(3, stride=2).to(self.device)
        '''
        self.context_gnn = ContextGNN(config['model']['hidden_channels']).to(self.device)
        self.citystation_graph = get_citystation_graph(config).to(self.device)
        self.citystation_graph['spot'].x = torch.rand(self.content_graph['spot'].x.size(0), config['model']['hidden_channels'])
        self.geo_graph = get_geo_graph(config).to(self.device)
        self.geo_gnn = SPGNN(config).to(self.device)
        self.citystation_gnn = TPGNN(self.citystation_graph, config).to(self.device)
        self.hidden_channels = config['model']['hidden_channels']
        self.geo_linear = Linear(self.hidden_channels*3, self.hidden_channels).to(self.device)
        self.fc = Linear(config['model']['hidden_channels']*2, 1).to(self.device)
        '''
    def forward(self):
        content_graph = self.content_graph #content graph, city station graph
        content_x_dict, content_out_dict = self.content_gnn(content_graph.x_dict, content_graph.edge_index_dict)
        x = content_x_dict['spot']
        #if self.config['data']['spot']:
        #    distance_x = self.distance_gnn(x, self.v, self.distances, self.aerial_emb)
            #print(x.shape)
            #print(distance_x.shape)
            #x = x + distance_x
        #spatial_x = self.pool(self.conv(torch.tensor(self.spatial_heatmap).float().relu())).view(42852, -1)
        #aerial_x = self.aerial_emb
        #print(x.shape, distance_x.shape, spatial_x.shape, aerial_x.shape)
        #x = torch.cat([x, spatial_x], dim=1)
        return self.fc(x), content_x_dict
        if self.config['data']['spot']:
            spot_x = self.spot_embedding.weight
            distance_x = self.distance_gnn(spot_x, self.v, self.distances, self.aerial_emb)
            x = distance_x
            content_x_dict = {}
            #x = torch.cat([x, distance_x], dim=1)
        out = self.fc(x)
        return out, content_x_dict
        image_x = self.image_linear(self.image_emb)
        #image_x = image_x/torch.norm(image_x, dim=2).unsqueeze(-1) #[spot_num, 5, hidden]
        word_x = self.word_linear(self.word_emb)
        #word_x = word_x/torch.norm(word_x, dim=2).unsqueeze(-1) #[spot_num, 5, hidden]
        image_x = torch.mean(image_x, dim=1)
        word_x = torch.mean(word_x, dim=1)
        x = torch.cat([image_x, word_x, content_x_dict['spot']], dim=1)
        return self.fc(x)
        feature_crossing = (image_x.reshape(-1, 5, 1, self.hidden_channels)) * (word_x.reshape(-1, 1, 15, self.hidden_channels)) #[spot_num, 5, 15, hidden_channels]
        feature_weight = torch.matmul(feature_crossing, self.w1) #[spot_num, 5, 15]
        image_weight = torch.softmax(feature_weight, dim=2).unsqueeze(-1) #[spot_num, 5, 15, 1]
        word_weight = torch.softmax(feature_weight, dim=1).unsqueeze(-1) #[spot_num, 5, 15, 1]
        image_x_tmp = torch.sum(feature_crossing*image_weight, dim=2) #[spot_num, 5, hidden]
        word_x_tmp = torch.sum(feature_crossing*word_weight, dim=1) #[spot_num, 15, hidden]
        image_x_sum = torch.sum(image_x, dim=1).reshape(-1, 1, self.hidden_channels) #[spot_num, 1, hidden]
        word_x_sum = torch.sum(word_x, dim=1).reshape(-1, 1, self.hidden_channels) #[spot_num, 1, hidden]
        image_weight = torch.matmul(image_x * word_x_sum, self.w2) #[spot_num, 5]
        word_weight = torch.matmul(word_x * image_x_sum , self.w3)#[spot_num, 15]
        image_weight = torch.softmax(image_weight, dim=1).unsqueeze(2) #[spot_num, 5, 1]
        word_weight = torch.softmax(word_weight, dim=1).unsqueeze(2) #[spot_num, 15, 1]
        image_x = torch.sum(image_x_tmp* image_weight, dim=1)
        word_x = torch.sum(word_x_tmp*word_weight, dim=1)
        #image_x_weight = image_x*image_x_sum.reshape()
        x = torch.cat([image_x, word_x, content_x_dict['spot']], dim=1)
        return self.fc(x)
        context_x_dict = self.context_gnn({'spot': content_x_dict['spot']})

        geo_graph = self.geo_graph
        geo_graph['spot'].x = content_x_dict['spot']
        geo_x_dict = self.geo_gnn(geo_graph.x_dict, geo_graph.edge_index_dict, geo_graph.edge_attr_dict,)
        citystation_graph = self.citystation_graph
        citystation_graph['spot'].x = content_x_dict['spot']
        citystation_x_dict = self.citystation_gnn(citystation_graph.x_dict, citystation_graph.edge_index_dict)
        geo_out = self.geo_linear(torch.cat([citystation_x_dict['spot'], geo_x_dict['spot'], context_x_dict['spot']], dim=1))
        concat_x = torch.cat([content_x_dict['spot'],geo_out], dim=1)
        out = self.fc(concat_x)
        return out

if __name__=='__main__':
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    config['data']['word'] = True
    config['data']['category'] = False
    config['data']['station'] = False
    config['data']['city'] = False
    config['data']['station'] = False
    config['data']['prefecture'] = False
    config['data']['spot'] = None
    config['model']['ReLU'] = True
    config['model']['tpgnn_layers'] = 2
    config['model']['spgnn_layers'] = 2
    config['model']['hidden_channels'] = 128
    config['device'] = 'cuda:1'
    gnn = DeepTour(config)
    out=gnn()
    optim = torch.optim.Adam(gnn.parameters())
    y = torch.rand(out.size(0)).to(config['device'])
    loss = torch.nn.functional.mse_loss(y.reshape(-1,1), out)
    optim.zero_grad()
    loss.backward()
    optim.step()
    