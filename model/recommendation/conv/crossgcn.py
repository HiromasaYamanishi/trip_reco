import torch
import torch.nn as nn
from torch_geometric.nn import Linear
from collections import defaultdict
from torch_scatter.scatter import scatter
import yaml
import numpy as np
import os
#from conv.attention import AttentionModule
#from conv.heterolinear import HeteroLinear
import sys
from torch_geometric.nn.conv.gcnlight_conv import GCNLightConv
#from get_data import get_data
class Path:
    def __init__(self):
        self.df_experience_path = '/home/yamanishi/project/trip_recommend/data/jalan/spot/experience.csv'
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
        self.spot_pref_path = os.path.join(self.data_graph_dir, 'spot_pref.npy')
        self.spot_spot_path = os.path.join(self.data_graph_dir, 'spot_spot.npy')
        self.pref_attr_path = os.path.join(self.data_graph_dir, 'pref_attr.npy')
        self.city_attr_path = os.path.join(self.data_graph_dir, 'city_attr.npy')
        self.spot_img_emb_path = os.path.join(self.data_graph_dir, 'spot_img_emb_ResNet.npy')
        self.category_img_emb_path = os.path.join(self.data_graph_dir, 'category_img_emb.npy')
        self.category_emb_path = os.path.join(self.data_graph_dir, 'category_emb.npy')
        self.spot_img_emb_multi_path = os.path.join(self.data_graph_dir,'spot_img_emb_multi.npy')
        self.spot_img_emb_clip_path = os.path.join(self.data_graph_dir, 'spot_img_emb_clip.npy')


class UserSpotConv(torch.nn.Module):
    def __init__(self, user_spot):
        super().__init__()
        self.n_user = 27094
        self.m_spot = 42852
        self.user_div = torch.zeros(self.n_user).long()
        self.spot_div = torch.zeros(self.m_spot).long()
        user_value, user_count = torch.unique(user_spot[0].cpu(), return_counts=True)
        self.user_div[user_value.long()] = user_count
        spot_value, spot_count = torch.unique(user_spot[1].cpu(), return_counts=True)
        self.spot_div[spot_value.long()] = spot_count
        self.div = torch.sqrt(self.user_div[user_spot[0]]*self.spot_div[user_spot[1]]).unsqueeze(1)
        self.user_spot = user_spot

    def forward(self, spot_x, user_x):
        user_spot = self.user_spot.to(spot_x.device)
        source_spot = spot_x[user_spot[1]]
        source_spot = source_spot/self.div.to(spot_x.device)
        user_out = torch.zeros_like(user_x).to(user_x.device)
        user_out = scatter(source_spot, user_spot[0], out=user_out, dim=0, reduce='sum')
        source_user = user_x[user_spot[0]]
        source_user = source_user/self.div.to(spot_x.device)
        spot_out = torch.zeros_like(spot_x).to(spot_x.device)
        spot_out = scatter(source_user, user_spot[1], out=spot_out, dim=0, reduce='sum')
        del source_user, source_spot, 
        return spot_out, user_out

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
    word_vec_all = np.mean(word_vec_all.reshape(42852, 15, 300), 1)
    return torch.from_numpy(word_vec_all)

class FeatureCrossLayer(torch.nn.Module):
    def __init__(self, in_channels, num_layers=3):
        super().__init__()
        self.in_channels = in_channels
        self.num_layers = num_layers
        self.Ws = torch.nn.ParameterList()
        for i in range(num_layers):
            W = torch.nn.Parameter(torch.rand(in_channels))
            torch.nn.init.normal_(W, std=0.1)
            self.Ws.append(W)
        self.bs = torch.nn.ParameterList()
        for i in range(num_layers):
            b = torch.nn.Parameter(torch.rand(in_channels))
            torch.nn.init.normal_(W, std=0.1)
            self.bs.append(b)

    def forward(self, x):
        init_x = x
        for i in range(self.num_layers):
            x = torch.matmul(torch.sum(torch.mul(x, init_x), dim=1).reshape(-1, 1), self.Ws[i].reshape(1, -1))+ self.bs[i] + x
            x = x/torch.norm(x, dim=1).reshape(-1, 1)
        return x


class CrossGCN(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config['device']
        self.user_spot = torch.from_numpy(np.load('/home/yamanishi/project/trip_recommend/model/recommendation/data/train_edge.npy')).to(self.device)
        self.hidden_channels = config['model']['hidden_channels']
        self.n_users = 27094
        self.m_items = 42852
        self.user_embedding = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.hidden_channels)
        self.spot_embedding = torch.nn.Embedding(num_embeddings=self.m_items, embedding_dim=self.hidden_channels)
        torch.nn.init.normal_(self.user_embedding.weight, std=0.1)
        torch.nn.init.normal_(self.spot_embedding.weight, std=0.1)
        self.num_layers = config['model']['num_layers']
        self.layers = torch.nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(UserSpotConv(self.user_spot))
        self.jalan_graph_dir = '/home/yamanishi/project/trip_recommend/data/jalan/graph'

        self.spot_image_attr = torch.from_numpy(np.load(os.path.join(self.jalan_graph_dir, f'spot_img_emb_multi_ResNet.npy'))).float().to(self.device)
        self.spot_image_attr = torch.mean(self.spot_image_attr.reshape(-1, 5, 512), dim=1)
        self.spot_word_attr = get_wordvec().float().to(self.device)
        self.cross_faeture_dim = self.hidden_channels + self.spot_image_attr.size(1) + self.spot_word_attr.size(1)
        self.spot_cross_layer = FeatureCrossLayer(self.cross_faeture_dim)
        self.user_cross_layer = FeatureCrossLayer(self.cross_faeture_dim)

    def forward(self):
        spot_x = self.spot_embedding.weight
        user_x = self.user_embedding.weight
        spot_x_out = spot_x
        user_x_out = user_x
        for i in range(self.num_layers):
            spot_x, user_x = self.layers[i](spot_x, user_x)
            spot_x_out = spot_x_out + spot_x
            user_x_out = user_x_out + user_x

        spot_x_out = spot_x_out/(self.num_layers+1)
        user_x_out = user_x_out/(self.num_layers+1)

        return spot_x_out, user_x_out

    def bpr_loss(self, spot_x_out, user_x_out, users, pos, neg):
        users_emb = user_x_out[users]
        pos_emb = spot_x_out[pos]
        neg_emb = spot_x_out[neg]
        spot_pos_x = torch.cat([pos_emb, self.spot_image_attr[pos], self.spot_word_attr[pos]], dim=1)
        spot_neg_x = torch.cat([pos_emb, self.spot_image_attr[neg], self.spot_word_attr[neg]], dim=1)
        user_pos_x = torch.cat([users_emb, self.spot_image_attr[pos], self.spot_word_attr[pos]], dim=1)
        user_neg_x = torch.cat([users_emb, self.spot_image_attr[neg], self.spot_word_attr[neg]], dim=1)
        cross_spot_pos, cross_spot_neg = self.spot_cross_layer(spot_pos_x), self.spot_cross_layer(spot_neg_x)
        cross_user_pos, cross_user_neg = self.user_cross_layer(user_pos_x), self.user_cross_layer(user_neg_x)

        users_emb_ego = self.user_embedding(users.to(self.device))
        pos_emb_ego = self.spot_embedding(pos.to(self.device))
        neg_emb_ego = self.spot_embedding(neg.to(self.device))
        pos_scores = torch.sum(torch.mul(users_emb, pos_emb), dim=1)
        neg_scores = torch.sum(torch.mul(users_emb, neg_emb), dim=1)
        pos_scores_cross = torch.sum(torch.mul(cross_user_pos, cross_spot_pos), dim=1)
        neg_scores_cross = torch.sum(torch.mul(cross_user_neg, cross_spot_neg), dim=1)
        cross_weight = 1e-1
        #print(torch.nn.functional.softplus(neg_scores-pos_scores))
        #print(torch.nn.functional.softplus(neg_scores_cross-pos_scores_cross))
        loss = torch.mean(torch.nn.functional.softplus(neg_scores-pos_scores)) + cross_weight*torch.mean(torch.nn.functional.softplus(neg_scores_cross-pos_scores_cross))
        reg_loss = (1/2)*(users_emb_ego.norm(2).pow(2) + 
                            pos_emb_ego.norm(2).pow(2) +
                            neg_emb_ego.norm(2).pow(2))/len(users)
        del users_emb, pos_emb, neg_emb
        del users_emb_ego,pos_emb_ego, neg_emb_ego, pos_scores, neg_scores
        del users, pos, neg

        return loss, reg_loss

    def stageOne(self, users, pos, neg):
        spot_x_out, user_x_out = self.forward()
        loss, reg_loss = self.bpr_loss(spot_x_out, user_x_out, users, pos, neg)
        del spot_x_out, user_x_out
        return loss, reg_loss

    @torch.no_grad()
    def getRating(self):
        spot_out, user_out = self.forward()
        rating = torch.matmul(user_out, spot_out.T)
        return rating




if __name__=='__main__':
    with open('../config.yaml') as f:
        config = yaml.safe_load(f)
    config['k'] = 20
    config['explain_num'] = 10
    config['epoch_num'] = 2500
    config['model']['model_type'] = 'ggnn'
    config['model']['num_layers'] = 2
    config['model']['hidden_channels'] = 64
    config['model']['concat'] = False
    config['model']['ReLU'] = True
    config['model']['pool'] = 'mean'
    config['trainer']['explain_span'] = 50
    config['trainer']['lr'] = 0.0003
    config['trainer']['loss_city_weight'] = 0
    config['trainer']['loss_category_weight'] = 0
    config['trainer']['loss_word_weight'] = 0.2
    config['trainer']['loss_pref_weight'] = 0
    config['trainer']['city_pop_weight']=0
    config['trainer']['spot_pop_weight']=0
    config['data']['word'] = False
    config['data']['city'] = True
    config['data']['category'] = True
    config['data']['pref'] = True

    n_spot = 42852
    n_user = 27904

    spot_x = torch.rand(n_spot, 128)
    user_x = torch.rand(n_user, 128)
    user_spot = torch.from_numpy(np.load('/home/yamanishi/project/trip_recommend/model/recommendation/data/train_edge.npy'))
    device = 'cuda:6'
    config['device'] = device
    config['model']['hidden_channels'] = 128
    config['model']['num_layers'] = 3
    config['model']['pre'] = True
    config['model']['mid'] = True
    config['model']['post'] = True
    config['model']['cat'] = False
    model = CrossGCN(config).to(device)
    spot_x_out, user_x_out = model()
    users = torch.randint(0,100, (20,))
    pos = torch.randint(0,100, (20,))
    neg = torch.randint(0, 100, (20,))
    optim = torch.optim.Adam(model.parameters())
    loss, reg_loss = model.bpr_loss(spot_x_out, user_x_out, users, pos, neg)
    print(loss, reg_loss)
    optim.zero_grad()
    loss = loss+1e-4*reg_loss
    loss.backward()
    optim.step()
    #print(spot_x_out, user_x_out)

    '''
    data = get_data(word=True, category=True, city=True, prefecture=True, multi=True)
    
    model = HeteroLightGCN(data,config)
    data.to(device)
    model.to(device)
    print(model)
    x_dict, out_dict= model(data.x_dict, data.edge_index_dict)
    print(x_dict)
    '''