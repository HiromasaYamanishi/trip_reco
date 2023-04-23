from __future__ import print_function, division

import os
from tqdm import tqdm
import yaml
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import HeteroData
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

def get_train_graph(config, model_name='ResNet', multi=True):
    df = pd.read_csv('/home/yamanishi/project/trip_recommend/data/jalan/spot/experience_light.csv')
    df['y'] = np.log10(df['review_count'])
    mask = np.load('/home/yamanishi/project/trip_recommend/data/jalan/graph/valid_idx.npy')
    print('mask is', mask) 
    assert len(df)==len(mask)
    data = HeteroData()
    train_mask = torch.tensor(mask>=2)
    val_mask = torch.tensor(mask==0)
    test_mask = torch.tensor(mask==1)
    data['spot'].train_mask = train_mask
    data['spot'].valid_mask = val_mask
    data['spot'].test_mask = test_mask
    data['spot'].y = torch.from_numpy(df['y'].values).float()
    return data

    
    
def get_content_graph(config, model_name='ResNet', multi=True):
    path = Path()
    print('category graph')
    jalan_graph_dir = '/home/yamanishi/project/trip_recommend/data/jalan/graph/'
    data = HeteroData()
    word = config['data']['word']
    category = config['data']['category']
    city = config['data']['city']
    prefecture = config['data']['prefecture']
    station = config['data']['station']
    if multi:
        #data['spot'].x = torch.from_numpy(np.load('/home/yamanishi/project/trip_recommend/data/graph/spot_img_emb_multi_VGG.npy'))
        data['spot'].x = torch.from_numpy(np.load(os.path.join(jalan_graph_dir, 'spot_img_emb_multi_ResNet.npy'))).float()
    else:
        data['spot'].x = torch.from_numpy(np.load(os.path.join(jalan_graph_dir, f'spot_img_emb_{model_name}.npy')))
    if word:
        data['word'].x = torch.from_numpy(np.load(os.path.join(jalan_graph_dir, 'word_embs_finetune.npy'))).float()
        spot_word = torch.from_numpy(np.load(os.path.join(jalan_graph_dir, 'spot_word.npy')))
        data["spot", "relate", "word"].edge_index = spot_word
        data["word", "revrelate", "spot"].edge_index = torch.stack([spot_word[1], spot_word[0]]).long()
        #data['word', 'pmi', 'word'].edge_index = torch.from_numpy(np.load(os.path.join(jalan_graph_dir, 'word_word.npy')))
        
    if category:
        data['category'].x = torch.from_numpy(np.load(os.path.join(jalan_graph_dir, 'category_emb.npy'))).float()
        spot_category = torch.from_numpy(np.load(os.path.join(jalan_graph_dir, 'spot_category.npy')))
        data['spot', 'has', 'category'].edge_index = spot_category
        data['category', 'revhas', 'spot'].edge_index = torch.stack([spot_category[1], spot_category[0]]).long()
    if city:
        data['city'].x = torch.from_numpy(np.load(os.path.join(jalan_graph_dir, 'city_attr_cat.npy'))).float()#torch.rand(city_size, 5)
        data['city'].y = torch.from_numpy(np.load(os.path.join(jalan_graph_dir, 'city_popularity.npy')))
        data['city'].y = torch.log(data['city'].y)
        city_valid = np.load('/home/yamanishi/project/trip_recommend/data/jalan/graph/city_split.npy')
        data['city'].train_mask= torch.tensor(city_valid>=2)
        data['city'].valid_mask= torch.tensor(city_valid==0)
        data['city'].test_mask= torch.tensor(city_valid==1)
        spot_city = torch.from_numpy(np.load(path.spot_city_path)).long()
        city_spot = torch.stack([spot_city[1], spot_city[0]]).long()
        city_city = torch.from_numpy(np.load(path.city_city_path)).long()
        data['spot','belongs','city'].edge_index = torch.from_numpy(np.load(path.spot_city_path)).long()
        data['city','revbelong','spot'].edge_index = city_spot
        data['city', 'cityadj','city'] = city_city
        data['city'].city_word = torch.load('/home/yamanishi/project/trip_recommend/data/jalan/graph/city_words.pt')

    if config['data']['station']:
        data['station'].x = torch.from_numpy(np.load(os.path.join(jalan_graph_dir, 'station_attr.npy'))).float()
        torch.nn.init.normal_(data['station'].x, std=0.1)
        spot_station = torch.from_numpy(np.load(os.path.join(jalan_graph_dir, 'spot_station.npy')))
        station_spot = torch.stack([spot_station[1], spot_station[0]]).long()
        data['spot', 'at', 'station'] = spot_station
        data['station', 'revat', 'spot'].edge_index = station_spot
        station_station = torch.from_numpy(np.load(os.path.join(jalan_graph_dir, 'station_station.npy')))
        data['station', 'connect', 'station'].edge_index = station_station   
        station_city = torch.from_numpy(np.load(os.path.join(jalan_graph_dir, 'station_city.npy'))).long()
        data['station', 'belong', 'city'].edge_index = station_city
        data['city', 'revbelong', 'station'].edge_index = torch.stack([station_city[1], station_city[0]]).long()

    if prefecture:
        data['pref'].x =  torch.from_numpy(np.load(os.path.join(jalan_graph_dir, 'pref_attr_cat.npy'))).float()
        data['pref'].y = torch.from_numpy(np.load(path.pref_popularity_path))
        data['pref'].y = data['pref'].y/data['pref'].y.max()
        pref_valid = np.load('/home/yamanishi/project/trip_recommend/data/jalan/graph/pref_split.npy')
        data['pref'].train_mask= torch.tensor(pref_valid>=2)
        data['pref'].valid_mask= torch.tensor(pref_valid==0)
        data['pref'].test_mask= torch.tensor(pref_valid==1)
        data['pref', 'prefadj', 'pref'].edge_index = torch.from_numpy(np.load(path.pref_pref_path)).long()
        city_pref = torch.from_numpy(np.load(path.city_pref_path))
        pref_city = torch.stack([city_pref[1], city_pref[0]]).long()
        data['city','belong','pref'].edge_index = city_pref
        data['pref', 'rebelong','city'].edge_index = pref_city
        data['pref'].pref_word = torch.load('/home/yamanishi/project/trip_recommend/data/jalan/graph/pref_words.pt')


    if config['data']['spot']:
        data['spot', 'near', 'spot'].edge_index = torch.from_numpy(np.load('/home/yamanishi/project/trip_recommend/data/jalan/graph/spot_spot.npy'))
    return data

def get_geo_graph(config):
    jalan_graph_dir = '/home/yamanishi/project/trip_recommend/data/jalan/graph/'
    data = HeteroData()
    data['spot', 'near', 'spot'].edge_index = torch.from_numpy(np.load(os.path.join(jalan_graph_dir, 'spot_spot.npy')))
    data['spot', 'near', 'spot'].edge_attr = torch.from_numpy(np.load(os.path.join(jalan_graph_dir, 'spot_spot_distances.npy')))
    return data

def get_citystation_graph(config):
    print('city station')
    data = HeteroData()
    jalan_graph_dir = '/home/yamanishi/project/trip_recommend/data/jalan/graph/'
    if config['data']['city']:
        data['city'].x = torch.from_numpy(np.load(os.path.join(jalan_graph_dir, 'city_attr.npy'))).float()
        spot_city = torch.from_numpy(np.load(os.path.join(jalan_graph_dir, 'spot_city.npy')))
        data['spot','belong', 'city'].edge_index = spot_city
        data['city','revbelong', 'spot'].edge_index=torch.stack([spot_city[1], spot_city[0]]).long()
    if config['data']['station']:
        data['station'].x = torch.from_numpy(np.load(os.path.join(jalan_graph_dir, 'station_attr.npy'))).float()
        torch.nn.init.normal_(data['station'].x, std=0.1)
        spot_station = torch.from_numpy(np.load(os.path.join(jalan_graph_dir, 'spot_station.npy')))
        station_spot = torch.stack([spot_station[1], spot_station[0]]).long()
        data['spot', 'at', 'station'] = spot_station
        data['station', 'revat', 'spot'].edge_index = station_spot
        station_station = torch.from_numpy(np.load(os.path.join(jalan_graph_dir, 'station_station.npy')))
        data['station', 'connect', 'station'].edge_index = station_station   
    return data


def get_data(config):
    print('graph called')
    graph_dict = {}
    graph_dict['content_graph'] = get_content_graph(config).to(config['device'])
    graph_dict['citystation_graph'] = get_citystation_graph(config).to(config['device'])
    print('data loaded')
    return graph_dict

if __name__=='__main__':
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    config['data']['word'] = True
    config['data']['category'] = True
    config['data']['city'] = True
    config['data']['station'] = True
    data = get_citystation_graph(config)
    print(data)