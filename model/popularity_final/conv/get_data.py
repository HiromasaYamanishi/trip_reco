from torch_geometric.data import HeteroData
import sys
import pandas as pd
import torch
import numpy as np
sys.path.append('../../..')
from data.jalan.preprocessing import Path
def get_data(model_name='ResNet', multi=False, word=True, category=False, city=False, prefecture=False):
    data = HeteroData()
    path = Path()
    df = pd.read_csv('/home/yamanishi/project/trip_recommend/data/jalan/spot/experience_light.csv')
    #df['y']=(df['page_view']/df['page_view'].max())*100
    #df['y'] = np.log10(df['page_view']+1)
    #df['y'] = df['jalan_review_count']
    df['y'] = np.log10(df['review_count'])
    #df['y'] = df['jalan_review_rate']
    mask = np.load(path.valid_idx_path)
    assert len(df)==len(mask)
    train_mask = torch.tensor(mask>=2)
    val_mask = torch.tensor(mask==0)
    test_mask = torch.tensor(mask==1)
    data['spot'].train_mask = train_mask
    data['spot'].valid_mask = val_mask
    data['spot'].test_mask = test_mask
    data['spot'].y = torch.from_numpy(df['y'].values).float()
    data['spot', 'near', 'spot'].edge_index = torch.from_numpy(np.load('/home/yamanishi/project/trip_recommend/data/jalan/graph/spot_spot.npy'))
    #data['spot'].x = torch.from_numpy(np.load(path.spot_img_emb_path)).float()#np.load('spot_emb.npy') #[num_spots, num_features]
    if multi==False:
        data['spot'].x = torch.from_numpy(np.load('/home/yamanishi/project/trip_recommend/data/jalan/graph/spot_img_emb_ResNet.npy')).float()
    else:
        if model_name=='ResNet':
            img_emb_path = '/home/yamanishi/project/trip_recommend/data/jalan/graph/spot_img_emb_multi_ResNet.npy'
            data['spot'].x = torch.from_numpy(np.load(img_emb_path))
        else:
            img_emb_path = os.path.join(path.data_graph_dir, f'spot_img_emb_multi_{model_name}.npy')
            data['spot'].x = torch.from_numpy(np.load(img_emb_path))

    #data['spot'].x = torch.from_numpy(np.load(path.spot_img_emb_clip_path)).float()

    #num_words = np.load(path.word_embs_wiki_path).shape[0]
    #data['word'].x = torch.rand((num_words, 300))
    #data['word'].x = torch.from_numpy(np.load(path.word_emb_clip_path)).float()
    #category_size = len(df['category'].unique())
    #city_size = len(df['city'].unique())
    #pref_size = len(df['都道府県'].unique())
    if word==True:    
    #data['spot','near','spot'].edge_index = torch.from_numpy(np.load(path.spot_spot_path)).long()
        data['word'].x = torch.from_numpy(np.load(path.word_embs_ensemble_path)).float() #[num_spots, num_features]
        #data['word'].x = torch.from_numpy(np.load(path.word_embs_finetune_path)).float()
        spot_word = torch.from_numpy(np.load(path.spot_word_path)).long()
        word_spot = torch.stack([spot_word[1], spot_word[0]]).long()
        data["spot", "relate", "word"].edge_index = torch.from_numpy(np.load(path.spot_word_path)).long() #[2, num_edges_describe]
        data['word', 'revrelate', 'spot'].edge_index = word_spot
    
    if category==True:
        #torch.from_numpy(np.load(path.category_img_emb_path)).float()#torch.rand(category_size, 5)#torch.rand(category_size,10)
        spot_category = torch.from_numpy(np.load(path.spot_category_path)).long()
        category_spot = torch.stack([spot_category[1], spot_category[0]]).long()
        category_size = len(spot_category[1].unique())
        data['category'].x =torch.nn.functional.one_hot(torch.arange(0,category_size), num_classes=category_size).float()
        data['spot', 'has', 'category'].edge_index = torch.from_numpy(np.load(path.spot_category_path)).long()
        data['category', 'revhas', 'spot'].edge_index = category_spot
    if city==True:
        data['city'].x = torch.from_numpy(np.load(path.city_attr_path)).float()#torch.rand(city_size, 5)
        data['city'].y = torch.from_numpy(np.load(path.city_popularity_path))
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

    if prefecture==True and city==True:
        data['pref'].x =  torch.from_numpy(np.load(path.pref_attr_path)).float()
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
    
    return data

if __name__=='__main__':
    data = get_data()
    print(data)