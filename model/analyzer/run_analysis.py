import yaml
import argparse
import sys
sys.path.append('..')
from utils.path import Path
from model.trip_popularity import MyHetero

#TODO: change input to config style
def get_data(config):
    data = HeteroData()
    path = Path()
    #df = pd.read_csv('/home/yamanishi/project/trip_recommend/data/df_experience.csv')
    df = pd.read_csv(path.df_experience_path)
    #df['y']=(df['page_view']/df['page_view'].max())*100
    #df['y'] = np.log10(df['page_view']+1)
    #df['y'] = df['jalan_review_count']
    df['y'] = np.log10(df['jalan_review_count']+1)
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
    #data['spot'].x = torch.from_numpy(np.load(path.spot_img_emb_path)).float()#np.load('spot_emb.npy') #[num_spots, num_features]
    if config['data']['multi']==False:
        data['spot'].x = torch.from_numpy(np.load(path.spot_img_emb_path)).float()
    else:
        if config['data']['model_name']=='ResNet':
            img_emb_path = '/home/yamanishi/project/trip_recommend/data/graph/spot_img_emb_multi.npy'
            data['spot'].x = torch.from_numpy(np.load(img_emb_path))
        else:
            img_emb_path = os.path.join(path.data_graph_dir, f'spot_img_emb_multi_{model_name}.npy')
            data['spot'].x = torch.from_numpy(np.load(img_emb_path))

    #data['spot'].x = torch.from_numpy(np.load(path.spot_img_emb_clip_path)).float()

    num_words = np.load(path.word_embs_path).shape[0]
    #data['word'].x = torch.rand((num_words, 300))
    #data['word'].x = torch.from_numpy(np.load(path.word_emb_clip_path)).float()
    category_size = len(df['category'].unique())
    city_size = len(df['city'].unique())
    pref_size = len(df['都道府県'].unique())
    if config['data']['word']==True:    
    #data['spot','near','spot'].edge_index = torch.from_numpy(np.load(path.spot_spot_path)).long()
        data['word'].x = torch.from_numpy(np.load(path.word_embs_finetune_path)).float() #[num_spots, num_features]
        spot_word = torch.from_numpy(np.load(path.spot_word_path)).long()
        word_spot = torch.stack([spot_word[1], spot_word[0]]).long()
        data["spot", "relate", "word"].edge_index = torch.from_numpy(np.load(path.spot_word_path)).long() #[2, num_edges_describe]
        data['word', 'revrelate', 'spot'].edge_index = word_spot
    
    if config['data']['category']:
        data['category'].x =torch.rand(category_size,10)#torch.from_numpy(np.load(path.category_img_emb_path)).float()#torch.rand(category_size, 5)#torch.rand(category_size,10)
        spot_category = torch.from_numpy(np.load(path.spot_category_path)).long()
        category_spot = torch.stack([spot_category[1], spot_category[0]]).long()
        data['spot', 'has', 'category'].edge_index = torch.from_numpy(np.load(path.spot_category_path)).long()
        data['category', 'revhas', 'spot'].edge_index = category_spot
    if config['data']['city']:
        data['city'].x = torch.from_numpy(np.load(path.city_attr_path)).float()#torch.rand(city_size, 5)
        spot_city = torch.from_numpy(np.load(path.spot_city_path)).long()
        city_spot = torch.stack([spot_city[1], spot_city[0]]).long()
        city_city = torch.from_numpy(np.load(path.city_city_path)).long()
        data['spot','belongs','city'].edge_index = torch.from_numpy(np.load(path.spot_city_path)).long()
        data['city','revbelong','spot'].edge_index = city_spot
        data['city', 'cityadj','city'] = city_city

    if config['data']['prefecture']==True and config['data']['city']==True:
        data['pref'].x =  torch.from_numpy(np.load(path.pref_attr_path)).float()
        spot_pref = torch.from_numpy(np.load(path.spot_pref_path)).long()
        pref_spot = torch.stack([spot_pref[1], spot_pref[0]]).long()
        data['pref', 'prefadj', 'pref'].edge_index = torch.from_numpy(np.load(path.pref_pref_path)).long()
        city_pref = torch.from_numpy(np.load(path.city_pref_path))
        pref_city = torch.stack([city_pref[1], city_pref[0]]).long()
        data['city','belong','pref'].edge_index = city_pref
        data['pref', 'rebelong','city'].edge_index = pref_city
    
    
    return data

def get_model(config):
    return MyHetero(config)

class AnalysisRunner:
    self.configuration = None

    def __init__(self, config):
        self.data = get_data(config)
        self.model = get_model(config)

        self.analyzer= GraphMaskAnalyzer()
        self.analyzer.initialise_model(model)

    def run_analysis():
        self.analyzer.fit()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Train according to a specified configuration file.')
    parser.add_argument("--gpu", default=-1, type=int)
    parser.add_argument("--configuration", default="configurations/trip_popularity.yaml")
    args = parser.parse_args()

    with open(args.configuration) as file:
        config = yaml.safe_load(file)

    analyser = AnalysisRunner(config)
    analyser.run_analysis()