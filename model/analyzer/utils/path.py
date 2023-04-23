import os

class Path:
    def __init__(self):
        self.df_experience_path = '/home/yamanishi/project/trip_recommend/data/jalan/spot/experience.csv'
        self.df_experience_light_path = '/home/yamanishi/project/trip_recommend/data/jalan/spot/experience_light.csv'
        self.df_review_path = '/home/yamanishi/project/trip_recommend/data/jalan/review/df_all.csv'
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
        self.tfidf_topk_index_path = os.path.join(self.data_graph_dir, 'tfidf_topk_index.npy')
        self.tfidf_top_words_path = os.path.join(self.data_graph_dir, 'tfidf_top_words.npy')
        self.tfidf_word_path = os.path.join(self.data_graph_dir,'tfidf_words.npy')
        self.word_popularity_path = os.path.join(self.data_graph_dir, 'word_popularity.npy')
        self.word_embs_path = os.path.join(self.data_graph_dir,'word_embs.npy')
        self.word_embs_wiki_path = os.path.join(self.data_graph_dir,'word_embs_wiki.npy')
        self.word_embs_finetune_path = os.path.join(self.data_graph_dir,'word_embs_finetune.npy')
        self.word_emb_clip_path = os.path.join(self.data_graph_dir,'word_emb_clip.npy')
        self.spot_word_path = os.path.join(self.data_graph_dir,'spot_word.npy')
        self.spot_category_path = os.path.join(self.data_graph_dir, 'spot_category.npy')
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
        self.spot_img_emb_multi_path = os.path.join(self.data_graph_dir,'spot_img_emb_multi.npy')
        self.spot_img_emb_clip_path = os.path.join(self.data_graph_dir, 'spot_img_emb_clip.npy')