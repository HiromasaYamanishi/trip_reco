from sklearn.svm import SVR
import numpy as np
import pandas as pd
from utils import save_cor
import sys
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
sys.path.append('..')
#from collect_data.preprocessing.preprocess_refactor import Path
from data.jalan.preprocessing import Path

from sklearn.neural_network import MLPRegressor 
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

if __name__ == '__main__':
    path = Path()
    df = pd.read_csv(path.df_experience_path)
    y = np.log10(df['review_count']).values
    X = np.load(path.spot_img_emb_path)
    print(df.columns.values)
    le = LabelEncoder()
    #df['category_label'] = le.fit_transform(df['category'])
    le = LabelEncoder()
    #df['city_label'] = le.fit_transform(df['city'])
    le = LabelEncoder()
    #df['pref_label'] = le.fit_transform(df['都道府県'])
    #oh = OneHotEncoder(sparse=False)
    #place_labels = oh.fit_transform(df[['category_label','city_label','pref_label']])
    word_vec = get_wordvec()
    print(word_vec.shape)
    word_vec = np.mean(word_vec.reshape(42852, 15, 300), 1)
    X = np.concatenate([X,word_vec], 1)
    #X = np.concatenate([X, place_labels, word_vec], 1)
    print('X shape', X.shape)

    valid_id = np.load(path.valid_idx_path)
    train_mask = (valid_id>1)
    val_mask = (valid_id==0)
    test_mask = (valid_id==1)
    X_train, X_val, X_test = X[train_mask], X[val_mask], X[test_mask]
    y_train, y_val, y_test = y[train_mask], y[val_mask], y[test_mask]

    clf = SVR()
    #clf = MLPRegressor()
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    save_cor(y_test, pred,'gt: log(review_count)','pred: log(review_count)','cor_SVR')
    print(np.corrcoef(y_test, pred)[0][1])

    