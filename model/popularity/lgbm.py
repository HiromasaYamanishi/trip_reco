import pandas as pd
import numpy as np
import sys
from catboost import CatBoostRegressor
import lightgbm as lgb
sys.path.append('..')
from collect_data.preprocessing.preprocess_refactor import Path
import sklearn.preprocessing as sp

if __name__=='__main__':
    path = Path()

    mask = np.load(path.valid_idx_path)
    train_mask = mask[mask>=2]
    val_mask = mask[mask==0]
    test_mask = mask[mask==1]

    df_experience = pd.read_csv(path.df_experience_path)
    image_features = np.load(path.spot_img_emb_path)
    enc = sp.OneHotEncoder( sparse=False )
    tfidf_index = np.load(path.tfidf_topk_index_path).astype(int)
    tfidf_index = enc.fit_transform(pd.DataFrame(tfidf_index))
    print(tfidf_index.shape)
    enc = sp.OneHotEncoder( sparse=False )
    place_labels=enc.fit_transform(df_experience[['city_label','category_label','pref_label']])
    print(place_labels.shape)
    X = np.concatenate([image_features, tfidf_index, place_labels], axis=1)
    y = (np.log10(df_experience['page_view']+1)).values

    X_train, X_val, X_test = X[train_mask], X[val_mask], X[test_mask]
    y_train, y_val, y_test = y[train_mask], y[val_mask], y[test_mask]

    cat_features = [i for i in range(512, X_train.shape[1])]
    #model= CatBoostRegressor(iterations=1000,learning_rate=1, depth=5, loss_function='RMSE')
    model = lgb.LGBMRegressor(
            random_state = 71,
        )
    model.fit(X_train, y_train, eval_set=(X_val, y_val),verbose=True)
    pred = model.predict(X_test)
    print(pred.shape, print(y_test.shape))
    print(y, pred)
    print(np.corrcoef(pred, y_test))

