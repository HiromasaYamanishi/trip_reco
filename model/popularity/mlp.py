import pandas as pd
import numpy as np
import sys
from catboost import CatBoostRegressor
import lightgbm as lgb
sys.path.append('..')
from collect_data.preprocessing.preprocess_refactor import Path
import sklearn.preprocessing as sp
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam

class EarlyStopping:
    """earlystoppingクラス"""

    def __init__(self, patience=5, verbose=False, path='checkpoint/checkpoint_model.pth'):
        """引数：最小値の非更新数カウンタ、表示設定、モデル格納path"""

        self.patience = patience    #設定ストップカウンタ
        self.verbose = verbose      #表示の有無
        self.counter = 0            #現在のカウンタ値
        self.best_score = None      #ベストスコア
        self.early_stop = False     #ストップフラグ
        self.val_loss_min = np.Inf   #前回のベストスコア記憶用
        self.path = path             #ベストモデル格納path

    def __call__(self, val_loss, model):
        """
        特殊(call)メソッド
        実際に学習ループ内で最小lossを更新したか否かを計算させる部分
        """
        score = -val_loss

        if self.best_score is None:  #1Epoch目の処理
            self.best_score = score   #1Epoch目はそのままベストスコアとして記録する
            self.checkpoint(val_loss, model)  #記録後にモデルを保存してスコア表示する
        elif score < self.best_score:  # ベストスコアを更新できなかった場合
            self.counter += 1   #ストップカウンタを+1
            if self.verbose:  #表示を有効にした場合は経過を表示
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')  #現在のカウンタを表示する 
            if self.counter >= self.patience:  #設定カウントを上回ったらストップフラグをTrueに変更
                torch.save(self.best_model, self.path)
                self.early_stop = True
        else:  #ベストスコアを更新した場合
            self.best_score = score  #ベストスコアを上書き
            self.best_model = model.state_dict()
            self.checkpoint(val_loss, model)  #モデルを保存してスコア表示
            self.counter = 0  #ストップカウンタリセット

    def checkpoint(self, val_loss, model):
        '''ベストスコア更新時に実行されるチェックポイント関数'''
        if self.verbose:  #表示を有効にした場合は、前回のベストスコアからどれだけ更新したか？を表示
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        #torch.save(model.state_dict(), self.path)  #ベストモデルを指定したpathに保存
        self.val_loss_min = val_loss  #その時のlossを記録する


class MLP(nn.Module):
    def __init__(self, input_sizes, hidden_features=200):
        super().__init__()
        self.input_sizes = input_sizes
        print(self.input_sizes)
        self.hidden_features = hidden_features
        self.fc1=(nn.Sequential(
                    nn.Linear(input_sizes[0], hidden_features),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(hidden_features, hidden_features)
                    ))
        self.fc2=(nn.Sequential(
                    nn.Linear(input_sizes[1], hidden_features),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(hidden_features, hidden_features)
                    ))

        self.fc3=(nn.Sequential(
                    nn.Linear(input_sizes[2], hidden_features),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(hidden_features, hidden_features)
                    ))

        self.fc = (nn.Sequential(
                    nn.Linear(hidden_features*1, hidden_features),
                    nn.ReLU(),
                    nn.Linear(hidden_features, 1)
        ))

    def forward(self, x):
        index1 = self.input_sizes[0]
        index2 = self.input_sizes[0] + self.input_sizes[1]
        index3 = self.input_sizes[0] + self.input_sizes[1] + self.input_sizes[2]
        y1 = self.fc1(x[:, :index1])
        #y2 = self.fc2(x[:, index1:index2])
        y3 = self.fc3(x[:, index2:index3])
        #y = torch.concat([y1, y2, y3], axis=1)
        #y = torch.concat([y1, y3], axis=1)
        y = y1
        y = self.fc(y)
        return y

if __name__=='__main__':
    path = Path()
    device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')

    mask = np.load(path.valid_idx_path)
    train_mask = mask[mask>=2]
    val_mask = mask[mask==0]
    test_mask = mask[mask==1]

    df_experience = pd.read_csv(path.df_experience_path)
    image_features = np.load(path.spot_img_emb_path)
    enc = sp.OneHotEncoder( sparse=False )
    tfidf_index = np.load(path.tfidf_topk_index_path).astype(int)
    tfidf_index = enc.fit_transform(pd.DataFrame(tfidf_index))
    print(image_features.shape)
    print(tfidf_index.shape)
    enc = sp.OneHotEncoder( sparse=False )
    place_labels=enc.fit_transform(df_experience[['category_label','pref_label']])
    print(place_labels.shape)
    X = np.concatenate([image_features, tfidf_index, place_labels], axis=1)
    y = (np.log10(df_experience['jalan_review_count']+1)).values

    X = torch.tensor(X).to(device).float()
    y = torch.tensor(y).to(device).float()

    X_train, X_val, X_test = X[train_mask], X[val_mask], X[test_mask]
    y_train, y_val, y_test = y[train_mask], y[val_mask], y[test_mask]

    input_sizes = (image_features.shape[1], tfidf_index.shape[1],place_labels.shape[1])
    
    model = MLP(input_sizes)
    model = model.to(device)
    earlystopping = EarlyStopping(patience=200, verbose=True)

    epoch = 500
    optimizer = Adam(model.parameters(), lr=1e-5)

    for i in range(epoch):
        for phase in ['train','val']:
            if phase=='train':
                model.train()
                out = model(X_train)
                loss = F.mse_loss(out, y_train)
                loss.backward()
                optimizer.step()
            else:
                model.eval()
                out = model(X_val)
                loss = F.mse_loss(out, y_val)
                earlystopping(loss, model) #callメソッド呼び出し
            
            print(f'epoch: {i} phase:{phase}, loss:{loss}')
        if earlystopping.early_stop: #ストップフラグがTrueの場合、breakでforループを抜ける
            print("Early Stopping!")
            
    #model.load_state_dict(torch.load('checkpoint_model.pth'))
    model.eval()
    pred = model(X_test)
    print(model(X_val))
    print(pred.shape, y_test.shape)
    print(y_test.flatten().shape)
    print(np.corrcoef(y_test.cpu().detach().numpy(),pred.flatten().cpu().detach().numpy()))

