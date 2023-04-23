# trip_reccommend
POI Popularity Prediction and Explanationtion
Prediction Model: HetSpot
<img width="600" alt="スクリーンショット 2023-04-23 22 57 23" src="https://user-images.githubusercontent.com/72190773/233844074-919f0042-3636-4bec-98e2-a13b7ee000e9.png">

retult of prediction on 42856 Japanese POI data (extracted from jalan.net)

| model  | cor |
| ------------- | ------------- |
| MLP | 0.722  |
| SVR  | 0.807 |
| GCN  | 0.772 |
| HAN  | 0.753 |
| HGT  | 0.765 |
| GraphSAGE  | 0.805 |
| **HetSpot**  | **0.825** |

Explanation Model: LIME

<img width="500" alt="スクリーンショット 2023-04-23 23 01 08" src="https://user-images.githubusercontent.com/72190773/233844428-c7b9b998-7c70-4f97-98b1-f99f22f15de7.png">

Expanation Result Example

<img width="400" alt="スクリーンショット 2023-04-23 23 01 40" src="https://user-images.githubusercontent.com/72190773/233844435-e06ac6c6-5a01-4d11-8292-b71298a2f202.png">

Recommendation Result

| model  | Recall@20 | Coverage@20| Novelty@20 | 
| ------------- | ------------- | ------------- | ------------- |
| HetSpot | 0.196  | **0.571** | **0.818** |
| LightGCN  | **0.233** | 0.252 | 0.783 |
