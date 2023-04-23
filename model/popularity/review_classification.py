import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchtext
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import BertJapaneseTokenizer

df_review = pd.read_csv('/home/yamanishi/project/trip_recommend/data/jalan/review/review_all_period_.csv')
print(len(df_review))
categories = list(set(df_review['rating']))
categories
id2cat = dict(zip(list(range(len(categories))), categories))
cat2id = dict(zip(categories, list(range(len(categories)))))
print(id2cat)
print(cat2id)
df_review['label'] = df_review['rating'].map(cat2id)

# 念の為シャッフル
review_data = df_review.sample(frac=1).reset_index(drop=True)
review_data = review_data[:int(0.1*len(review_data))]
review_data['text'] =  review_data['review']
# データセットを本文とカテゴリーID列だけにする
review_data = review_data[['text', 'label']]

train_df, valid_df = train_test_split(review_data, train_size=0.8)
valid_df, test_df = train_test_split(valid_df, train_size=0.5)
print(f'train size:{len(train_df)}, valid_size: {len(valid_df)}, test size: {len(test_df)}')

dataset_packed = Dataset.from_pandas(review_data)
dataset_split = dataset_packed.train_test_split(test_size=0.2, seed=0)
print(dataset_split)

tokenizer= AutoTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-v2')

def preprocess_function(examples):
    MAX_LENGTH = 512
    return tokenizer(examples["text"], max_length=MAX_LENGTH, truncation=True)

tokenized_dataset = dataset_split.map(preprocess_function, batched=True)
'''
with open('/home/yamanishi/project/trip_recommend/data/review_classification/tokenized_text.pkl', 'wb') as f:    pickle.dump(tokenized_dataset, f)

with open('/home/yamanishi/project/trip_recommend/data/review_classification/tokenized_text.pkl', 'rb') as f:
    tokenized_dataset = pickle.load(f)

print('loaded tokenized text')
'''
#tokenizer= AutoTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-v2')
tokenizer = BertJapaneseTokenizer("/home/yamanishi/project/trip_recommend/model/results/vocab.txt", word_tokenizer_type='mecab')
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
model = AutoModelForSequenceClassification.from_pretrained("cl-tohoku/bert-base-japanese-v2", num_labels=5)
print('loaded model')

from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {'accuracy':acc, 'f1':f1}


os.environ["CUDA_VISIBLE_DEVICES"] ="1"
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy='epoch',
    logging_strategy='epoch',
    save_strategy='epoch',
    save_total_limit=1,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    no_cuda=False, # GPUを使用する場合はFalse
)

trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test'],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

trainer.save_state()
trainer.save_model()

pred_result = trainer.predict(tokenized_dataset['test'], ignore_keys=['loss', 'last_hidden_state', 'hidden_states'])
pred_label= pred_result.predictions.argmax(axis=1).tolist()
print(pred_label)
from sklearn.metrics import classification_report
print(tokenized_dataset['test']['label'], pred_label, categories)
print(classification_report(tokenized_dataset['test']['label'], pred_label, target_names=categories))
pred_attention = pred_result.attention
print(pred_attention)