import spacy
import pandas as pd
nlp = spacy.load('ja_ginza_electra')
#df_review = pd.read_csv('/home/yamanishi/project/trip_recommend/data/jalan/review/review_all_period_.csv')
df_review = pd.read_csv('/home/yamanishi/project/trip_recommend/model/recommendation/data/df_review_train.csv')
docs = list(nlp.pipe(list(df_review['review']), disable=['ner']))
with open('/home/yamanishi/project/trip_recommend/data/ginza/docs_train.pkl', 'wb') as f:
    pickle.dump(docs, f)