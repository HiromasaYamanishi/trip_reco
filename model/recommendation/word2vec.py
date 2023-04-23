from janome.tokenizer import Tokenizer
from janome.analyzer import Analyzer
import gensim
from gensim.models import Word2Vec, KeyedVectors
import pandas as pd
from tqdm import tqdm
import MeCab
import numpy as np
from janome.charfilter import RegexReplaceCharFilter, UnicodeNormalizeCharFilter

def tokenize(text):
    t = MeCab.Tagger("")
    t.parse("")
    m = t.parseToNode(text)
    tokens = []
    while m:
        tokenData = m.feature.split(",")
        #token = [m.surface]
        #for data in tokenData:
        #    token.append(data)
        if m.surface not in '、。！':
            tokens.append(m.surface)
        m = m.next
    tokens.pop(0)
    if len(tokens)>0:
        tokens.pop(-1)
    return tokens
if __name__ == '__main__':
    df = pd.read_csv('/home/yamanishi/project/trip_recommend/model/recommendation/data/df_review_train.csv')
    print('loaded df')
    char_filters = [UnicodeNormalizeCharFilter(),
                    RegexReplaceCharFilter('[#!:;<>{}・`.,()-=$/_\d\'"\[\]\|~]+', ' ')]
    
    analyzer = Analyzer(char_filters = char_filters)
    sentences = df['review']
    sentences_tokenized = [tokenize(s) for s in tqdm(sentences)]
    model = gensim.models.Word2Vec.load('/home/yamanishi/project/trip_recommend/data/ja/ja.bin')
    model.min_count=10
    model.build_vocab(sentences_tokenized, update=True)
    total_examples = model.corpus_count
    model.train(sentences_tokenized, total_examples = total_examples, epochs = 500)
    model.save('/home/yamanishi/project/trip_recommend/model/recommendation/data/ja_review.model')