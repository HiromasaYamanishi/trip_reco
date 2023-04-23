from transformers import pipeline
from transformers import AutoModelForSequenceClassification
from transformers import BertJapaneseTokenizer
from transformers import BertJapaneseTokenizer
import time
import pandas as pd


class SAJ:
    def __init__(self):
        self.model = AutoModelForSequenceClassification.from_pretrained('daigo/bert-base-japanese-sentiment')
        self.tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
        self.nlp = pipeline("sentiment-analysis", model = self.model, tokenizer = self.tokenizer)
    
    def sentiment_analysis_japanese(self, TARGET_TEXT, pandas = False):
        result = self.nlp(TARGET_TEXT)
        label = result[0]["label"]
        score = result[0]["score"]

        return label, score

if __name__ == "__main__":
    TARGET_TEXT='メロスは激怒した'
    saj=SAJ()
    s=time.time()
    print(saj.sentiment_analysis_japanese(TARGET_TEXT))
    print(time.time()-s)
