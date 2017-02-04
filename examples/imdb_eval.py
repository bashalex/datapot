#!/usr/bin/env python
"""
Bag of Words Meets Bags of Popcorn
Usage example for unstructured textual bzip2-compressed data
datapot.fit method subsamples the data to detect language and choose corresponding stopwords and stemming.
For each review datapot.transform generates an SVD-compressed 12-dimensional tfidf-vector representation.
"""

import sys
import datapot as dp
from datapot.datasets import fetch_imdb
from time import time
import bz2
from sklearn.model_selection import cross_val_score
import xgboost as xgb

if __name__ == "__main__":
    main()

def main():
    fetch_imdb()
    data = bz2.BZ2File('data/imdb.jsonlines.bz2')
    datapot = dp.DataPot()
    start = time()
    datapot.fit(data)
    print('fit time:', time()-start)
    start = time()
    df = datapot.transform(data, verbose=True)
    print('transform time:', time()-start)
    X = df.drop(['sentiment', 'id'], axis=1)
    y = df['sentiment']
    model = xgb.XGBClassifier()
    print('Cross-val score:', cross_val_score(model, X, y, cv=5))
    model.fit(X, y)
    print('Feature importance:')
    print(list(zip(X.columns, model.feature_importances_)))
    return 0
