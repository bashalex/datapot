#!/usr/bin/env python
'''
Bag of Words Meets Bags of Popcorn
Usage example for unstructured textual bzip2-compressed data
datapot.fit method subsamples the data to detect language and choose corresponding stopwords and stemming.
For each review datapot.transform generates an SVD-compressed 12-dimensional tfidf-vector representation.
'''
from __future__ import print_function

import sys
import bz2
import time
import xgboost as xgb
from sklearn.model_selection import cross_val_score
import datapot as dp
from datapot.datasets import fetch_imdb

fetch_imdb()
data = bz2.BZ2File('data/imdb.jsonlines.bz2')

datapot = dp.DataPot()
t0 = time.time()
datapot.fit(data)
print('fit time:', time.time()-t0)
datapot.remove_transformer('sentiment', 0)

t0 = time.time()
df = datapot.transform(data, verbose=True)
print('transform time:', time.time()-t0)

X = df.drop(['sentiment', 'id'], axis=1)
y = df['sentiment']
model = xgb.XGBClassifier()
cv_score = cross_val_score(model, X, y, cv=5)
assert all(i > 0.5 for i in cv_score)

print('Cross-val score:', cv_score)

model.fit(X, y)
fi = model.feature_importances_
assert len(fi) > 0

print('Feature importance:')
print(*(list(zip(X.columns, fi))), sep='\n')
