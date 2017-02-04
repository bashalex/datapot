#!/usr/bin/env python
"""
Tinkoff Boosters contest
Common Use case for a scoring dataset
5 categorical and 7 numerical variables
Use OneHot-encoding with SVD compression
"""
import sys
import datapot as dp
from time import time
from datapot.datasets import fetch_tinkoff
import bz2
from sklearn.model_selection import cross_val_score
import xgboost as xgb

fetch_tinkoff()
data = bz2.BZ2File('data/tinkoff.jsonlines.bz2')
datapot = dp.DataPot()
start = time()
datapot.fit(data)
print('fit time:', time()-start)
start = time()
df = datapot.transform(data, verbose=True)
print('transform time:', time()-start)

X = df.drop(['open_account_flg'], axis=1)
y = df['open_account_flg']
model = xgb.XGBClassifier()
print('Cross-val score', cross_val_score(model, X, y, cv=5))
model.fit(X, y)
print('Feature importance:')
print(list(zip(X.columns, model.feature_importances_)))
