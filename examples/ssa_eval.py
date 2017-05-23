#!/usr/bin/env python

import sys
import bz2
import time
import xgboost as xgb
from sklearn.model_selection import cross_val_score
import datapot as dp
from datapot.datasets import load_mallat

data = load_mallat()
datapot = dp.DataPot()
datapot.detect(data)

tr = dp.transformer.ssa_transformer.SSATransformer()
datapot.remove_transformer('class', 0)
datapot.add_transformer('time_series', tr)

t0 = time.time()
datapot.fit(data)
print('fit time:', time.time()-t0)


t0 = time.time()
df = datapot.transform(data, verbose=True)
print('transform time:', time.time()-t0)
X = df.drop(['class'], axis=1)
y = df['class']

model = xgb.XGBClassifier()
cv_score = cross_val_score(model, X, y, cv=5)
assert all(i > 0.5 for i in cv_score), 'Low score!'
print('Cross-val score:', cv_score)
