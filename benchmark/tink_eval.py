#!/usr/bin/env python
import sys
sys.path.append('..')
import datapot as dp
from time import time

data = open('tink.jsonlines', 'r')
datapot = dp.DataPot()
start = time()
datapot.fit(data)
print('fit time:', time()-start)
start = time()
df = datapot.transform(data, verbose=True)
print('transform time:', time()-start)

X = df.drop(['open_account_flg'], axis=1)
y = df['open_account_flg']
from sklearn.model_selection import cross_val_score
import xgboost as xgb
model = xgb.XGBClassifier()
print(cross_val_score(model, X, y, cv=5))
model.fit(X, y)
print(list(zip(X.columns, model.feature_importances_)))
