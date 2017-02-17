#!/usr/bin/env python
'''
Tinkoff Boosters contest
Common case of a scoring dataset
5 categorical and 7 numerical variables
Use OneHot-encoding representation for categorical variables
Apply SVD compression for features with number of categories >= 10
datapot.remove_transformer to exclude redundant transformations
'''

import sys
import bz2
import time
import xgboost as xgb
from sklearn.model_selection import cross_val_score
import datapot as dp
from datapot.datasets import fetch_tinkoff

def main():
    fetch_tinkoff()
    data = bz2.BZ2File('data/tink.jsonlines.bz2')

    datapot = dp.DataPot()
    t0 = time.time()
    datapot.fit(data)
    print('fit time:', time.time()-t0)
    datapot.remove_transformer('living_region', 0)

    t0 = time.time()
    df = datapot.transform(data, verbose=True)
    print('transform time:', time.time()-t0)

    X = df.drop(['open_account_flg'], axis=1)
    y = df['open_account_flg']
    model = xgb.XGBClassifier()
    print('Cross-val score', cross_val_score(model, X, y, cv=5))

    model.fit(X, y)
    print('Feature importance:')
    print(list(zip(X.columns, model.feature_importances_)))

if __name__ == "__main__":
    main()

