
from __future__ import print_function

import sys
import bz2
import time
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import cross_val_score
import datapot as dp
from datapot.datasets import fetch_job_salary

fetch_job_salary()
data = bz2.BZ2File('data/job.jsonlines.bz2')

datapot = dp.DataPot()
t0 = time.time()
datapot.fit(data)
print('fit time:', time.time()-t0)
t0 = time.time()

try:
    df = datapot.transform(data, verbose=True)
except:
    sys.exit(1)
print('transform time:', time.time()-t0)

X = df.drop(['SalaryNormalized', 'LocationNormalized', 'Id'], axis=1)
y = pd.qcut(df['SalaryNormalized'].values, q=2, labels=[0,1]).ravel()
model = xgb.XGBClassifier()
cv_score = cross_val_score(model, X, y, cv=5)
print('Cross-val score:', cv_score)
try:
    assert all(i > 0.5 for i in cv_score)
except AssertionError:
    sys.exit(1)

model.fit(X, y)
fi = model.feature_importances_
try:
    assert len(fi) > 0
except AssertionError:
    sys.exit(1)

print('Feature importance:')
print(*(list(zip(X.columns, fi))), sep='\n')
