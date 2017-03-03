from sklearn.model_selection import cross_val_score
import xgboost as xgb

import datapot as dp

dummy_data = [
    '{"name": "Gilbert", "wins": [3, 4, 12], "rating": 32}',
    '{"name": "Alexa", "wins": [1, 2, 5, 7], "rating": 24}',
    '{"name": "May", "wins": [], "rating": 1240}',
    '{"name": "Deloise", "wins": [6, 8, 9, 10, 11], "rating": 25}',
]

# create DataPot instance
data = dp.DataPot()
print(data)

# fit it with data
data.fit(dummy_data)
print(data)
print(data.fields())

# apply transformers
df = data.transform(dummy_data, drop_non_numerical=True)
print(df)

# we are going to predict rating
y = df['rating_as_is']
X = df.drop('rating_as_is', axis=1)

# evaluate prediction score using xgboost
model = xgb.XGBRegressor()
print(cross_val_score(model, X, y, cv=2))
