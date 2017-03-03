**[Datapot](#datapot)** |
**[Usage](#usage)** |
**[Examples](./notebooks/)** |
**[Features](#features)** |
**[Authors](#authors)** 

# Datapot
[![Build Status](https://travis-ci.org/bashalex/datapot.svg?branch=master)](https://travis-ci.org/bashalex/datapot)
*Open source tool for machine learning on semi-structured data that creates numeric object-feature matrix from JSON. 
The idea of Datapot is to make the process of data preparation and feature extraction automatic, easy and effective.*

<img src="data/datapot_feature_extraction.png">


## Usage


**Install Datapot:**
```bash
$ git clone https://github.com/bashalex/datapot.git
$ cd datapot
$ pip install .
```

To **create a Datapot** object simply write the following:

```python
>>> import datapot as dp 
>>> data = dp.DataPot()
```


#### DataPot has two main methods:
- fit()
- transform()

Method `fit(self, data, limit)` goes through the first  N  objects (N = limit), passes the possible features to Transformers. Each Transformer evaluates if a feature from current field or a number of fields can be created. As a result a dict of features  and Transformers is created.

To apply `fit()` to JSON file:
```python
>>> f = open('data/matches_test.jsonlines', 'r')
>>> data.fit(f, limit=100)
>>> data
DataPot class instance
 - number of features without transformation: 806
 - number of new features: 315
features to transform: 
    (u'players.0.gold_t', [ComplexTransformer])
    (u'picks_bans.0.is_pick', [BoolToIntTransformer])
    (u'players.0.kills_log.0.unit', [TfidfTransformer])
    (u'players.1.xp_t', [ComplexTransformer])
    (u'picks_bans.1.is_pick', [BoolToIntTransformer])
    (u'players.1.kills_log.0.unit', [TfidfTransformer])
    ...
```

Method `transform(self, data, verbose)` generates a pandas. DataFrame with new features that were detected on the fit() call. If parameter verbose is true, progress description is printed during the feature extraction.

```python
>>> df = data.transform(f, verbose=False)
fit transformers...OK
num of new features: 315
```


## Examples 

Look for [more examples](./notebooks/) of using Datapot with different datasets and more Transformer specific.




## Features
Datapot provides many ways of extracting features from JSON-s.

Data types that can be processed:
 - Boolean 
 - Numerical array (transform array to their sum divided by average length of array in training set)
 - Time series (сalculate descriptive statistical properties of a given time series)
 - Timestamp  (date, time, day of week, day of month etc.)
 - Text (bag of words tf-idf, word2vec)
 - Categorial (one-hot encoding, dimension reduction)


## Authors

- Alex Bash
- Yuriy Mokriy
- Nikita Savelyev
- Michal Rozenwald
- Peter Romov

Datapot is a course work project of [the Faculty of Computer Science](https://cs.hse.ru/en/) of [the Higher School of Economics](https://www.hse.ru/en/).