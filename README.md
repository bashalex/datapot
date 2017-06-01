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



**Install Datapot**

Using `pip`:

```bash
$ pip install datapot
```

Or clone Datapot repo:

```bash
$ git clone https://github.com/bashalex/datapot.git
$ cd datapot
$ pip install .
```

To **create a Datapot** object simply write the following:

```python
>>> import datapot as dp 
>>> datapot = dp.DataPot()
```


#### DataPot has two main methods:
- detect()
- fit()
- transform()

Method `detect(data, limit)` goes through the first N  objects (N = limit), passes the possible features to Transformers. Each Transformer evaluates if a feature from current field or a number of fields can be created. As a result a dict of features and Transformers is created. Method  `fit(data)` trains the detected Transformers on the given set if it is required. 

To apply `fit(data)` to JSON file:
```python
>>> data = open('datapot/data/job.jsonlines', 'r')
>>> datapot.detect(data, limit=100)
>>> datapot.fit(data)
DataPot class instance
 - number of features without transformation: 9
 - number of new features: 82
features to transform: 
	('Id', [NumericTransformer])
	('FullDescription', [TfidfTransformer])
	('ContractType', [SVDOneHotTransformer])
	('ContractTime', [SVDOneHotTransformer])
	('Company', [SVDOneHotTransformer])
	('Category', [SVDOneHotTransformer])
	('SalaryNormalized', [NumericTransformer])

```

Method `transform(data)` generates a pandas. DataFrame with new features that were detected and trained on the detect() and fit() calls.

```python
>>> df = datapot.transform(data)
num of new features: 82
```


## Examples 

Look for [more examples](./notebooks/) of using Datapot with different datasets and more Transformer specific.




## Features
Datapot provides many ways of extracting features from JSON-s.

Data types that can be processed:
 - Boolean 
 - Numerical
 - Numerical array (transform array to their sum divided by average length of array in training set)
 - Time series (сalculate descriptive statistical properties of a given time series)
 - Timestamp  (date, time, day of week, day of month etc.)
 - Text (bag of words tf-idf, word2vec)
 - Categorial (one-hot encoding, dimension reduction)
 
 Manually selected fields:
 - Identity 
 - Group Dimensionality Reduce (change the dimensionality of features in the same JSON field) 


## Authors

- Alex Bash
- Yuriy Mokriy
- Nikita Savelyev
- Michal Rozenwald
- Peter Romov

Datapot is a course work project of [the Faculty of Computer Science](https://cs.hse.ru/en/) of [the Higher School of Economics](https://www.hse.ru/en/).