**[Datapot](#idea-and-description)** |
**[Usage](#setup)** |
**[Examples](./notebooks/)** |
**[Features](#setup)** |
**[Authors](#authors)** |

# Datapot
---
*Open source tool for machine learning on semi-structured data.
Datapot creates numeric object-feature matrix from JSON.*


The idea of Datapot is to make the process of data preparation and feature extraction from semi-structured data for machine learning  automatic, easy and effective.

<img src="data/datapot_feature_extraction.png">


## Usage
---

### Install Datapot:
```bash
$ git clone https://github.com/bashalex/datapot.git
$ cd datapot
$ pip install .
```

To create a Datapot object simply write the following:

```bash
import datapot as dp 
data = dp.DataPot()
```


### DataPot has two main methods:
- fit()
- transform()

####  fit()
Method **fit**(self, data, limit) goes through the first  N  objects (N = limit), passes the possible features to Transformers. Each Transformer evaluates if a feature from current field or a number of fields can be created. As a result a dict of features  and Transformers is created.

To apply fit() to JSON file:
```bash
f = open('data/matches_test.jsonlines', 'r')
data.fit(f, limit=100)
data
```

Output:
```bash
DataPot class instance
 - number of features without transformation: 806
 - number of new features: 315
features to transform: 
    (u'players.0.gold_t', [TestComplexTransformer(average_len_of_array=None)])
    (u'players.9.gold_t', [TestComplexTransformer(average_len_of_array=None)])
    (u'picks_bans.13.is_pick', [TestBoolToIntTransformer])
    (u'picks_bans.5.is_pick', [TestBoolToIntTransformer])
    (u'players.5.lh_t', [TestComplexTransformer(average_len_of_array=None)])
    (u'players.0.xp_t', [TestComplexTransformer(average_len_of_array=None)])
    (u'players.3.kills_log.0.unit', [TfidfTransformer])
```


####  transform()
Method **transform**(self, data, verbose) generates a pandas.DataFrame with new features that were detected on the fit() call. If parameter verbose is true, progress description is printed during the feature extraction.

```bash
df = data.transform(f, verbose=False)
```
Output:
```bash
fit transformers...OK
num of new features: 315
```


### Examples 
---
Look for [more examples](./notebooks/) of using Datapot with different datasets and more Transformer specific.




## Features
---
 - Boolean Transformer (Replaces 'False' and 'True' with zeros and ones)
 - Complex Transformer (Transform array of ints to their sum divided by average length of array in training set)
 - Timeseries Transformer (Calculate descriptive statistical properties of a given time series)
 - Timestamp Transformer (Extract date, time, day of week, day of monat etc.)
 -  Text Transformers: 
        -- Tfidf Transformer (Returns NMF transformation of text's Tfidf representation)
<!-- Rename Transformer as NMFTfidfTransformer ? => no comments? -->
        -- Word2Vec Transformer <!--(Returns the average Word2Vec vectors for each text)-->
 - Categorial Transformer:
         - SVD One-Hot Transformer <!--(One-hot encoding with dimension reduction (SVD) in case there are too many features)-->


## Authors
---
- Alex Bash
- Yuriy Mokriy
- Nikita Savelyev
- Michal Rozenwald
- Peter Romov

Datapot is a course work project of the Faculty of Computer Science of the Higher School of Economics.