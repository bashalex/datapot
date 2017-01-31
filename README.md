**[Idea and Description](#idea-and-description)** |
**[Setup](#setup)** |
**[Authors](#authors)** |


# Datapot

*Open source tool for machine learning on semi-structured data.
Datapot creates numeric object-feature matrix from JSON.*

---

# Idea and Description

The idea of Datapot is to make the process of data preparation and feature extraction from semi-structured data for machine learning  automatic, easy and effective.

<img src="data/datapot_feature extraction.png" width="224">

We created DataPot that takes JSON as an input with fields of object descriptions. It generates a numeric structured feature matrix. 
New features are extracted using  **[Transformers](#transformers)**. Each of the Transformers detects specific types of fields (during the **[fit()](#Transformers)**  call) and converts the values to new numeric features  (during the  **[transform()](#Transformers)**  call).

#### To create a Datapot object simply write the following:

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

# Transformers:
 - Boolean Transformer
 - Complex Transformer
 - Timeseries Transformer
 - Timestamp Transformer
 -  Text Transformers: 
        -- Tfidf Transformer 
        -- Word2Vec Transformer
 - Categorial Transformer:
        -- SVDOneHotTransformer
 

##### Transformers description
 - Boolean Transformer 
    Replaces 'False' and 'True' with zeros and ones

 - Complex Transformer 
 Transform array of ints to their sum divided  average length of array in training set

 - Timeseries Transformer

 - Timestamp Transformer
    Replaces most known formats to represent a date and/or time with date, time and other timestamp info 

 - Tfidf Transformer
    Returns NMF transformation of text's Tfidf representation.

 - Word2Vec Transformer
    Returns the average Word2Vec vectors for each text

 - SVD One Hot Transformer
    One-hot encoding with dimension reduction (SVD) in case there are too many features .


#### Examples 
More Examples of using Datapot with different datasets and more Transformer specific can be seen in https://github.com/bashalex/datapot/tree/master/notebooks

## Setup

### Check prerequisites

+ Ubuntu 14.04 or older (or OS X)
+ Python 2.7 or later

Required Python Packages:
- pandas
- numpy
- sklearn
- iso639
- langdetect
- gensim
- nltk
- tsfresh
- dateutil

<!--If you have Python 2 >=2.7.9 or Python 3 >=3.4 installed from python.org, you will already have pip and setuptools, but will need to upgrade to the latest version:-->
<!--On Linux or OS X:-->
<!--```bash-->
<!--pip install -U pip setuptools-->
<!--```-->


##### To install the required Python Packages:

```bash
$ pip install -U pip setuptools
$ pip install pandas numpy sklearn iso639 langdetect gensim nltk tsfresh python-dateutil
```


### Install Datapot

To install Datapot run
```bash
$ git clone https://github.com/bashalex/datapot.git
$ cd datapot
$ python setup.py install
```

<!--### Use datapot from command line-->

<!--```bash-->
<!--datapot --file={input video file path}-->
<!--```-->

<!--To find out more about datapot usage-->

<!--```bash-->
<!--datapot --help-->
<!--```-->

# Authors

Alex Bash (alexey@etherionlab.com)
Yuriy Mokriy (yurymokriy@gmail.com)
Nikita Savelyev (n.a.savelyev@gmail.com) 
Michal Rozenwald (michal.rozenwald@gmail.com)

Boss: Peter Romov


---
---

