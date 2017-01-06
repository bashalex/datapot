from ..base_transformer import  BaseTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.decomposition import NMF
from time import time
import re
import numpy as np
import iso639
import langdetect
import gensim

class BaseTextTransformer(BaseTransformer):
    """
    Text transformers basic class.
    """

    @staticmethod
    def requires_fit():
        return True

    def __init__(self):
        super().__init__()
        self.language = None
        self._params = {}

    def _detect_language(self, text_feature):
        # TODO: Remove the iso639 dependency
        iso_code = langdetect.detect(' '.join(text_feature[:self._detect_number]))
        self.language = iso639.languages.get(alpha2=iso_code).name.lower()
        if self.language not in SnowballStemmer.languages:
            self.language = 'other'
        print(self.language)

    @staticmethod
    def validate(field, feature_value):
        return isinstance(feature_value, str) and len(feature_value) > 10

    def _clean_text(self, text):
        # TODO: Do something with nltk.stopwords
        text = re.sub(r'[^[^\W\d_]]', ' ', text.lower())
        stopwords_set = set(stopwords.words(self.language))
        stem = (lambda x: x) if self.language == 'other' else SnowballStemmer(self.language).stem
        return ' '.join(stem(word) for word in text.split() if word not in stopwords_set)

    def _clean_text_feature(self, text_feature):
        return [self._clean_text(text) for text in text_feature]


class TfidfTransformer(BaseTextTransformer):
    """
    Returns NMF transformation of text's Tfidf representation.
    """

    def __str__(self):
        return "TfidfTransformer"

    def __repr__(self):
        return self.__str__()

    def __init__(self):
        super().__init__()
        self.vectorizer = TfidfVectorizer()
        self._vectorizer_params = {}
        self._nmf_params = {}

    @staticmethod
    def names():
        # TODO: Change to return None
        return list(map(str, range(12)))

    def _detect_parameters(self, text_feature):
        # TODO: Write the parameters autodetection
        self._detect_number = 10
        self._detect_language(text_feature)
        self._vectorizer_params = {'max_features': 5000}
        self._nmf_params = {'n_components' : 12, 'max_iter' : 200, 'init' : 'nndsvd'}
        self._nmf_fit_texts_number = 1000

    def fit(self, text_feature):
        self._detect_parameters(text_feature)
        text_feature = self._clean_text_feature(text_feature)
        self.vectorizer.set_params(**self._vectorizer_params)
        self.vectorizer.fit(text_feature)
        start = time()
        self.nmf = NMF(**self._nmf_params).fit(self.vectorizer.transform(text_feature)[:self._nmf_fit_texts_number])
        print(time() - start)
        return self

    def transform(self, text_feature):
        if isinstance(text_feature, str):
            text_feature = [self._clean_text(text_feature)]
        else:
            text_feature = self._clean_text_feature(text_feature)
        return self.nmf.transform(self.vectorizer.transform(text_feature)).tolist()[0]


class Word2VecTransformer(BaseTextTransformer):
    """
    Returns the average Word2Vec vectors for each text
    """

    def __str__(self):
        return "Word2VecTransformer"

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def names():
        # TODO: Change to return None
        return list(map(str, range(300)))

    def _find_word2vec_model(self):
        self.word2vec_model = gensim.models.Word2Vec.load_word2vec_format('~/GoogleNews-vectors-negative300.bin', binary=True)

    def _detect_parameters(self, text_feature):
        self._detect_number = 10
        self._detect_language(text_feature)
        self._find_word2vec_model()

    def fit(self, text_feature):
        self._detect_parameters(text_feature)
        return self

    def transform(self, text_feature):
        words_list = [self.word2vec_model[word] for word in self._clean_text(text_feature).split()
                      if word in self.word2vec_model]
        #print(np.array(words_list).shape)
        return np.mean(words_list, axis=0)
