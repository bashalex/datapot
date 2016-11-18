from ..base_transformer import  BaseTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.decomposition import NMF
from time import time
import re
import iso639
import langdetect


class BaseTextTransformer(BaseTransformer):
    """
    Text transformers basic class.
    """
    @staticmethod
    def requires_fit():
        return True

    def __init__(self):
        super().__init__()
        self._params = {}
        self.language = None

    def _detect_language(self, text_feature):
        iso_code = langdetect.detect(' '.join(text_feature[:10]))
        self.language = iso639.languages.get(alpha2=iso_code).name.lower()
        if self.language not in SnowballStemmer.languages:
            self.language = 'other'
        print(self.language)

    @staticmethod
    def validate(field, feature_value):
        return isinstance(feature_value, str) and len(feature_value) > 10

    def _clean_text(self, text):
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

    @staticmethod
    def names():
        return "Tfidf"

    def _detect_parameters(self, text_feature):
        self._params = {'max_features': 5000}
        return self._params

    @property
    def params(self):
        return self._params

    def fit(self, text_feature):
        self._detect_language(text_feature)
        text_feature = self._clean_text_feature(text_feature)
        self.vectorizer.set_params(**self._detect_parameters(text_feature))
        self.vectorizer.fit(text_feature)
        start = time()
        self.nmf = NMF(n_components=12, max_iter=200, init="nndsvd").fit(
            self.vectorizer.transform(text_feature)[:1000])
        print(time() - start)
        return self

    def transform(self, text_feature):
        if isinstance(text_feature, str):
            text_feature = [self._clean_text(text_feature)]
        else:
            text_feature = self._clean_text_feature(text_feature)
        return self.nmf.transform(self.vectorizer.transform(text_feature)).tolist()[0]