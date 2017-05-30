import re

import iso639
import langdetect
import numpy as np

from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
from langdetect.lang_detect_exception import LangDetectException
from future.builtins import map, range, str
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from Stemmer import Stemmer
from six import string_types
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from time import time

from .base_transformer import BaseTransformer

NONE_TEXT = 'NoneNoneNone'
WORD_2_VEC_SIZE = 100
MAX_DICTIONARY_SIZE = 5000
N_COMPONENTS = 12
LANGUAGE_DETECTION_EXAMPLES = 10
NORMAL_TEXT_MIN_SIZE = 10
NMF_FIT_NUMBER = 1000
NON_STRING_PENALTY = 0.15
SMALL_STRING_LENGHT_PENALTY = 0.1
STRING_IS_TEXT_REWARD = 0.1
NMF_ITERS = 200


class BaseTextTransformer(BaseTransformer):
    """Text transformers basic class."""

    alpha_numeric_regexp = re.compile(r'[^[^\W\d_]]')
    space_symbol = re.compile('\s')

    def __init__(self):
        self.language = None
        self._params = {}

    @staticmethod
    def requires_fit():
        return True

    def _detect_language(self, text_feature):
        # TODO: Remove the iso639 dependency
        try:
            iso_code = langdetect.detect(' '.join(text_feature[:self._detect_number]))
            self.language = iso639.languages.get(alpha2=iso_code).name.lower()
        except LangDetectException:
            self.language = 'other'

        if self.language not in SnowballStemmer.languages:
            self.language = 'other'
            self.stopwords_set = set()
        else:
            self.stopwords_set = set(stopwords.words(self.language))

    def validate(self, field, feature_value):
        # TODO: change this logic, just an example
        if not isinstance(feature_value, string_types):
            self.confidence = max(self.confidence - NON_STRING_PENALTY, 0)
            return False

        if len(feature_value) <= NORMAL_TEXT_MIN_SIZE or len(self.space_symbol.findall(feature_value)) <= 2:
            self.confidence = max(self.confidence - SMALL_STRING_LENGHT_PENALTY, 0)
            return False

        self.confidence = min(self.confidence + STRING_IS_TEXT_REWARD, 1)
        return True

    def _clean_text(self, text):
        if not isinstance(text, string_types):
            return NONE_TEXT

        text = self.alpha_numeric_regexp.sub(' ', text.lower())
        return text

    def _stemming(self, text):
        self.stem = (lambda x: x) if self.language == 'other' else SnowballStemmer(self.language).stem
        return ' '.join(self.stem(word) for word in text.split() if word not in self.stopwords_set)

    def _clean_text_feature(self, text_feature):
        return [self._clean_text(text) for text in text_feature]


class TfidfTransformer(BaseTextTransformer):
    """Returns NMF transformation of text's Tfidf representation."""

    def __str__(self):
        return 'TfidfTransformer'

    def __repr__(self):
        return self.__str__()

    def __init__(self):
        super(TfidfTransformer, self).__init__()
        self.vectorizer = TfidfVectorizer()
        self._vectorizer_params = {}
        self._nmf_params = {}
        self.apply_dimension_reduction = False

    def names(self):
        if not self._nmf_params:
            return None
        return list(map(str, range(self._nmf_params['n_components'])))

    def _detect_parameters(self, text_feature):
        # TODO: Write the parameters autodetection
        self._detect_number = LANGUAGE_DETECTION_EXAMPLES
        self._detect_language(text_feature)
        self._vectorizer_params = dict(
            max_features=MAX_DICTIONARY_SIZE,
        )
        self._nmf_params = dict(
            n_components=N_COMPONENTS,
            max_iter=NMF_ITERS,
            init='nndsvd',
        )
        self._nmf_fit_number = NMF_FIT_NUMBER

    def fit(self, text_feature):
        start = time()
        text_feature = self._clean_text_feature(text_feature)
        self._detect_parameters(text_feature)
        #text_feature = [self._stemming(text) for text in text_feature]
        self.vectorizer.set_params(**self._vectorizer_params)
        self.vectorizer.fit(text_feature)
        data_to_nmf_fit = self.vectorizer.transform(text_feature[:self._nmf_fit_number])
        self._nmf_params['n_components'] = min(self._nmf_params['n_components'], data_to_nmf_fit.shape[1])
        self.nmf = NMF(**self._nmf_params).fit(data_to_nmf_fit)
        return self

    def transform(self, text_feature):
        text_feature = [self._clean_text(text_feature)]
        vectorized_feature = self.vectorizer.transform(text_feature)
        return self.nmf.transform(vectorized_feature).tolist()[0]

    def transform_batch(self, all_values):
        texts = self._clean_text_feature(all_values)
        vectorized_feature = self.vectorizer.transform(texts)
        return self.nmf.transform(vectorized_feature)



class Word2VecTransformer(BaseTextTransformer):
    """ Returns the average Word2Vec vectors for each text """

    def __init__(self, word2vec_model_dir=None):
        """Word2Vec transformer

        :param word2vec_model_dir: if it is not None - the pretrained model will be used,
                                   otherwise it will be trained during the fit procedure
        """

        self.language = None
        self._params = {}
        self.model_dir = word2vec_model_dir
        self.vector_size = None

    def __str__(self):
        return 'Word2VecTransformer'

    def __repr__(self):
        return self.__str__()

    def names(self):
        if self.vector_size is None:
            return None
        return list(map(lambda i: 'w2c_{}'.format(i), range(self.vector_size)))

    def _get_word2vec_model(self, text_feature):
        if self.model_dir is None:
            from nltk.tokenize import sent_tokenize

            sentences = [
                self._clean_text(s).split()
                for text in text_feature for s in sent_tokenize(text, language=self.language)
            ]
            self.word2vec_model = Word2Vec(sentences, size=WORD_2_VEC_SIZE)
        else:
            self.word2vec_model = KeyedVectors.load_word2vec_format(self.model_dir, binary=True)

    def _detect_parameters(self, text_feature):
        self._detect_number = LANGUAGE_DETECTION_EXAMPLES
        self._detect_language(text_feature)
        self._get_word2vec_model(text_feature)
        print(type(self.word2vec_model))
        print(vars(self.word2vec_model))
        print(dir(self.word2vec_model))
        self.vector_size = self.word2vec_model.syn0.shape[1]

    def fit(self, text_feature):
        self._detect_parameters(text_feature)
        return self

    def transform(self, text_feature):
        all_words = self._clean_text(text_feature).split()
        words_list = [self.word2vec_model[word] for word in all_words if word in self.word2vec_model]
        return np.mean(words_list, axis=0)
