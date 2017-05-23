import collections

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import OneHotEncoder

from .base_transformer import BaseTransformer

# TODO: change constant's name
CATEGORICAL_FEATURES_MAX_NUMBER = 100
SVD_COMPONENTS = 10
REPEATS_RATE_TO_CHOOSE_THRESHOLD = 0.8
CONFIDENT = 1
NONCONFIDENT = 0.6


class BaseCategoricalTransformer(BaseTransformer):
    """Base class for categorical transformers"""

    def __init__(self):
        self.validate_set = set()
        self.repeats = 0
        self._n_components = 0

    @staticmethod
    def requires_fit():
        return True

    def validate(self, field, value):

        if not isinstance(value, collections.Hashable):
            return

        if value in self.validate_set:
            self.repeats += 1
        else:
            self.validate_set.add(value)

        repeats_rate = float(self.repeats) / (len(self.validate_set) + self.repeats)

        if repeats_rate >= REPEATS_RATE_TO_CHOOSE_THRESHOLD:
            self.confidence = CONFIDENT
        else:
            self.confidence = NONCONFIDENT


class SVDOneHotTransformer(BaseCategoricalTransformer):
    """One-hot encoding + SVD dimension reduction

    One-hot encoding with dimension reduction (SVD)
    in case there are too many features .
    """

    def __init__(self, dimension_reduction=True):
        super(SVDOneHotTransformer, self).__init__()
        self.apply_dimension_reduction = False
        self.features = dict()
        self.dimension_reduction = dimension_reduction

    def __str__(self):
        return 'SVDOneHotTransformer'

    def names(self):
        if self.apply_dimension_reduction:
            return ['one_hot' + str(i) for i in range(self._n_components)]
        else:
            names = [''] * len(self.features)
            for category, number in self.features.items():
                names[number] = str(category)
            return names


    def fit(self, all_values):
        # TODO: what if value is not hashable
        self.features = dict()
        self.apply_dimension_reduction = False

        for value in all_values:
            if value not in self.features:
                self.features[value] = len(self.features)

        if len(self.features) <= CATEGORICAL_FEATURES_MAX_NUMBER:
            self.one_hot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            self._n_components = len(self.features)
        else:
            self.apply_dimension_reduction = True
            self.one_hot_encoder = OneHotEncoder(sparse=True, handle_unknown='ignore')
            self._n_components = SVD_COMPONENTS

        self.one_hot_encoder.fit([[self.features[value]] for value in all_values])

        if self.apply_dimension_reduction:
            self.dim_reducer = TruncatedSVD(n_components=self._n_components)
            numeric_values = [[self.features[x]] for x in all_values]
            encoded_values = self.one_hot_encoder.transform(numeric_values)
            self.dim_reducer.fit(encoded_values)

        return self

    def transform(self, value):
        if not isinstance(value, collections.Hashable):
            return None
        value = [[self.features[value]]] if value in self.features else [[len(self.features)]]
        feature_array = self.one_hot_encoder.transform(value)
        if self.apply_dimension_reduction:
            return self.dim_reducer.transform(feature_array)[0].tolist()
        else:
            return feature_array[0].tolist()

    def transform_batch(self, all_values):
        def is_encoded(value):
            if not isinstance(value, collections.Hashable) or value not in self.features:
                return False
            return True
        all_values = [[self.features[value] if is_encoded(value) else len(self.features)]
                      for value in all_values]
        feature_array = self.one_hot_encoder.transform(all_values)
        if self.apply_dimension_reduction:
            return self.dim_reducer.transform(feature_array)
        else:
            return feature_array



class CountersTransformer(BaseCategoricalTransformer):
    """Counters transformer."""
    pass
