import collections

from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import TruncatedSVD

from .base_transformer import BaseTransformer

# TODO: change constant's name
CATEGORICAL_MAX_SIZE = 100
SVD_COMPONENTS = 10
THRESHOLD = 0.8


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

        if (float(self.repeats) /
           (len(self.validate_set) + self.repeats) >= THRESHOLD):
            self.confidence = 1
        else:
            self.confidence = 0.6


class SVDOneHotTransformer(BaseCategoricalTransformer):
    """One-hot encoding + SVD dimension reduction

    One-hot encoding with dimension reduction (SVD)
    in case there are too many features .
    """

    def __init__(self, dimension_reduction=True):
        super().__init__()
        self.apply_dimension_reduction = False
        self.features = dict()
        self.dimension_reduction = dimension_reduction

    def __str__(self):
        return "SVDOneHotTransformer"

    def names(self):
        return ['one_hot' + str(i) for i in range(self._n_components)]

    def fit(self, all_values):
        # TODO: what if value is not hashable
        self.features = dict()
        self.apply_dimension_reduction = False

        for value in all_values:
            if value not in self.features:
                self.features[value] = len(self.features)

        if len(self.features) <= CATEGORICAL_MAX_SIZE:
            self.one_hot_encoder = OneHotEncoder(sparse=False,
                                                 handle_unknown='ignore')
            self._n_components = len(self.features)
        else:
            self.apply_dimension_reduction = True
            self.one_hot_encoder = OneHotEncoder(sparse=True,
                                                 handle_unknown='ignore')
            self._n_components = SVD_COMPONENTS

        self.one_hot_encoder.fit([[self.features[value]]
                                  for value in all_values])

        if self.apply_dimension_reduction:
            self.dim_reducer = TruncatedSVD(n_components=self._n_components)
            self.dim_reducer.fit(
                self.one_hot_encoder.transform([[self.features[x]]
                                                for x in all_values]))

        return self

    def transform(self, value):
        if not isinstance(value, collections.Hashable):
            return None
        value = ([[self.features[value]]]
                 if value in self.features else [[len(self.features)]])
        feature_array = self.one_hot_encoder.transform(value)
        if self.apply_dimension_reduction:
            return self.dim_reducer.transform(feature_array)[0].tolist()
        else:
            return feature_array[0].tolist()


class CountersTransformer(BaseCategoricalTransformer):
    """Counters transformer."""
    pass
