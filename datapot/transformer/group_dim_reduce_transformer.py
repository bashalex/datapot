from .base_transformer import BaseTransformer
import numpy as np
from sklearn.decomposition import TruncatedSVD

class BaseGroupDimReduceTransformer(BaseTransformer):
    """
    Base class for Group Dimension Reduction Transformer transformers
    Extract from JSON field with a list or a dict of NUMERIC values representation in reduced dimension
    """

    def __init__(self, field_name):
        self.field_name = field_name
        self.features = list()
        # self.new_dimension_
        # self.valid_number = 0.
        # self.all_number = 0.

    def validate(self, field, value):
        """
        Custom method: validation is not required
        """
        pass


class SVDGroupDimReduceTransformer(BaseGroupDimReduceTransformer):
    """
    Group Dimension Reduction Transformer transformers
    Extract from JSON field with a list or a dict of values representation in reduced dimension
    Using Truncated SVD for linear dimensionality reduction
    TruncatedSVD from sklearn: http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
    """

    @staticmethod
    def requires_fit():
        return True

    def __str__(self):
        return 'SVDGroupDimReduceTransformer'

    def __repr__(self):
        return self.__str__()

    def __init__(self, n_components, group_type='None'):
        self._n_components = n_components
        self.group_type = group_type

    def names(self):
        return [feature for feature in self.new_features]

    def get_features(self, all_values):
        '''
        Get names of features
        By the first element decide if the it's a group of lists or group of dicts
        :param all_values:
        :return:
        '''
        if not len(all_values):  # empty set of values
            raise ValueError("ERROR: SVDGroupDimReduceTransformer - Empty list of values")
        # features = list()
        if self.type == 'dict':
            features_set = set().union(*all_values)  # Union of all keys from a list of dictionaries
            features = list(features_set)

        elif self.type == 'list':
            all_lens = np.array([len(value) for value in all_values])
            count_lens = np.bincount(all_lens)  # count freq
            max_freq_len = np.argmax(count_lens)
            features = [str(feature_num) for feature_num in range(max_freq_len)]
        else:
            raise ValueError("ERROR: SVDGroupDimReduceTransformer - Group is not a dict or a list")

        return features

    def take_values_add_missings(self, value_list):
        # add missing_value to the end of short lists
        num_list = value_list[:self.features_len]
        if len(num_list) < self.features_len:
            num_list.extend([self.missing_value for i in range(self.features_len - len(num_list))])
        return num_list

    def find_group_type(self, all_values):
    # TODO: take fist not empty value instead of all_values[0]
        if isinstance(all_values[0], dict):
            type = 'dict'
        elif isinstance(all_values[0], list):
            type = 'list'
        else:
            raise ValueError("ERROR: SVDGroupDimReduceTransformer - Group is not a dict or a list")
        return type

    def fit(self, all_values, get_features_limit=100, missing_value=0):
        '''
        Fit transformer
        Creat from all dicts/lists matrix of all features and reduce the dimensionality
        :param all_values: dicts or lists (group) with same group elements, exp.:[ {'data': 2, 'pot': 0}, ... , {'data': 1, 'pot': 7} ]

        :return:
        '''
        self.missing_value = missing_value

        self.type = self.find_group_type(all_values)

        self.features = self.get_features(all_values[:get_features_limit])
        self.features_len = len(self.features)

        self.dim_reducer = TruncatedSVD(n_components=self._n_components)

        numeric_values = list()
        if self.type == 'dict':
            numeric_values = [[value_dict.get(key, missing_value) for key in self.features] for value_dict in all_values]

        elif self.type == 'list':
            # add missing_value to the end of short lists
            numeric_values = [self.take_values_add_missings(value_list) for value_list in all_values]

        self.dim_reducer.fit(numeric_values)
        self.new_features = ['SVD_' + str(i) for i in range(self._n_components)]
        return self

    def transform(self, value):
        if  value == None:
            numeric_val = [self.missing_value for i in self.features_len]

        elif self.type == 'dict':
            numeric_val = [value.get(key, self.missing_value) for key in self.features]

        elif self.type == 'list':
            numeric_val = self.take_values_add_missings(value)

        return self.dim_reducer.transform(numeric_val)[0].tolist()

    def transform_batch(self, all_values):
        def to_numeric_list(value):
            if value == None:
                numeric_val = [self.missing_value for i in self.features_len]
            elif self.type == 'dict':
                numeric_val = [value.get(key, self.missing_value) for key in self.features]
            elif self.type == 'list':
                numeric_val = self.take_values_add_missings(value)
            return numeric_val

        all_values = [to_numeric_list(value) for value in all_values]
        return self.dim_reducer.transform(all_values)


# class PCAGroupDimensionalityReductionTransformer(BaseGroupDimensionReductionTransformer):
#     """
#     Group Dimension Reduction Transformer transformers
#     Extract from JSON field with a list or a dict of values representation in reduced dimension
#     Using Principal component analysis (PCA)  for linear dimensionality reduction
#     """