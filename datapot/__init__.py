from __future__ import absolute_import, division, print_function

import bz2
import io
import json
import numpy as np
import pandas as pd
from six import string_types
from future.builtins import *

from . import transformer
from .transformer.base_transformer import BaseTransformer
from .iterators import create_full_iterator

CONFIDENCE_LEVEL_TO_ACCEPT = 0.7
CONFIDENCE_LEVEL_TO_BEGIN = 0.5
TOO_MANY_VALUES_IN_DICT_TO_PARSE = 100


class DataPot:

    def __init__(self):
        # list of all possible encoders
        self.__transformers = transformer.__all__

        # 'field name' -> list of suitable transformers
        # TODO: decide how to choose one of them
        self.__fields = {}
        self.__num_new_features = None

    def __str__(self):
        res = (
            'DataPot class instance\n'
            ' - number of features without transformation: {n_old_features}\n'
            ' - number of new features: {n_new_features}\n'
            'features to transform: \n'
        ).format(
            n_old_features=len(self.__fields),
            n_new_features='Unknown' if self.__num_new_features is None else self.__num_new_features
        )

        for x in self.__fields.items():
            if len(x[1]) > 0:
                res += '\t{}\n'.format(x)
        return res

    def __repr__(self):
        return self.__str__()

    def fields(self):
        return list(self.__fields.keys())

    def add_transformer(self, field_name, transformer):
        if not isinstance(transformer, BaseTransformer):
            raise TypeError('second argument must be an instance of transformer')
        if field_name not in self.__fields:
            raise KeyError('field with the given name doesn\'t exist')
        self.__fields[field_name].append(transformer)
        return self

    def remove_transformer(self, field_name, transformer_index):
        if field_name not in self.__fields:
            raise KeyError('field with the given name doesn\'t exist')
        transformers = self.__fields[field_name]
        if transformer_index >= len(transformers):
            raise IndexError('transformer with given index doesn\'t exist')
        del transformers[transformer_index]
        return self

    def detect(self, data, limit=50):
        """
        :param data: iterable that contains strings representing Json object
                     OR file where every line contains one Json object
        :param limit: max number of objects to use for fitting
        """
        # clear params
        self.__num_new_features = None
        self.__fields = {}

        iterator = create_full_iterator(data)

        for iteration, obj in enumerate(iterator(data)):
            for name, value in obj.items():
                self.__parse(name, value)
            if iteration == limit:
                break

        for _field, _transformers in self.__fields.items():
            accepted_transformers_number = 0
            while accepted_transformers_number < len(_transformers):
                if _transformers[accepted_transformers_number].confidence < CONFIDENCE_LEVEL_TO_ACCEPT:
                    del _transformers[accepted_transformers_number]
                else:
                    accepted_transformers_number += 1

        self.__num_new_features = self.__num_of_new_features()

        return self

    def fit(self, data, verbose=False):
        """
        :param verbose: if true prints progress
        :param data: iterable that contains strings representing Json object
                     OR file where every line contains one Json object
        :return: fitted instance
        """

        if verbose:
            print('fit transformers...')
        # fit transformers
        self.__fit_transformers(data, verbose)
        if verbose:
            print('fit transformers...OK')

        self.__num_new_features = self.__num_of_new_features()

        if verbose:
            print('num of new features:', self.__num_new_features)

        return self

    def transform(self, data):
        decoder = json.JSONDecoder()
        columns = []

        for _field, _transformers in self.__fields.items():
            if not len(_transformers):
                continue
            all_values = self.__extract_all_values(data, _field.split('.'))
            for _transformer in _transformers:
                columns.append(_transformer.transform_batch(all_values))

        # get all feature names
        names = self.__all_features_names()

        # convert list to DataFrame
        return pd.DataFrame(data=np.hstack(columns), columns=names).convert_objects(convert_numeric=True)

    def fit_transform(self, data, verbose=False):
        return self.fit(data, verbose=verbose).transform(data)

    def __all_features_names(self):
        """
        :return: list of all feature names after transformation
        """
        def format_field_name(field, suffix):
            # we may have some problem with unicode non-ascii characters
            if suffix == '':
                return str(field)
            return '{}_{}'.format(field, suffix)

        result = []
        for _field, _transformers in self.__fields.items():
            for _transformer in _transformers:
                suffixes = _transformer.names()
                if isinstance(suffixes, list):
                    result += [format_field_name(_field, suffix) for suffix in suffixes]
                else:
                    result.append(format_field_name(_field, suffixes))
        return result

    def __fit_transformers(self, data, verbose):
        """
        fit every transformer with given data
        :param data: data for fitting
        """
        for x in self.__fields.items():
            if len(x[1]) == 0:
                continue
            if verbose:
                print('fit: {}'.format(x))
            values = None
            for i in range(len(x[1])):
                if not x[1][i].requires_fit():
                    continue
                if values is None:
                    values = self.__extract_all_values(data, x[0].split('.'))
                x[1][i].fit(values)

    def __generate_feature(self, obj, field, transformers):

        value = self.__extract_value(obj, field.split('.'))

        if len(transformers) == 0:
            return None  # nothing to apply

        res = []

        # apply all available transformers
        for t in transformers:
            r = t.transform(value)
            if isinstance(r, list):
                res += r
            else:
                res.append(r)
        return res

    def __extract_value(self, obj, location):
        """
        :param obj: json object
        :param location: location of the field
        :return: extracted value from the given field
        """
        for loc in location:
            if isinstance(obj, dict) and loc in obj:
                obj = obj.get(loc)
                continue
            try:
                obj = obj[int(loc)]
            except (ValueError, IndexError, TypeError):
                return None
        return obj

    def __extract_all_values(self, data, location):
        """
        :param data: iterable that contains strings representing Json object
                     OR file where every line contains one Json object
        :param location: location of the field
        :return: list of extracted values from the given field
                 from all objects in data
        """
        result = []
        iterator = create_full_iterator(data)
        for obj in iterator(data):
            result.append(self.__extract_value(obj, location))
        return result

    def __parse(self, name, value):
        """
        parses json object and associates suitable transformers for fields
        :param name: name of current field
        :param value: value in the field
        """
        final_obj = self.__ask_transformers(name, value)

        if isinstance(value, dict) and len(value) < TOO_MANY_VALUES_IN_DICT_TO_PARSE:
            for _name, _value in value.items():
                self.__parse(name + '.' + _name, _value)
        elif not final_obj:
            # no one transformer is suitable for the given field
            # but we can add it to the list of fields
            # in order to leave it in final dataset though
            # without any transformation
            self.__fields[name] = []

    def __ask_transformers(self, name, value):
        """
        searching for suitable transformers
        :param name: field name
        :param value: value in the field
        :return: bool, whether at least one transformer is suitable
                 for the given field
        """
        if name in self.__fields.keys():
            # value from the same field was already checked
            _transformers = self.__fields.get(name)
            accepted_transformers = 0
            while accepted_transformers < len(_transformers):
                _transformers[accepted_transformers].validate(name, value)
                if _transformers[accepted_transformers].confidence == 0:
                    del _transformers[accepted_transformers]
                else:
                    accepted_transformers += 1

            if accepted_transformers == 0 and isinstance(value, (list, dict)):
                # remove field at all if it is a complex field
                # (array or json object)
                del self.__fields[name]

            return accepted_transformers > 0  # at least one transformer left in the list

        suitable_transformers = []
        for _transformer in self.__transformers:
            t = _transformer()
            t.validate(name, value)
            if t.confidence > CONFIDENCE_LEVEL_TO_BEGIN:
                suitable_transformers.append(t)

        # add field to the set if at least one transformer was added
        if len(suitable_transformers) > 0:
            self.__fields[name] = suitable_transformers

        return len(suitable_transformers) > 0  # at least one transformer was added

    def __num_of_new_features(self):
        result = 0
        for _field, _transformers in self.__fields.items():
            for t in _transformers:
                if t.names() is None:
                    return None
                if isinstance(t.names(), string_types):
                    result += 1
                else:
                    result += len(t.names())
        return result
