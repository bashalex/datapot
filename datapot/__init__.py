from __future__ import absolute_import, division, print_function
from future.builtins import (ascii, bytes, chr, dict, filter, hex, input,
                             int, map, next, oct, open, pow, range, round,
                             str, super, zip)
import json
import datapot.transformer
import pandas as pd
import io


# Check dependencies
dependencies = ('pandas', 'numpy')
missing_dependencies = []

for dependency in dependencies:
    try:
        __import__(dependency)
    except ImportError as e:
        missing_dependencies.append(dependency)

if missing_dependencies:
    raise ImportError("Missing required dependencies {}".format(missing_dependencies))


class DataPot:

    def __init__(self):
        # list of all possible encoders
        self.__transformers = transformer.__all__

        # 'field name' -> list of suitable transformers
        # TODO: decide how to choose one of them
        self.__fields = {}
        self.__num_new_features = None

    def __str__(self):
        res = 'DataPot class instance\n'
        res += ' - number of features without transformation: {}\n'.format(len(self.__fields.keys()))
        res += ' - number of new features: '
        if self.__num_new_features is None:
            res += 'Unknown\n'
        else:
            res += '{}\n'.format(self.__num_new_features)
        res += 'features to transform: \n'
        for x in self.__fields.iteritems():
            if len(x[1]) > 0:
                res += '\t{}\n'.format(x)
        return res

    def __repr__(self):
        return self.__str__()

    def fit(self, data, limit=50):
        """
        :param data: iterable that contains strings representing Json object
                     OR file where every line contains one Json object
        :param limit: max number of objects to use for fitting
        """
        # clear params
        self.__num_new_features = None
        self.__fields = {}

        decoder = json.JSONDecoder()

        self.__move_pointer_to_start(data)
        for n, obj in enumerate(data):
            # decode string to dictionary
            obj_fields = decoder.decode(obj)
            for name, value in obj_fields.items():
                self.__parse(name, value)
            if n == limit:
                break
        self.__move_pointer_to_start(data)
        self.__num_new_features = self.__num_of_new_features()

    def transform(self, data, verbose=False):
        """
        :param verbose: if true prints progress
        :important: this method calls both 'fit' and 'transform' methods of transformers
        :param data: iterable that contains strings representing Json object
                     OR file where every line contains one Json object
        :return: generated DataFrame
        """
        decoder = json.JSONDecoder()
        rows = []

        if verbose:
            print('fit transformers...')
        # fit transformers
        self.__fit_transformers(data, verbose)
        if verbose:
            print('fit transformers...OK')

        # get all feature names
        names = self.__all_features_names()

        self.__move_pointer_to_start(data)
        for obj in data:
            obj_fields = decoder.decode(obj)
            row = []
            for _field, _transformers in self.__fields.iteritems():
                new_features = self.__generate_feature(obj_fields, _field, _transformers)
                print("new features:", new_features)
                if isinstance(new_features, list):
                    row += new_features
                else:
                    row.append(new_features)
            rows.append(row)
        self.__move_pointer_to_start(data)
        
        if verbose:
            print("rows:", rows)
        # save final number of features
        self.__num_new_features = len(rows[0])

        if verbose:
            print("num of new features:", self.__num_new_features)

        # convert list to DataFrame
        df = pd.DataFrame(data=rows, columns=names)

        return df

    def __move_pointer_to_start(self, data):
        if isinstance(data, io.TextIOWrapper):
            data.seek(0, 0)  # move pointer to the beginning of the file

    def __all_features_names(self):
        """
        :return: list of all feature names after transformation
        """
        result = []
        for _field, _transformers in self.__fields.iteritems():
            if len(_transformers) == 0:
                result.append(_field)
                continue
            for _transformer in _transformers:
                suffixes = _transformer.names()
                if isinstance(suffixes, list):
                    result += ['{}_{}'.format(_field, suffix) for suffix in suffixes]
                else:
                    result.append('{}_{}'.format(_field, suffixes))
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
                print("fit: {}".format(x))
            values = None
            for i in range(len(x[1])):
                if not x[1][i].requires_fit():
                    continue
                self.__move_pointer_to_start(data)
                if values is None:
                    values = self.__extract_all_values(data, x[0].split('.'))
                x[1][i].fit(values)

    def __generate_feature(self, obj, field, transformers):

        value = self.__extract_value(obj, field.split('.'))

        if len(transformers) == 0:
            return value  # nothing to apply

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
        :return: list of extracted values from the given field from all objects in data
        """
        decoder = json.JSONDecoder()
        result = []
        for obj in data:
            obj_fields = decoder.decode(obj)
            result.append(self.__extract_value(obj_fields, location))
        return result

    def __parse(self, name, value):
        """
        parses json object and associates suitable transformers for fields
        :param name: name of current field
        :param value: value in the field
        """
        final_obj = self.__ask_transformers(name, value)
        if final_obj:
            return  # doesn't make sense to parse further

        if isinstance(value, list):
            for i, obj in enumerate(value):
                self.__parse(name + '.{}'.format(i), obj)
        elif isinstance(value, dict):
            for _name, _value in value.items():
                self.__parse(name + '.' + _name, _value)
        else:
            # no one transformer is suitable for the given field
            # but we can add it to the list of fields
            # in order to leave it in final dataset though without any transformation
            self.__fields[name] = []

    def __ask_transformers(self, name, value):
        """
        searching for suitable transformers
        :param name: field name
        :param value: value in the field
        :return: bool, whether at least one transformer is suitable for the given field
        """
        if name in self.__fields.keys():
            # value from the same field was already checked
            _transformers = self.__fields.get(name)
            num = 0
            while num < len(_transformers):
                if not _transformers[num].validate(name, value):
                    del _transformers[num]
                else:
                    num += 1
            if num == 0 and (isinstance(value, list) or isinstance(value, dict)):
                # remove field at all if it is a complex field (array or json object)
                # it doesn't work correctly in all cases!
                # because result depends on order of objects
                # TODO: change this logic
                self.__fields.pop(name)
            return num > 0  # at least one transformer left in the list

        suitable_transformers = []
        for _transformer in self.__transformers:
            if _transformer.validate(name, value):
                suitable_transformers.append(_transformer())

        # add field to the set if at least one transformer was added
        if len(suitable_transformers) > 0:
            self.__fields[name] = suitable_transformers

        return len(suitable_transformers) > 0  # at least one transformer was added

    def __num_of_new_features(self):
        result = 0
        for _field, _transformers in self.__fields.iteritems():
            for t in _transformers:
                if t.names() is None:
                    return None
                if isinstance(t.names(), str):
                    result += 1
                else:
                    result += len(t.names())
        return result
