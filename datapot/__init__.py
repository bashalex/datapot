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
        self.__num_features_after_transform = None

    def __str__(self):
        res = 'DataPot class instance\n'
        res += ' - number of features without transformation: {}\n'.format(len(self.__fields.keys()))
        res += ' - number of features after transformation: '
        if self.__num_features_after_transform is None:
            res += 'Unknown\n'
        else:
            res += '{}\n'.format(self.__num_features_after_transform)
        res += 'features to transform: \n'
        for x in self.__fields.items():
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
        self.__num_features_after_transform = None
        self.__fields = {}

        decoder = json.JSONDecoder()

        self.__move_pointer_to_start(data)
        for n, obj in enumerate(data):
            # decode string to dictionary
            obj_fields = decoder.decode(obj)
            for name, value in obj_fields.items():
                self.__for_each_field_call(name, value, self.__ask_transformers)
            if n == limit:
                break
        self.__move_pointer_to_start(data)

    def transform(self, data):
        """
        :param data: iterable that contains strings representing Json object
                     OR file where every line contains one Json object
        :return: list of transformed json objects with added features
        """
        # TODO: implement later
        pass

    def extract(self, data) -> pd.DataFrame:
        """
        :important: this method calls both 'fit' and 'transform' methods of transformers
        :param data: iterable that contains strings representing Json object
                     OR file where every line contains one Json object
        :return: generated DataFrame
        """
        decoder = json.JSONDecoder()
        rows = []

        # get all feature names
        names = self.__all_features_names()

        # fit transformers
        self.__fit_transformers(data)

        self.__move_pointer_to_start(data)
        for obj in data:
            obj_fields = decoder.decode(obj)
            row = []
            for _field, _transformers in self.__fields.items():
                new_features = self.__generate_feature(obj_fields, _field, _transformers)
                if isinstance(new_features, list):
                    row += new_features
                else:
                    row.append(new_features)
            rows.append(row)
        self.__move_pointer_to_start(data)

        # save final number of features
        self.__num_features_after_transform = len(rows[0])

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
        for _field, _transformers in self.__fields.items():
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

    def __fit_transformers(self, data):
        """
        fit every transformer with given data
        :param data: data for fitting
        """
        for x in self.__fields.items():
            for i in range(len(x[1])):
                self.__move_pointer_to_start(data)
                x[1][i].fit(data)

    def __generate_feature(self, obj, field, transformers):

        # let's say we always apply only first transformer from the list
        # TODO: change it later
        t = transformers[0] if len(transformers) > 0 else None

        value = self.__extract_value(obj, field.split('.'))

        return value if t is None else t.transform(value)

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

    def __for_each_field_call(self, name, value, func):
        """
        parses json object and calls 'func' for each field
        :param name: name of current field
        :param value: value in the field
        :param func: function to call
        """
        if isinstance(value, list):
            for i, obj in enumerate(value):
                self.__for_each_field_call(name + '.{}'.format(i), obj, func)
        elif isinstance(value, dict):
            for _name, _value in value.items():
                self.__for_each_field_call(name + '.' + _name, _value, func)
        else:
            func(name, value)

    def __ask_transformers(self, name, value):
        """
        searching for for suitable transformers
        :param name: field name
        :param value: value in the field
        """
        if name in self.__fields.keys():
            # value from the same field was already checked
            _transformers = self.__fields.get(name)
            num = 0
            while num < len(_transformers):
                if not _transformers[num].validate(value):
                    del _transformers[num]
                else:
                    num += 1
            return

        suitable_transformers = []
        for _transformer in self.__transformers:
            if _transformer.validate(value):
                suitable_transformers.append(_transformer())
        self.__fields[name] = suitable_transformers
