from __future__ import absolute_import, division, print_function

from abc import ABCMeta, abstractmethod

import numpy as np

INITIAL_CONFIDENCE = 0.5


class BaseTransformer:
    """Base transformer's class

    :param confidence: value from 0 to 1
                       if confidence = 0 the transformer will be removed
                       if confidence is less than 0.7
                       by the end of fitting the transformer
                       will not be used as well
    """

    __metaclass__ = ABCMeta

    confidence = INITIAL_CONFIDENCE

    @abstractmethod
    def validate(self, field, value):
        """To Override

        :param field: the name of the field
        :param value: value to check; can be simple object as int,
        String etc; dict or list
        :return: boolean, whether the value is suitable for the transformer
        """
        pass

    @staticmethod
    @abstractmethod
    def requires_fit():
        """To Override

        This method reduce amount of operations because
        when we fit out transformer we use list of values from particular field
        from all objects in training dataset.
        And it's a quite slow operation to extract this list of values

        :return: bool, whether transformer requires fitting before
                 transforming
        """
        pass

    @abstractmethod
    def fit(self, all_values):
        """ To Override

        :param all_values: list of particular values from every object in data
                           fits itself using given values
        """

    @abstractmethod
    def transform(self, value):
        """ To Override

        :param value: value to transform
        :return list of generated values
        """
        pass

    def transform_batch(self, all_values):
        """

        :param values:  values iterator from particular field in json file
        :return:
        """
        rows = [self.transform(value) for value in all_values]
        if not len(rows):
            return []
        if isinstance(rows[0], (list, tuple, np.ndarray)):
            return rows
        else:
            return np.array(rows).reshape(-1, 1)




    @abstractmethod
    def names(self):
        """ To Override

        :return list of generated features names in the same order
                as 'transform' method returns them
                None, if transformer is not fitted yet and
                do not know how many features it will produce
        """
        pass

    def __str__(self):
        return 'BaseTransformer'

    def __repr__(self):
        return self.__str__()
