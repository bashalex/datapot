from abc import ABCMeta, abstractmethod


class BaseTransformer(metaclass=ABCMeta):

    @staticmethod
    @abstractmethod
    def validate(field, value):
        """
        To Override
        :param field: the name of the field
        :param value: value to check; can be simple object as int, String etc; dict or list
        :return: boolean, whether the value is suitable for the transformer
        """
        pass

    @staticmethod
    @abstractmethod
    def requires_fit():
        """
        To Override
        This method reduce amount of operations because when we fit out transformer
        we use list of values from particular field from all objects in training dataset.
        And it's a quite slow operation to extract this list of values

        :return: bool, whether transformer requires fitting before transforming
        """
        pass

    @abstractmethod
    def fit(self, all_values):
        """
        To Override
        :param all_values: list of particular values from every object in data
        fits itself using given values
        """

    @abstractmethod
    def transform(self, value):
        """
        To Override
        :param value: value to transform
        :return list of generated values
        """
        pass

    @abstractmethod
    def names(self):
        """
        To Override
        :return list of generated features names in the same order as 'transform' method returns them
        """
        pass

    def __str__(self):
        return 'BaseTransformer'

    def __repr__(self):
        return self.__str__()
