from abc import ABCMeta, abstractmethod


class BaseTransformer(metaclass=ABCMeta):

    @staticmethod
    @abstractmethod
    def validate(value) -> bool:
        """
        To Override
        :param value: value to check
        :return: boolean, whether the value is suitable for the transformer
        """
        pass

    @abstractmethod
    def fit(self, data):
        """
        To Override
        :param data: iterable that contains strings representing Json object
                     OR file where every line contains one Json object
        fits itself using given data
        """

    @abstractmethod
    def transform(self, value):
        """
        To Override
        :param value: value to transform
        :return list of generated values
        """
        pass

    @staticmethod
    @abstractmethod
    def names():
        """
        To Override
        :return list of generated features names in the same order as 'transform' method returns them
        """
        pass
