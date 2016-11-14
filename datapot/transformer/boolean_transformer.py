from .base_transformer import BaseTransformer


class TestBoolTransformer(BaseTransformer):
    """
    Replaces 'False' and 'True' with zeros and ones
    """

    def __str__(self):
        return 'TestBoolToIntTransformer'

    def __repr__(self):
        return self.__str__()

    def __init__(self):
        super().__init__()
        # here could be some specific parameters for this particular transformer

    @staticmethod
    def names():
        return 'binary'

    @staticmethod
    def validate(value) -> bool:
        return isinstance(value, bool)

    def fit(self, data):
        # do nothing
        pass

    def transform(self, value):
        if isinstance(value, bool):
            return 1 if value else 0
        return None
