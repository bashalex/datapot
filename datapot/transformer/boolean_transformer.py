from .base_transformer import BaseTransformer


class TestBoolTransformer(BaseTransformer):
    """
    Replaces 'False' and 'True' with zeros and ones
    """

    def requires_fit(self):
        return False

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
    def validate(field, value):
        return isinstance(value, bool)

    def fit(self, all_values):
        # do nothing
        pass

    def transform(self, value):
        if isinstance(value, bool):
            return 1 if value else 0
        return None
