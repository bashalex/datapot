from .base_transformer import BaseTransformer


class TestBoolTransformer(BaseTransformer):
    """
    Replaces 'False' and 'True' with zeros and ones
    """

    @staticmethod
    def requires_fit():
        return False

    def __str__(self):
        return 'TestBoolToIntTransformer'

    def __repr__(self):
        return self.__str__()

    def __init__(self):
        # here could be some specific parameters for this particular transformer
        pass

    def names(self):
        return 'binary'

    def validate(self, field, value):
        if not isinstance(value, bool):
            self.confidence = max(self.confidence - 0.1, 0)
            return False

        self.confidence = min(self.confidence + 0.1, 1)
        return True

    def fit(self, all_values):
        # do nothing
        pass

    def transform(self, value):
        if isinstance(value, bool):
            return 1 if value else 0
        return None
