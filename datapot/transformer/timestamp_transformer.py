from .base_transformer import BaseTransformer
from datetime import datetime


class TestTimestampTransformer(BaseTransformer):
    """
    Replaces timestamps with date and time
    """

    @staticmethod
    def requires_fit():
        return True

    def __str__(self):
        return 'TestTimestampTransformer'

    def __repr__(self):
        return self.__str__()

    def __init__(self):
        # here could be some specific parameters for this particular transformer
        pass

    def names(self):
        return ['date', 'time']

    @staticmethod
    def validate(field, value):
        return isinstance(value, int) and value > 1000000000

    def fit(self, all_values):
        # do nothing
        pass

    def transform(self, value):
        try:
            d = datetime.fromtimestamp(value)
            return [d.date(), d.time()]
        except (OverflowError, OSError):
            return [None, None]
