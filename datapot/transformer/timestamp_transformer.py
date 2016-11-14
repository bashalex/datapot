from .base_transformer import BaseTransformer
from datetime import datetime


class TestTimestampTransformer(BaseTransformer):
    """
    Replaces timestamps with date and time
    """

    def __str__(self):
        return 'TestTimestampTransformer'

    def __repr__(self):
        return self.__str__()

    def __init__(self):
        super().__init__()
        # here could be some specific parameters for this particular transformer

    @staticmethod
    def names():
        return ['date', 'time']

    @staticmethod
    def validate(value) -> bool:
        return isinstance(value, int) and value > 1000000000

    def fit(self, data):
        # do nothing
        pass

    def transform(self, value):
        try:
            d = datetime.fromtimestamp(value)
            return [d.date(), d.time()]
        except (OverflowError, OSError):
            return [None, None]
