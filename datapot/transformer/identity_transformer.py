from .base_transformer import BaseTransformer


class IdentityTransformer(BaseTransformer):
    """Transformer that doesn't transform the data and returns it as is"""

    def __str__(self):
        return 'IdentityTransformer'

    @staticmethod
    def requires_fit():
        return False

    def validate(self, field, value):
        return True

    def names(self):
        return ''

    def fit(self, all_values):
        pass

    def transform(self, value):
        return value
