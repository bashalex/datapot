from .base_transformer import BaseTransformer


VALIDATE_SMOOTHNESS_CONSTANT = 0.1


class BoolTransformer(BaseTransformer):
    """Replaces 'False' and 'True' with zeros and ones"""

    @staticmethod
    def requires_fit():
        return False

    def __str__(self):
        return 'BoolToIntTransformer'

    def __repr__(self):
        return self.__str__()

    def __init__(self):
        self.valid_number = 0.
        self.all_number = 0.

    def names(self):
        return 'binary'


    def validate(self, field, value):
        is_valid_value = False
        if isinstance(value, bool):
            is_valid_value = True
            self.valid_number += 1
        self.all_number += 1
        smooth_valid_number = self.valid_number + VALIDATE_SMOOTHNESS_CONSTANT
        smooth_all_number = self.all_number + VALIDATE_SMOOTHNESS_CONSTANT
        self.confidence = smooth_valid_number / smooth_all_number
        return is_valid_value

    def fit(self, all_values):
        # do nothing
        pass

    def transform(self, value):
        if isinstance(value, bool):
            return 1 if value else 0
        return None
