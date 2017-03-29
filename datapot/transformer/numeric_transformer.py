import re
from six import string_types

import fastnumbers
import numpy as np

from .base_transformer import BaseTransformer

VALIDATE_SMOOTHNESS_CONSTANT = 4.


class NumericTransformer(BaseTransformer):
    """Transformers extracts numbers from numbers/strings"""

    def __init__(self):
        self.valid_number = 0.
        self.all_number = 0.

    def __str__(self):
        return 'NumericTransformer'

    @staticmethod
    def requires_fit():
        return False

    def names(self):
        return ''

    def validate(self, field, value):
        if isinstance(value, string_types):
            value = re.sub(',', '.', value)
        if fastnumbers.isreal(value):
            self.valid_number += 1
        self.all_number += 1
        smooth_valid_number = self.valid_number + VALIDATE_SMOOTHNESS_CONSTANT
        smooth_all_number = self.all_number + VALIDATE_SMOOTHNESS_CONSTANT
        self.confidence = smooth_valid_number / smooth_all_number

    def fit(self, all_values):
        pass

    def transform(self, value):
        if isinstance(value, string_types):
            value = re.sub(',', '.', value)
        try:
            return fastnumbers.fast_real(value, np.nan, raise_on_invalid=False)
        except TypeError:
            return np.nan

