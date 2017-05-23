from .base_transformer import BaseTransformer
import numpy as np
TIME_SERIES_MIN_LENGTH = 15
CONFIDENCE_PENALTY = 0.0


class SSATransformer(BaseTransformer):

    @staticmethod
    def requires_fit():
        return True

    def __str__(self):
        return 'SSA_Decomposition'

    def __repr__(self):
        return self.__str__()

    def __init__(self):
        self.maximum_size = None
        self.window_size = None

    @classmethod
    def _is_numeric(self, obj):
        attrs = [
            '__add__',
            '__sub__',
            '__mul__',
            '__truediv__',
            '__pow__'
        ]
        return all(hasattr(obj, attr) for attr in attrs)


    def names(self):
        return ['ssa_decomposition_' + str(i) for i in range(self.maximum_size)]


    def validate(self, field, value):
        if not isinstance(value, list):
            self.confidence = max(self.confidence - CONFIDENCE_PENALTY, 0)
            return False

        if len(value) < TIME_SERIES_MIN_LENGTH:
            self.confidence = max(self.confidence - CONFIDENCE_PENALTY, 0)
            return False

        for val in value:
            if not SSATransformer._is_numeric(val):
                self.confidence = max(self.confidence - CONFIDENCE_PENALTY, 0)
                return False

        return True

    def fit(self, all_values):
        self.maximum_size = len(max(all_values, key=len))
        self.window_size = self.maximum_size//2
        return self

    def transform(self, value):
        l = self.window_size
        if len(value) < self.maximum_size:
            value.extend([0 for _ in range(self.maximum_size - len(value))])
        n = len(value)

        #construct the trajectory matrix
        X = np.array([value[i:i+l] for i in range(0, n-l)])
        C = X.T*X*1/l
        V, L, VT = np.linalg.svd(C)

        #select the most significant components
        num_of_components = len(L)
        k = num_of_components//2
        k_top_components = np.argsort(L)[num_of_components-k:]
        V_hat = V[k_top_components]
        U_hat = V_hat.dot(X)
        X_R = V_hat*U_hat

        #sum over anti-diagonals to reconstruct the array
        ret = [np.sum(X_R.flatten()[i:i*l+1:l]) for i in range(n)]
        return ret
