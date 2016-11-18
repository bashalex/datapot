from .base_transformer import BaseTransformer
import numpy as np
from tsfresh.feature_extraction.feature_calculators import \
        binned_entropy, abs_energy, kurtosis, mean_abs_change, mean_autocorrelation, \
        skewness, symmetry_looking, count_above_mean, count_below_mean

class TimeSeriesTransformer(BaseTransformer):
    def requires_fit(self):
        return False

    def __str__(self):
        return 'TimeSeriesTransfomer...'

    def __repr__(self):
        return self.__str__()

    def __init__(self):
        super().__init__()

    @classmethod
    def _is_numeric(self, obj):
        attrs = ['__add__', '__sub__', '__mul__', '__truediv__', '__pow__']
        return all(hasattr(obj, attr) for attr in attrs)
    @classmethod
    def _entropy(self, ts):
        #simple way to determine stationarity that doesn't work
        return 0.1
        """
        def _maxdist(x_i, x_j):
            return max([abs(ua - va) for ua, va in zip(x_i, x_j)])
        def _phi(m):
            x = [[ts[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
            C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0) for x_i in x]
            return (N - m + 1.0)**(-1) * sum(np.log(C))
        r = np.mean(ts) #TODO: use rolling mean to find optimal r
        m = 2 #default heuristic
        N = len(ts)
        return abs(_phi(m + 1) - _phi(m))
        """

    @staticmethod
    def names():
        return ['ts_abs_energy', 'ts_kurtosis', 'ts_mean_abs_change', 'ts_mean_autocorrelation', \
        'ts_skewness', 'ts_count_above_mean', 'ts_count_below_mean']

    @staticmethod
    def validate(field, value):
        # check that value is list
        if not isinstance(value, list):
            return False
        if len(value) < 5:
            return False
        for val in value:
            if not TimeSeriesTransformer._is_numeric(val):
                return False
        if TimeSeriesTransformer._entropy(value) > 0.2:
            return False #assume series is way too stochastic
        return True

    def fit():
        pass

    def transform(self, value):
        if value is None:
            return None
        return [abs_energy(value), kurtosis(value), mean_abs_change(value), mean_autocorrelation(value), \
        skewness(value), count_above_mean(value)/len(value), count_below_mean(value)/len(value)]
