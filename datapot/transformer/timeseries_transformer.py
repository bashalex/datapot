from tsfresh.feature_extraction.feature_calculators import (
    abs_energy,
    binned_entropy,
    count_above_mean,
    count_below_mean,
    kurtosis,
    mean_abs_change,
    mean_autocorrelation,
    skewness,
    symmetry_looking
)

from .base_transformer import BaseTransformer

TIME_SERIES_MIN_LENGTH = 5
CONFIDENCE_PENALTY = 0.1
MAX_ENTROPY = 0.2


class TimeSeriesTransformer(BaseTransformer):

    @staticmethod
    def requires_fit():
        return False

    def __str__(self):
        return 'TimeSeriesTransformer...'

    def __repr__(self):
        return self.__str__()

    def __init__(self):
        # here could be some specific parameters
        # for this particular transformer
        pass

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

    @classmethod
    def _entropy(self, ts):
        # simple way to determine stationarity that doesn't work
        return 0.1
        """ This doesn't work now
        def _maxdist(x_i, x_j):
            return max([abs(ua - va) for ua, va in zip(x_i, x_j)])
        def _phi(m):
            x = [[ts[j] for j in range(i, i + m - 1 + 1)]
                 for i in range(N - m + 1)]
            C = [(len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) /
                 (N - m + 1.0)) for x_i in x]
            return (N - m + 1.0)**(-1) * sum(np.log(C))
        r = np.mean(ts) #TODO: use rolling mean to find optimal r
        m = 2 #default heuristic
        N = len(ts)
        return abs(_phi(m + 1) - _phi(m))
        """

    def names(self):
        return [
            'ts_abs_energy',
            'ts_kurtosis',
            'ts_mean_abs_change',
            'ts_mean_autocorrelation',
            'ts_skewness',
            'ts_count_above_mean',
            'ts_count_below_mean'
        ]

    def validate(self, field, value):
        # TODO: change logic with confidence
        # check that value is list
        if not isinstance(value, list):
            self.confidence = max(self.confidence - CONFIDENCE_PENALTY, 0)
            return False

        if len(value) < TIME_SERIES_MIN_LENGTH:
            self.confidence = max(self.confidence - CONFIDENCE_PENALTY, 0)
            return False

        for val in value:
            if not TimeSeriesTransformer._is_numeric(val):
                self.confidence = max(self.confidence - CONFIDENCE_PENALTY, 0)
                return False

        if TimeSeriesTransformer._entropy(value) > MAX_ENTROPY:
            return False  # assume series is way too stochastic

        return True

    def fit(self, all_values):
        pass

    def transform(self, value):
        if value is None:
            return None
        # TODO: remove try-except and validate value in order to avoid exception
        try:
            return [
                abs_energy(value),
                kurtosis(value),
                mean_abs_change(value),
                mean_autocorrelation(value),
                skewness(value),
                count_above_mean(value)/len(value),
                count_below_mean(value)/len(value)
            ]
        except:
            return None
