from datetime import *
from time import mktime
import time

from dateutil.parser import *

from .base_transformer import BaseTransformer


class BaseTimestampTransformer(BaseTransformer):
    """Base class for timestamp transformers"""

    num_of_examples = 0.
    num_of_valid = 0.

    def validate(self, field, value):
        """Check is the value is timestamp

        If the value is a float or int it can be a unix time form
        If the value is a string, try to use parse method from dateutil package
        to convert string to datetime

        :param field: the name of the field
        :param value: value to check; can be simple object as int,
               String etc; dict or list
        :return: boolean, whether the value is suitable for the transformer
        """
        is_valid_value = False
        self.num_of_examples += 1

        try:
            """Set time interval to detect unixtime value

            Time interval boundaries:
                low: 01 / 01 / 1975 @ 12:00 am(UTC) === 157766400
                high: current time + 20 years (01/01/1990 @ 12:00am (UTC) -
                01/01/1970 @ 12:00am (UTC)) === time.time() + 631152000
            """
            if ((isinstance(value, float) or isinstance(value, int)) and
               157766400 < value < time.time() + 631152000):
                datetime.fromtimestamp(value)
                is_valid_value = True
                self.num_of_valid += 1

            elif isinstance(value, str):
                parse(value)
                is_valid_value = True
                self.num_of_valid += 1
        except:
            is_valid_value = False

        self.confidence = self.num_of_valid / self.num_of_examples

        return is_valid_value


class TimestampTransformer(BaseTimestampTransformer):
    """Timestamp transfomer

    Replaces most known formats to represent a date and/or time with date,
    time and other timestamp info (new_features)
    """

    @staticmethod
    def requires_fit():
        return False

    new_features = ['unixtime',
                    'week_day',
                    'month_day',
                    'hour',
                    'minute']  # TODO: add features (is_holliday, is_weekend)

    def __str__(self):
        return 'TimestampTransformer'

    def __repr__(self):
        return self.__str__()

    def __init__(self, dayfirst=False):
        # TODO: 1484673907123 type + add user/manual flexability (dayfirst=True), select features form the created set (new_features)
        self.dayfirst = dayfirst

    def names(self):
        return ['timestamp_' + feature for feature in self.new_features]

    def fit(self, all_values):
        """Fit transformer

        Each value is transformed independently.
        Fit is not required.
        """
        pass

    def transform(self, value):
        """Each value is transformed to new time features

        Each value is transformed to new_features:
        ['unixtime', 'week_day', 'month_day', 'hour', 'minute']

        # TODO: add features (is_holliday, is_weekend)
        :param value: value to transform
        :return list of generated values
        """

        try:
            if isinstance(value, float) or isinstance(value, int):
                date = datetime.fromtimestamp(value)
            if isinstance(value, str):
                date = parse(value)

            new_features_values = dict()
            new_features_values['unixtime'] = mktime(date.timetuple())
            new_features_values['week_day'] = date.today().weekday()
            new_features_values['month_day'] = date.day
            new_features_values['hour'] = date.hour
            new_features_values['minute'] = date.minute

            return [new_features_values[feature]
                    for feature in self.new_features]
        except:
            return [None for feature in self.new_features]


class TestTimestampTransformer(BaseTransformer):
    """Replaces timestamps with date and time"""

    @staticmethod
    def requires_fit():
        return True

    def __str__(self):
        return 'TestTimestampTransformer'

    def __repr__(self):
        return self.__str__()

    def __init__(self):
        # here could be some specific parameters
        # for this particular transformer
        pass

    def names(self):
        return ['date', 'time']

    def validate(self, field, value):
        if not isinstance(value, int):
            self.confidence = max(self.confidence - 0.1, 0)
            return False
        if value <= 1000000000:
            self.confidence = max(self.confidence - 0.1, 0)
            return False

        self.confidence = min(self.confidence + 0.1, 1)
        return True

    def fit(self, all_values):
        # do nothing
        pass

    def transform(self, value):
        try:
            d = datetime.fromtimestamp(value)
            return [d.date(), d.time()]
        except (OverflowError, OSError):
            return [None, None]
