import time
from time import mktime
from datetime import *
from dateutil.parser import *

from .base_transformer import BaseTransformer

YEAR_1975 = 157766400
TWENTY_YEARS = 631152000


class BaseTimestampTransformer(BaseTransformer):
    """Base class for timestamp transformers"""

    def __init__(self):
        self.valid_number = 0.
        self.all_number = 0.


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
        try:
            """Set time interval to detect unixtime value

            Time interval boundaries:
                low: 01 / 01 / 1975 @ 12:00 am(UTC) === 157766400
                high: current time + 20 years (01/01/1990 @ 12:00am (UTC) -
                01/01/1970 @ 12:00am (UTC)) === time.time() + 631152000
            """
            if isinstance(value, (float, int)) and YEAR_1975 < value < time.time() + TWENTY_YEARS:
                datetime.fromtimestamp(value)
                self.valid_number += 1
            elif isinstance(value, str):
                parse(value)
                self.valid_number += 1
        except:
            # current value is not detected as timestamp
            pass
        self.all_number += 1
        self.confidence = self.valid_number / self.all_number


class TimestampTransformer(BaseTimestampTransformer):
    """Timestamp transfomer

    Replaces most known formats to represent a date and/or time with date,
    time and other timestamp info (new_features)
    """

    @staticmethod
    def requires_fit():
        return False

    new_features = [
        'unixtime',
        'week_day',
        'month_day',
        'hour',
        'minute'
    ]  # TODO: add features (is_holliday, is_weekend)

    def __str__(self):
        return 'TimestampTransformer'

    def __repr__(self):
        return self.__str__()

    def __init__(self, dayfirst=False):
        self.valid_number = 0.
        self.all_number = 0.
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

            new_features_values = {
                'unixtime': mktime(date.timetuple()),
                'week_day': date.today().weekday(),
                'month_day': date.day,
                'hour': date.hour,
                'minute': date.minute
            }

            return [new_features_values[feature] for feature in self.new_features]
        except:
            return [None for feature in self.new_features]
