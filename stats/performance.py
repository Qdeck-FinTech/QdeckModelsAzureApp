import datetime
import numpy as np
from dateutil.relativedelta import relativedelta
from pandas import Series
from typing import Tuple

from stats.enums import Frequency


NUM_DAYS_IN_YEAR = 250


def get_annual_factor(daily_returns: Series, frequency: Frequency) -> int:
    """
    Gets annual factor
    :param daily_returns :type Series: daily returns
    :param frequency :type Frequency: daily returns frequency
    :returns :type int: daily returns annual factor
    """
    if frequency == Frequency.daily:
        if daily_returns.empty:
            return NUM_DAYS_IN_YEAR
        number_of_years = (daily_returns.index[-1].year - daily_returns.index[0].year) + 1
        average_points = daily_returns.size / number_of_years
        return NUM_DAYS_IN_YEAR if average_points <= 350 else 365
    elif frequency == Frequency.monthly:
        return 12
    elif frequency == Frequency.annually:
        return 1
    else:
        raise ValueError('Unable to calculate performance.  Unsupported frequency {}.'.format(frequency.name))


def calc_performance(daily_returns: Series, frequency: Frequency) \
        -> Tuple[float, float, float, float, float, float, float]:
    """
    Calculates performance
    :param daily_returns :type Series: daily returns
    :param frequency :type Frequency: daily returns frequency
    :returns :type tuple: ytd, one_year, two_years, three_years, five_years, ten_years, twenty_years performance
    """
    if daily_returns.empty:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    utc_now = datetime.datetime.now(datetime.UTC)

    try:
        ytd = round(daily_returns[str(utc_now.year)].sum() * 100, 4)
    except KeyError:
        ytd = np.nan

    if frequency == Frequency.daily:
        end_date = daily_returns.index.max()
        one_year = round(daily_returns[end_date-relativedelta(years=1):].sum() * 100, 4)
        two_years = round(daily_returns[end_date-relativedelta(years=2):].sum() * 100, 4)
        three_years = round(daily_returns[end_date-relativedelta(years=3):].sum() * 100, 4)
        five_years = round(daily_returns[end_date-relativedelta(years=5):].sum() * 100, 4)
        ten_years = round(daily_returns[end_date-relativedelta(years=10):].sum() * 100, 4)
        twenty_years = round(daily_returns[end_date-relativedelta(years=20):].sum() * 100, 4)
    elif frequency == Frequency.monthly:
        one_year = round(daily_returns.iloc[-12:].sum() * 100, 4)
        two_years = round(daily_returns.iloc[-24:].sum() * 100, 4)
        three_years = round(daily_returns.iloc[-36:].sum() * 100, 4)
        five_years = round(daily_returns.iloc[-60:].sum() * 100, 4)
        ten_years = round(daily_returns.iloc[-120:].sum() * 100, 4)
        twenty_years = round(daily_returns.iloc[-240:].sum() * 100, 4)
    elif frequency == Frequency.annually:
        one_year = round(daily_returns.iloc[-1:].sum() * 100, 4)
        two_years = round(daily_returns.iloc[-2:].sum() * 100, 4)
        three_years = round(daily_returns.iloc[-3:].sum() * 100, 4)
        five_years = round(daily_returns.iloc[-5:].sum() * 100, 4)
        ten_years = round(daily_returns.iloc[-10:].sum() * 100, 4)
        twenty_years = round(daily_returns.iloc[-20:].sum() * 100, 4)
    else:
        raise ValueError('Unable to calculate performance.  Unsupported frequency {}.'.format(frequency.name))
    return ytd, one_year, two_years, three_years, five_years, ten_years, twenty_years
