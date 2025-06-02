import numexpr as ne
import numpy as np
from pandas import concat, Grouper, Series, DataFrame
from typing import Optional, Tuple, Union

from stats.performance import NUM_DAYS_IN_YEAR

try:
    from numba import njit
except ImportError:
    print('Warning: numba not found, severe performance degredation is expected if you try to evaluate stuff.  ' 
          '(OK for rcmserver branch)')

    def njit(cache=None, nogil=False):
        def wrap(f):
            def wrapped_f(*args):
                f(*args)

            return wrapped_f

        return wrap


def calc_sharpe_ratio(daily_returns: Series, risk_free_rate: Optional[float] = 0,
                      annual_factor: Optional[int] = NUM_DAYS_IN_YEAR, compounded: Optional[bool] = True) -> float:
    """
    Calculates sharpe ratio from returns

    :param daily_returns:
    :param risk_free_rate :type float: average yield of 1 year treasury over period
    :param annual_factor :type int: annual factor
    :param compounded :type bool:
    :return: scalar value
    """
    ann_ret = calc_annualized_return(daily_returns, annual_factor, compounded)
    volatility = calc_volatility(daily_returns, annual_factor)

    try:
        return float((ann_ret - risk_free_rate) / volatility)
    except ZeroDivisionError:
        return np.nan


def calc_annualized_return(daily_returns: Series, annual_factor: Optional[int] = NUM_DAYS_IN_YEAR,
                           compounded: Optional[bool] = True) -> float:
    """
    Calculates average annual returns
    :param daily_returns:
    :param annual_factor :type int: annual factor
    :param compounded :type bool: (Optional) compounded flag
    :return: annualized return
    """
    if daily_returns.empty:
        return np.nan

    if compounded:
        annual_returns = (np.int32(1) + daily_returns).resample('YE').prod() - np.int32(1)
        number_of_years = len(daily_returns) / annual_factor
        return float(annual_returns.sum().iloc[0] / number_of_years)
    else:
        return float(daily_returns.mean().iloc[0] * annual_factor)

def calc_turnover(traded_notional: Series, portfolio_notional: float) -> float:
    """
    Calculates average daily turnover given notional traded
    :param traded_notional: amount of notional traded
    :param portfolio_notional :type float: portfolio notional
    :return: :type float average daily turnover
    """

    daily_traded_notional = traded_notional.abs().sum(axis=1)
    try:
        return float(daily_traded_notional.mean() / (2.0 * portfolio_notional))
    except ZeroDivisionError:
        return np.nan

def calc_profit_cps(pnl_local: DataFrame, shares_traded: DataFrame) -> float:
    """
    Calculates profit in cents per share

    :param pnl_local: 2D matrix
    :param shares_traded: 2D matrix
    :return: profit in cents per share
    """

    try:
        return np.sum(pnl_local.values) / np.sum(np.abs(shares_traded.values)) * 100.0
    except ZeroDivisionError:
        return np.nan


def calc_profit_bps(daily_pnl: Series, portfolio_size_usd: float, traded_notional: DataFrame) -> float:
    """
    Calculate profit in basis points

    :param daily_pnl: 2D
    :param portfolio_size_usd: constant
    :param traded_notional: 2D
    :return: profit in basis points
    """

    # % Profit per traded notional (in bps)
    # pnl_bps = sum(daily_pnl)*M / sum(traded_notional(:));

    try:
        return np.sum(daily_pnl.values) * portfolio_size_usd / np.sum(np.abs(traded_notional.values)) * 10000.0
    except ZeroDivisionError:
        return np.nan


def calc_volatility(daily_returns: Series, annual_factor: Optional[int] = NUM_DAYS_IN_YEAR) -> float:
    """
    Calculate volatility

    :param daily_returns :type Series: daily returns
    :param annual_factor :type int: annual factor
    :return :type float volatility
    """
    return float(daily_returns.std().iloc[0] * np.sqrt(annual_factor))


def calc_draw_down(daily_returns: Series, compounded: Optional[bool] = True) -> Series:
    """
    Calculate draw down

    :param daily_returns :type Series: daily returns
    :param compounded :type bool: (Optional) compounded flag
    :return :type Series: draw down
    """
    if compounded:
        cum_factor = (daily_returns+1).cumprod()
        peaks = np.maximum(1, cum_factor.cummax())  # 1 to account for edge case where daily returns are always negative
        return np.maximum((peaks - cum_factor)/peaks, 0)
    cs = daily_returns.cumsum()
    return np.maximum(cs.cummax()-cs, 0)


def calc_max_draw_down(daily_returns: Series, compounded: Optional[bool] = True) -> float:
    """
    Calculates max draw down

    :param daily_returns :type Series: daily returns
    :param compounded :type bool: (Optional) compounded flag
    :return :type float: max draw down
    """
    return float(calc_draw_down(daily_returns, compounded).max().iloc[0])


def calc_min_draw_down(daily_returns: Series, compounded: Optional[bool] = True) -> float:
    """
    Calculates min draw down

    :param daily_returns :type Series: daily returns
    :param compounded :type bool: (Optional) compounded flag
    :return :type float: min draw down
    """
    return float(calc_draw_down(daily_returns, compounded).min().iloc[0])


def calc_current_draw_down(daily_returns: Series, compounded: Optional[bool] = True) -> float:
    """
    Calculates current draw down

    :param daily_returns :type Series: daily returns
    :param compounded :type bool: (Optional) compounded flag
    :return :type float: current draw down
    """
    if daily_returns.empty:
        return np.nan

    dd = calc_draw_down(daily_returns, compounded).iloc[0]
    return float(dd.iloc[0])


def calc_draw_down_recovery(daily_returns: Series, compounded: Optional[bool] = True) -> float:
    """
    Calculate draw down recovery
    :param daily_returns :type Series: daily returns
    :param compounded :type bool: (Optional) compounded flag
    :return :type float: draw down recovery
    """
    if daily_returns.empty:
        return np.nan

    draw_down = calc_draw_down(daily_returns, compounded) * -1
    if not draw_down.empty:
        current_draw_down = draw_down.iloc[-1]
        if current_draw_down < 0:
            positive = draw_down[draw_down >= 0]
            last_date = draw_down.index[0] if positive.empty else positive.index[-1]
            min_current_draw_down = draw_down[last_date:].min()
            return float((min_current_draw_down - current_draw_down) / min_current_draw_down)
    return 1


def calc_std_d(daily_returns: Series, annual_factor: Optional[int] = NUM_DAYS_IN_YEAR) -> float:
    """
    Calculate daily standard deviation

    :param daily_returns :type Series: daily returns
    :param annual_factor :type int: annual factor
    :return :type float: daily standard deviation
    """
    return float(100 * daily_returns.std().iloc[0] * np.sqrt(annual_factor))


def calc_std_m(daily_returns: Series, compounded: Optional[bool] = True) -> float:
    """
    Calculate monthly standard deviation

    :param daily_returns:
    :param compounded: type bool: (Optional) compounded
    :return: annualized return
    """
    if compounded:
        monthly_returns = (np.int32(1) + daily_returns).resample('ME').prod() - np.int32(1)
        return float(100 * monthly_returns.std().iloc[0] * np.sqrt(12))
    return float(100 * daily_returns.resample('ME').sum().std().iloc[0] * np.sqrt(12))


def calc_semi_deviation(daily_returns: Series, annual_factor: Optional[int] = NUM_DAYS_IN_YEAR,
                        target: Optional[float] = 0):
    """
    [Not used]
    Calculates semi deviation
    :param daily_returns :type Series: daily returns
    :param annual_factor :type float: (Optional) annual factor
    :param target :type float: (Optional) target
    :return :type float: semi deviation
    """
    return float(daily_returns[ne.evaluate('daily_returns <= target')].std() * np.sqrt(annual_factor))

def calc_downside_deviation(daily_returns: Series, annual_factor: Optional[int] = NUM_DAYS_IN_YEAR,
                            target: Optional[float] = 0):
    """
    Calculates semi deviation
    :param daily_returns :type Series: daily returns
    :param annual_factor :type float: (Optional) annual factor
    :param target :type float: (Optional) target  (likely always zero)
    :return :type float: semi deviation
    """

    excess_returns = daily_returns - target
    sum_of_squares = sum(excess_returns[excess_returns < target] ** 2)
    try:
        return float(np.sqrt(sum_of_squares / len(daily_returns)) * np.sqrt(annual_factor))
    except ZeroDivisionError:
        return np.nan

def calc_sortino_ratio(daily_returns: Series, risk_free_rate: Optional[float] = 0,
                       annual_factor: Optional[int] = NUM_DAYS_IN_YEAR, compounded: Optional[bool] = True) -> float:
    """
    Calculates sortino ratio from the daily returns
    :param daily_returns :type Series: daily returns
    :param risk_free_rate :type float: (Optional) average yield of 1 year treasury over period
    :param annual_factor :type float: (Optional) annual factor
    :param compounded: type bool: (Optional) compounded flag
    :return :type float: sortino ratio
    """

    #semi_deviation = calc_semi_deviation(daily_returns, annual_factor)  # Incorrect, as it is subtracting the mean which should not happen
    downside_deviation = calc_downside_deviation(daily_returns, annual_factor)
    ann_ret = calc_annualized_return(daily_returns, annual_factor, compounded)  # Defaulted to compounded returns

    try:
        return float((ann_ret - risk_free_rate) / downside_deviation)
    except ZeroDivisionError:
        return np.nan


def calc_inter_quantile_range(daily_returns: Series) -> float:
    """
    Calculates inter quantile range from the daily returns
    :param daily_returns :type Series: daily returns
    :return :type float: IQR
    """
    return float(daily_returns.quantile(.75) - daily_returns.quantile(.25))


def calc_percentile(daily_returns: Series, value: Optional[float] = None) -> float:
    """
    Calculates percentage of points less than a given value
    :param daily_returns :type Series: daily returns
    :param value :type float: (Optional) value to filter on
    :return :type float: percentage of points
    """
    try:
        return float(100 * daily_returns[ne.evaluate('daily_returns < value')].count() / daily_returns.size)
    except ZeroDivisionError:
        return np.nan


def calc_sterling_ratio(daily_returns: Series, risk_free_rate: Optional[float] = 0,
                        annual_factor: Optional[int] = NUM_DAYS_IN_YEAR, compounded: Optional[bool] = True) -> float:
    """
    Calculates sterling ratio
    :param daily_returns :type Series: daily returns
    :param risk_free_rate :type float: average yield of 1 year treasury over period
    :param annual_factor :type int: annual factor
    :param compounded :type bool: (Optional) compounded flag
    :return :type float: sterling ratio
    """
    draw_down = calc_max_draw_down(daily_returns, compounded)
    ann_ret = calc_annualized_return(daily_returns, annual_factor, compounded)

    try:
        return float((ann_ret - risk_free_rate)/ draw_down)
    except ZeroDivisionError:
        return np.nan


@njit(nogil=True)
def _deepest_draw_downs(daily_returns, output):

    current = 0
    peak = 0
    start_index = -1
    current_draw_down = -1
    for i in range(daily_returns.size):
        current += daily_returns[i]
        peak = max(peak, current)
        draw_down = current - peak

        if start_index >= 0:
            if draw_down == 0:
                start_index = -1
                peak = 0
                current = 0
                output[i] = current_draw_down
            else:
                current_draw_down = min(current_draw_down, draw_down)
        else:
            if draw_down != 0:
                start_index = i
                current_draw_down = draw_down

    if current_draw_down != 0 and output[-1] == 0:
        output[-1] = current_draw_down

def calc_average_sterling_ratio(daily_returns: Series, count: Optional[int] = 5,
                                annual_factor: Optional[int] = NUM_DAYS_IN_YEAR) -> float:
    """
    Calculates sterling ratio
    :param daily_returns :type Series: daily returns
    :param count :type int: draw down count
    :param annual_factor :type int: annual factor
    :return :type float: average sterling ratio
    """
    deepest_draw_downs = np.zeros(daily_returns.shape, dtype=np.float32)
    _deepest_draw_downs(daily_returns.values, deepest_draw_downs)
    draw_down_count = min(count, deepest_draw_downs[deepest_draw_downs < 0].size)

    average_draw_down = np.mean(np.sort(deepest_draw_downs)[:draw_down_count]) if draw_down_count > 0 else 0
    return np.nan if average_draw_down == 0 else \
        float(calc_annualized_return(daily_returns, annual_factor) / -average_draw_down)


def calc_percent_winning_ratio(daily_returns: Series) -> float:
    """
    Calculates percent winning ratio
    :param daily_returns :type Series: daily returns
    :return :type float: percent winning ratio
    """
    try:
        return float(daily_returns[ne.evaluate('daily_returns > 0')].count() / daily_returns.size)
    except ZeroDivisionError:
        return np.nan


def calc_average_rolling_standard_deviation(daily_returns: Series, window_size: int) -> float:
    """
    Calculates average rolling standard deviation
    :param daily_returns :type Series: daily returns
    :param window_size :type int: rolling window size
    :return :type float: percent winning ratio
    """
    return float(daily_returns.rolling(window_size).std().mean())


def calc_mv_av(daily_returns: Series, window_size: Optional[int] = 20) -> float:
    """
    Calculates average rolling standard deviation
    :param daily_returns :type Series: daily returns
    :param window_size :type int: window size
    :return :type float: MV/AV
    """
    std = daily_returns.std().iloc[0]

    try:
        return float(daily_returns.rolling(window_size).std().max().iloc[0] / std)
    except ZeroDivisionError:
        return np.nan


def calc_avg_std(daily_returns: Series, window_size: Optional[int] = 20,
                 annual_factor: Optional[int] = NUM_DAYS_IN_YEAR) -> float:
    """
    Calculates max std
    :param daily_returns :type Series: daily returns
    :param window_size :type int: window size
    :param annual_factor :type int: annual factor
    :return :type float: max std
    """
    return float(100 * calc_average_rolling_standard_deviation(daily_returns, window_size).iloc[0] * np.sqrt(annual_factor))


def calc_max_std(daily_returns: Series, window_size: Optional[int] = 20,
                 annual_factor: Optional[int] = NUM_DAYS_IN_YEAR) -> float:
    """
    Calculates max std
    :param daily_returns :type Series: daily returns
    :param window_size :type int: window size
    :param annual_factor :type int: annual factor
    :return :type float: max std
    """
    return float(calc_mv_av(daily_returns, window_size) * calc_std_d(daily_returns, annual_factor))


def calc_draw_down_divided_by_vol(daily_returns: Series, annual_factor: Optional[int] = NUM_DAYS_IN_YEAR,
                                  compounded: Optional[bool] = False) -> float:
    """
    Calculates draw down divided by vol
    :param daily_returns :type Series: daily returns
    :param annual_factor :type int: annual factor
    :param compounded :type bool: (Optional) compounded flag
    :return :type float: max std
    """
    try:
        return float(calc_max_draw_down(daily_returns, compounded) / calc_std_d(daily_returns, annual_factor) * 100)
    except ZeroDivisionError:
        return np.nan


def calc_omega_ratio(daily_returns: Series) -> float:
    """
    Calculates omega ratio
    :param daily_returns :type Series: daily returns
    :return :type float: omega ratio
    """
    positive = daily_returns[ne.evaluate('daily_returns > 0')].sum()
    negative = np.abs(daily_returns[ne.evaluate('daily_returns <= 0')].sum())

    if positive == 0 or negative == 0:
        return np.nan

    try:
        return float(positive / negative)
    except ZeroDivisionError:
        return np.nan

def calc_worst_day(daily_returns: Series) -> float:
    """
    Gets worst trading day
    :param daily_returns :type Series: daily returns
    :return :type float: worst trading
    """
    return float(daily_returns.min().iloc[0])


def calc_std_worst_day(daily_returns: Series, annual_factor: Optional[int] = NUM_DAYS_IN_YEAR) -> float:
    """
    Gets std worst trading day
    :param daily_returns :type Series: daily returns
    :param annual_factor :type int: annual factor
    :return :type float: worst trading
    """
    worst_day = calc_worst_day(daily_returns)
    try:
        return float(np.abs(calc_std_d(daily_returns, annual_factor) / (worst_day * 100)))
    except ZeroDivisionError:
        return np.nan


def calc_best_day(daily_returns: Series) -> float:
    """
    Gets best trading day
    :param daily_returns :type Series: daily returns
    :return :type float: best trading
    """
    return float(daily_returns.max().iloc[0])


def calc_ui(daily_returns: Series) -> float:
    """
    Calculates ui
    :param daily_returns :type Series: daily returns
    :return :type float: ui
    """
    return np.sqrt(daily_returns.pow(2).mean())


def calc_upi(daily_returns: Series, days: Optional[int] = NUM_DAYS_IN_YEAR,
             compounded: Optional[bool] = False) -> float:
    """
    Calculates upi
    :param daily_returns :type Series: daily returns
    :param days :type float: (Optional) days
    :param compounded :type bool: (Optional) compounded flag
    :return :type float: upi
    """
    draw_downs = calc_draw_down(daily_returns, compounded)
    ui = calc_ui(draw_downs)

    try:
        return float(daily_returns.mean() * days / ui)
    except ZeroDivisionError:
        return np.nan


def calc_long_trades(traded_shares: Union[Series,DataFrame]) -> float:
    """
    Calculates long trades
    :param traded_shares :type variable: traded shares
    :return :type int: traded share count
    """
    if traded_shares.empty:
        return np.nan

    if traded_shares.ndim == 1:
        return float(traded_shares[traded_shares > 0].sum())
    return float(traded_shares[traded_shares > 0].sum().sum())


def calc_short_trades(traded_shares: Union[Series,DataFrame]) -> float:
    """
    Calculates short trades
    :param traded_shares :type variable: traded shares
    :return :type int: traded share count
    """
    if traded_shares.empty:
        return np.nan

    if traded_shares.ndim == 1:
        return float(-traded_shares[traded_shares < 0].sum())
    return float(-traded_shares[traded_shares < 0].sum().sum())


def calc_long_short_ratio(traded_shares: Union[Series,DataFrame]) -> float:
    """
    Calculates long trades
    :param traded_shares :type variable: traded shares
    :return :type int: long short ratio
    """
    long_trades = calc_long_trades(traded_shares)
    short_trades = calc_short_trades(traded_shares)

    if long_trades == 0 or short_trades == 0:
        return np.nan

    try:
        return long_trades / short_trades
    except ZeroDivisionError:
        return np.nan


def calc_trade_total(traded_shares: Union[Series,DataFrame]) -> float:
    """
    Calculates trade total
    :param traded_shares :type variable: traded shares
    :return :type int: trade total
    """
    long_trades = calc_long_trades(traded_shares)
    short_trades = calc_short_trades(traded_shares)
    return long_trades + short_trades


def calc_info_ratio(strategy_returns: Series, benchmark_returns: Series, days: Optional[int] = NUM_DAYS_IN_YEAR):
    """
    Calculates info ratio
    :param strategy_returns :type Series: strategy returns monthly
    :param benchmark_returns :type Series: benchmark returns monthly
    :param days: type int: annual factor
    :return :type float: info ratio
    """
    delta = Series(ne.evaluate('strategy_returns - benchmark_returns'), index=strategy_returns.index,
                   dtype=strategy_returns.values.dtype)
    try:
        return float(np.sqrt(days) * delta.mean() / delta.std())
    except ZeroDivisionError:
        return np.nan


def calc_correlation_downside_upside(strategy_returns: Series, benchmark_returns: Series) -> float:
    """
    Calculates correlation downside upside
    :param strategy_returns :type Series: strategy returns monthly
    :param benchmark_returns :type Series: benchmark returns monthly
    :return :type str: correlation downside upside
    """
    downside_mask = ne.evaluate('benchmark_returns < 0')
    upside_mask = ne.evaluate('benchmark_returns > 0')

    try:
        correlation = round(strategy_returns.corr(benchmark_returns) * 100)
    except ValueError:
        correlation = np.nan

    try:
        downside = round(strategy_returns[downside_mask].corr(benchmark_returns[downside_mask]) * 100)
    except ValueError:
        downside = np.nan

    try:
        upside = round(strategy_returns[upside_mask].corr(benchmark_returns[upside_mask]) * 100)
    except ValueError:
        upside = np.nan

    return '{}/{}/{}%'.format(correlation, downside, upside)


def calc_alpha_beta(strategy_returns: Series, benchmark_returns: Series,
                    annual_factor: Optional[int] = NUM_DAYS_IN_YEAR) -> float:
    """
    Calculates alpha beta
    :param strategy_returns :type Series: strategy returns
    :param benchmark_returns :type Series: benchmark returns
    :param annual_factor :type int: (Optional) annual factor
    :return :type str: alpha beta
    """
    try:
        beta = strategy_returns.cov(benchmark_returns) / benchmark_returns.var()
    except ZeroDivisionError:
        beta = np.nan

    diffs_alpha = Series(ne.evaluate('strategy_returns - (beta * benchmark_returns)'),
                         index=strategy_returns.index, dtype=strategy_returns.values.dtype)
    alpha = diffs_alpha.mean() * annual_factor

    sign = '-' if beta < 0 else '+'
    return '{}% {} {}x'.format(round(alpha * 100, 1), sign, round(np.abs(beta), 2))


def calc_alpha_and_beta(strategy_returns: Series, benchmark_returns: Series,
                        annual_factor: Optional[int] = NUM_DAYS_IN_YEAR) -> Tuple[float, float]:
    """
    calculates alpha and beta
    :param strategy_returns :type Series: strategy returns
    :param benchmark_returns :type Series: benchmark returns
    :param annual_factor :type int: (Optional) annual factor
    :return :type tuple: alpha, beta
    """
    try:
        beta = strategy_returns.cov(benchmark_returns) / benchmark_returns.var()
    except ZeroDivisionError:
        beta = np.nan

    diffs_alpha = Series(ne.evaluate('strategy_returns - (beta * benchmark_returns)'), index=strategy_returns.index,
                         dtype=strategy_returns.values.dtype)
    alpha = diffs_alpha.mean() * annual_factor

    return round(float(alpha) * 100, 1), round(float(beta), 2)


def calc_percent_over_perform(strategy_returns: Series, benchmark_returns: Series,
                              annual_factor: Optional[int] = NUM_DAYS_IN_YEAR) -> float:
    """
    Calculates percent over perform
    :param strategy_returns :type Series: strategy returns
    :param benchmark_returns :type Series: benchmark returns
    :param annual_factor :type int: annual Factor
    :return :type str: percent over perform
    """
    diffs = Series(ne.evaluate('strategy_returns - benchmark_returns'), index=strategy_returns.index,
                   dtype=strategy_returns.values.dtype)
    diffs = diffs.rolling(annual_factor).sum()

    try:
        return float(diffs[diffs > 0].count() / diffs[~np.isnan(diffs)].count())
    except ZeroDivisionError:
        return np.nan


def calc_time_in_market(positions: DataFrame) -> float:
    """
    Calculates time in market
    :param positions :type DataFrame: positions
    :return :type time in market
    """
    if positions.empty:
        return np.nan

    total_positions = positions.sum(axis=1)
    return float(total_positions[total_positions != 0].count() / total_positions.index.size) * 100


def calc_total_return(daily_returns: Series, compounded: Optional[bool] = True) -> float:
    """
    Calculates total return
    :param daily_returns :type Series: daily returns
    :param compounded :type bool: (Optional) compounded flag
    :return :type total return
    """
    if compounded:
        return float((np.int32(1) + daily_returns).prod() - np.int32(1))
    return float(daily_returns.sum())


def calc_avg_monthly_return(daily_returns: Series, compounded: Optional[bool] = True) -> float:
    """
    Calculates average monthly return
    :param daily_returns :type Series: daily returns
    :param compounded :type bool: (Optional) compounded flag
    :return :type average monthly return
    """
    if compounded:
        monthly_returns = (np.int32(1) + daily_returns).resample('ME').prod() - np.int32(1)
        return float(monthly_returns.mean())
    return float(daily_returns.resample('ME').sum().mean())


def calc_avg_annual_max_dd(daily_returns: Series, compounded: Optional[bool] = True) -> float:
    """
    Calculates average annual max draw down
    :param daily_returns :type Series: daily returns
    :param compounded :type bool: (Optional) compounded flag
    :return :type average annual draw down
    """
    return float(daily_returns.resample('YE').apply(calc_max_draw_down, compounded=compounded).mean())


def calc_percent_profitable_months(daily_returns: Series, compounded: Optional[bool] = True) -> float:
    """
    Calculates percent profitable months
    :param daily_returns :type Series: daily returns
    :param compounded :type bool: (Optional) compounded flag
    :return :type percent profitable months
    """
    if compounded:
        monthly_returns = (np.int32(1) + daily_returns).resample('ME').prod() - np.int32(1)
    else:
        monthly_returns = daily_returns.resample('ME').sum()

    try:
        return float((monthly_returns > 0).count() / monthly_returns.count())
    except ZeroDivisionError:
        return np.nan


def calc_max_monthly_dd(daily_returns: Series, compounded: Optional[bool] = True) -> float:
    """
    Calculates max monthly draw down
    :param daily_returns :type Series: daily returns
    :param compounded :type bool: (Optional) compounded flag
    :return :type max monthly draw down
    """
    return float(daily_returns.resample('ME').apply(calc_max_draw_down, compounded=compounded).max())

def calc_vami(daily_returns: Series, compounded: Optional[bool] = True) -> Series:
    """
    Calculates vami
    :param daily_returns :type Series: daily returns
    :param compounded :type bool: (Optional) compounded flag
    :return :type Series: vami
    """
    if compounded:
        return (np.int32(1) + daily_returns).cumprod() * 1000
    return (np.int32(1) + daily_returns).cumsum() * 1000


def calc_compounded_return(daily_returns: Series) -> float:
    """
    Calculates compounded return value from a series of inputs
    :param daily_returns :type Series: daily returns
    :return :type float: compounded monthly return
    """
    return float((daily_returns + 1).prod() - 1)


@njit(nogil=True)
def _calc_compounded_sub_period(daily_returns, output):
    """
    Calculate sub period net returns
    :param daily_returns :type ndarray: returns and aum
    :param output :type ndarray: output holder
    """
    num_dates, _ = daily_returns.shape
    last_known_aum = daily_returns[0][1]
    sub_daily_return = 0

    for i in range(num_dates):
        return_value, aum = daily_returns[i]

        if aum != last_known_aum:
            output[i - 1] = sub_daily_return
            last_known_aum = aum
            sub_daily_return = return_value
        else:
            sub_daily_return += return_value

    output[-1] = sub_daily_return


def calc_compounded_monthly_returns(daily_returns: Series, aum: Series) -> Series:
    """
    Calculates compounded monthly returns
    :param daily_returns :type Series: daily returns
    :param aum :type Series: assets under management
    :return :type Series: compounded monthly returns
    """
    if aum.empty or aum.diff().abs().sum() == 0:
        return daily_returns.resample('ME').sum()

    unique_dates = np.union1d(daily_returns.index, aum.index)
    daily_returns_aligned = daily_returns.reindex(unique_dates)
    daily_returns_aligned.fillna(0, inplace=True)

    aum_aligned = aum.reindex(unique_dates)
    aum_aligned.ffill(inplace=True)
    aum_aligned.fillna(0, inplace=True)

    combined_returns = concat([daily_returns_aligned, aum_aligned], axis=1)
    combined_returns.columns = ['daily_returns', 'aum']
    df = combined_returns.groupby(Grouper(freq='M'))

    output = np.empty((len(df.groups, )), dtype=np.float32)
    output.fill(np.nan)
    compounded_returns = Series(output, index=list(df.groups))
    for trade_date in df.groups:
        monthly_returns = df.get_group(trade_date)
        output = np.empty((monthly_returns.index.size,))
        output.fill(np.nan)
        _calc_compounded_sub_period(monthly_returns.values, output)
        output = output[~np.isnan(output)]
        compounded_returns[trade_date] = output[0] if output.size == 1 else calc_compounded_return(Series(output))
    return compounded_returns


def calc_compounded_annual_returns(compounded_monthly_returns: Series) -> Series:
    """
    Calculates compounded annual returns
    :param compounded_monthly_returns :type Series: compounded monthly returns
    :return :type Series: compounded annual returns
    """
    return compounded_monthly_returns.resample('YE').apply(lambda x: calc_compounded_return(x))


def calc_winning_months(daily_returns: Series) -> int:
    """
    Calculates number of winning months
    :param daily_returns :type Series: daily returns
    :return :type int: number of winning months
    """
    monthly_returns = daily_returns.resample('ME').sum()
    return int(monthly_returns[ne.evaluate('monthly_returns > 0')].count())


def calc_losing_months(daily_returns: Series) -> int:
    """
    Calculates number of losing months
    :param daily_returns :type Series: daily returns
    :return :type int: number of losing months
    """
    monthly_returns = daily_returns.resample('ME').sum()
    return int(monthly_returns[ne.evaluate('monthly_returns < 0')].count())


def calc_avg_monthly_gain(daily_returns: Series) -> float:
    """
    Calculates average monthly gains
    :param daily_returns :type Series: daily returns
    :return :type float: average monthly gains
    """
    monthly_returns = daily_returns.resample('ME').sum()
    return float(monthly_returns[ne.evaluate('monthly_returns > 0')].mean())


def calc_avg_monthly_loss(daily_returns: Series) -> float:
    """
    Calculates average monthly losses
    :param daily_returns :type Series: daily returns
    :return :type float: average monthly losses
    """
    monthly_returns = daily_returns.resample('ME').sum()
    return float(monthly_returns[ne.evaluate('monthly_returns < 0')].mean())


def calc_cum_monthly_returns(daily_returns: Series) -> Series:
    """
    Calculates cumulative monthly returns
    :param daily_returns :type Series: daily returns
    :return :type Series: cumulative monthly returns
    """
    return daily_returns.resample('ME').sum().cumsum()


def calc_rolling_12month_returns(daily_returns: Series) -> Series:
    """
    Calculates cumulative rolling 12 month returns
    :param daily_returns :type Series: daily returns
    :return :type Series: rolling 12 month returns
    """
    return daily_returns.resample('ME').sum().rolling(window=12).sum()


def calc_cagr(daily_returns: Series, annual_factor: Optional[int] = NUM_DAYS_IN_YEAR) -> float:
    """
    Calculates CAGR
    :param daily_returns :type Series: daily_returns
    :param annual_factor :type int: annual factor
    :return :type float: CAGR
    """
    try:
        cumulative_returns = (np.int32(1) + daily_returns).prod()
        number_of_years = len(daily_returns) / annual_factor
        return float((cumulative_returns.iloc[0] ** (1/number_of_years) -1) *100)
    except (ZeroDivisionError, IndexError):
        return np.nan


def calc_yearly_sharpe_ratio(daily_returns, risk_free_rate: Optional[float] = 0,
                             annual_factor: Optional[int] = NUM_DAYS_IN_YEAR,
                             compounded: Optional[bool] = True) -> Series:
    """
    Calculates yearly sharpe ratios from returns
    :param daily_returns :type Series: daily returns
    :param risk_free_rate :type float: (Optional) average yield of 1 year treasury over period
    :param annual_factor :type int: annual factor
    :param compounded :type bool: (Optional) compounded flag
    :return :type Series: yearly sharpe ratios
    """
    return daily_returns.resample('YE').apply(lambda x: calc_sharpe_ratio(x, risk_free_rate, annual_factor, compounded))


def calc_yearly_annualized_return(daily_returns: Series, annual_factor: Optional[int] = NUM_DAYS_IN_YEAR,
                                  compounded: Optional[bool] = True) -> Series:
    """
    Calculates yearly annual returns ratios from returns
    :param daily_returns :type Series: daily returns
    :param annual_factor :type int: annual factor
    :param compounded :type bool: (Optional) compounded flag
    :return :type Series: yearly sharpe ratios
    """
    return daily_returns.resample('YE').apply(lambda x: calc_annualized_return(x, annual_factor, compounded))
