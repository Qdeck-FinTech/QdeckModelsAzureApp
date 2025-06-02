from enum import unique, IntEnum


@unique
class SignalTransformationGroupings(IntEnum):
    unknown = 0
    none = 1
    universe = 2
    sector = 3
    industry = 4


@unique
class ItemType(IntEnum):
    unknown = 0
    folder = 1
    factor = 2
    idea = 3
    strategy = 4
    portfolio = 5
    mixture = 6
    benchmark = 7
    market_place = 8
    instrument = 9


@unique
class StrategyStatus(IntEnum):
    unknown = 0,
    stopped = 1,
    running = 2,
    pending = 3,
    finished = 4,
    failed = 5,
    starting = 6,
    creating = 7
    canceled = 8
    retrying = 9
    canceling = 10


@unique
class OptimizationType(IntEnum):
    unknown = 0
    manual = 1
    algorithmic = 2
    allocation = 3


@unique
class StrategyType(IntEnum):
    unknown = 0
    strategy = 1
    portfolio = 2
    futures_strategy = 3
    draft = 4
    imported = 5
    imported_returns = 6
    symbolic_link = 7


@unique
class AlgorithmicOptimization(IntEnum):
    unknown = 0
    sharpe_ratios = 1
    sharpe_ratios_multiplied_by_strategy_returns = 2
    maximize_sharpe_ratio = 3
    maximize_sharpe_ratio_multiplied_by_strategy_returns = 4
    maximize_sharpe_ratio_analytic = 5
    technical = 6
    minimum_etl = 7


@unique
class AlgorithmicOptimizationPreconditions(IntEnum):
    unknown = 0
    equal_weighted = 1
    proportional_strategy_sharpe_ratio = 2
    proportional_strategy_sharpe_ratio_multiplied_by_factor_returns = 3
    random = 4


@unique
class FactorType(IntEnum):
    unknown = 0
    custom = 1
    imported = 2


@unique
class IndexOption(IntEnum):
    unknown = 0
    unbiased = 1
    biased = 2
    dropped = 3


@unique
class Technical(IntEnum):
    unknown = 0
    ema = 1
    dema = 2
    macd = 3


@unique
class MacdValue(IntEnum):
    unknown = 0
    diff = 1
    signal = 2
    indicator = 3


@unique
class BackAdjustMethod(IntEnum):
    unknown = 0
    none = 1
    geometric = 2
    arithmetic = 3


@unique
class BarFrequency(IntEnum):
    unknown = 0
    daily = 1
    minute = 2


@unique
class RollMethods(IntEnum):
    unknown = 0
    first_day_new_month = 1
    nearest_future = 2
    weighted_volume = 3
    contract_volume = 4
    price_index = 5
    average = 6


@unique
class Container(IntEnum):
    unknown = 0
    folder = 1
    strategy_group = 2
    portfolio_group = 3


@unique
class PortfolioType(IntEnum):
    unknown = 0
    portfolio = 1
    draft = 2
    symbolic_link = 3


@unique
class DateRange(IntEnum):
    unknown = 0
    common = 1
    custom = 2
    min_max = 3


@unique
class Frequency(IntEnum):
    unknown = 0
    daily = 1
    weekly = 2
    monthly = 3
    annually = 4
    quarterly = 5


@unique
class ErrorType(IntEnum):
    unknown = 0
    general = 1
    instrument_conversion = 2


@unique
class ReturnFormat(IntEnum):
    unknown = 0
    simple = 1
    compounded = 2


@unique
class WeightRedistribution(IntEnum):
    unknown = 0
    none = 1
    reweight = 2
    cash = 3
