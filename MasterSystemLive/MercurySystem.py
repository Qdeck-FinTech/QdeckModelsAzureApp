import numpy as np
import pandas as pd
import datetime
import ta
import json
import os, sys, inspect

current_path = os.path.abspath(inspect.getfile(inspect.currentframe()))
code_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
project_dir = os.path.dirname(code_dir)
sys.path.insert(0, project_dir)

import clr

from System import DateTime  # type: ignore
from System.Collections.Generic import List  # type: ignore

clr.AddReference("Mercury")
from Mercury import (
    MercuryRunner,
    MercuryRunConfig,
    IMercurySystem,
    PriceField,
    TickerInfo,
)  # type: ignore
from logger.net_logger import net_logger
from configuration.configuration import runner_config

from utils.np_interop import to_numpy
from stats.MercuryStats import (
    BestDayStat,
    WorstDayStat,
    StandardDeviationDayStat,
    SharpeRatioDayStat,
    AARStat,
    MVAVStat,
    DDMaxDayStat,
    DDMaxDayOverVolStat,
    CurrentDDStat,
    CAGRStat,
    YTDPct,
    QTDPctStat,
    OneYearPctStat,
    TwoYearPctStat,
    FiveYearPctStat,
    TenYearPctStat,
    TwentyYearPctStat,
    LongShortRatioStat,
)


class MasterSystemLive(IMercurySystem):
    __namespace__ = "Mercury"

    def __init__(self, cfg):
        self.name = "MasterSystemLive"

        self.cfg = cfg

        self.current_date = cfg.start

        self.indexes = []
        self.symbols = cfg.symbols
        self.previous_days_month = self.current_date.Month
        self.rebalance = True

        self.US_fixed_income_weights = []
        self.other_fixed_income_weights = []
        self.US_equity_weights = []
        self.sector_equity_weights = []
        self.international_equity_weights = []
        self.portfolio_level_weights_l = []
        self.portfolio_level_weights_l_2 = []
        self.prices = []

    # region Property methods

    def get_contextHolder(self):
        return self.contextHolder

    def set_contextHolder(self, contextHolder):
        self.contextHolder = contextHolder

    def get_name(self):
        return self.name

    def set_name(self, name):
        self.name = name

    def get_start(self):
        return self.start

    def set_start(self, start):
        self.start = start

    def get_end(self):
        return self.end

    def set_end(self, end):
        self.end = end

    def get_symbols(self):
        return self.symbols

    def set_symbols(self, symbols):
        self.symbols = symbols

    def get_indexes(self):
        return self.indexes

    def set_indexes(self, indexes):
        self.indexes = indexes

    def get_current_date(self):
        return self.current_date

    def set_current_date(self, current_date):
        self.current_date = current_date

    def get_stop_loss_date(self):
        return self.stop_loss_date

    def set_stop_loss_date(self, stop_loss_date):
        self.stop_loss_date = stop_loss_date

    # endregion

    def order_filled(self, order):
        pass

    def US_Fixed_Income_calculation(self):
        ## BIL_E = 1 to 3 month treasury (cash etf)
        ## SHV = <1 year gov, SHY = 1-3 year gov, IEF = 7-10 year gov, TLH= 10 - 20 ETF
        ## CSJ = short term corp, AGG = Agg, CLY = long term corp
        ## HYG = High Yield

        ## Currently 8 symbols, so default = 12.5%

        cfg_strategy_section = self.cfg.portfolio["Fixed Income"]["strategies"][
            "US Fixed Income"
        ]

        symbols_inc_cash = [sy for sy in cfg_strategy_section["symbols"]]
        symbols = [sy for sy in symbols_inc_cash if sy != self.cfg.cash_etf]
        tickers = [s for s in self.symbols if s.ticker in symbols]

        def vol(prices, window):
            returns = np.log(prices[:window]) - np.log(prices[1 : (window + 1)])
            return np.std(returns, axis=0) * np.sqrt(250)

        approach = cfg_strategy_section[
            "approach"
        ]  # 'trend' # 'vol' 'trend' 'duration'

        trend_vol_adjust = cfg_strategy_section["trend_vol_adjust"] == "True"  # True

        min_weight = cfg_strategy_section["min_weight"]  # 0
        max_weight = cfg_strategy_section["max_weight"]  # 0.25
        vol_window = cfg_strategy_section["vol_window"]  # 200

        fast_window = cfg_strategy_section["fast_window"]  # 20
        slow_window = cfg_strategy_section["slow_window"]  # 50

        if approach == "trend":
            ## Currently 8 symbols, so default = 12.5%

            def f_sma(prices, window):
                return np.mean(prices[:window], axis=0)

            close_less_window = False
            sy_trend_d = {}
            vol_d = {}

            for ticker in tickers:
                sy = ticker.ToString()

                close = self.close[sy]

                if len(close) <= vol_window:
                    close_less_window = True

                sma_short = f_sma(close, fast_window)
                sma_long = f_sma(close, slow_window)

                trend_exist = 1 if sma_short > sma_long else 0  #

                sy_trend_d[sy] = trend_exist

                ## Vol adjustement
                if trend_vol_adjust:
                    vol_val = vol(close, vol_window)
                    vol_val = np.clip(vol_val, 0.03, None)

                    vol_d[sy] = vol_val

            if trend_vol_adjust:  # If no trend then 'scale' up vol
                sy_adj_trend_d = {
                    sy: (trend / vol_d[sy]) for sy, trend in sy_trend_d.items()
                }

            else:
                sy_adj_trend_d = sy_trend_d.copy()

            total_weighted_trend = sum(sy_adj_trend_d.values())

            total_weight_ex_cash = 0
            raw_weight = 0

            for ticker in tickers:
                sy = ticker.ToString()

                if close_less_window:
                    weight = 1 / len(symbols)

                else:
                    raw_weight = (
                        sy_adj_trend_d[sy] / total_weighted_trend
                        if total_weighted_trend > 0
                        else 0
                    )

                weight = np.clip(raw_weight, min_weight, max_weight)
                total_weight_ex_cash += weight

                self.cfg.strategy_symbol_weights["Fixed Income"]["strategies"][
                    "US Fixed Income"
                ]["symbol weights"][sy] = weight

                if self.cfg.save_outputs:
                    self.US_fixed_income_weights.append(
                        {
                            "symbol": sy,
                            "date": self.current_date.ToString(),
                            "weight": weight,
                        }
                    )

            cash_weight = 1 - total_weight_ex_cash
            cash_weight = np.clip(cash_weight, 0, 1)

            cash_etf_ticker = [
                s for s in self.symbols if s.ticker == self.cfg.cash_etf
            ][0]
            cash_etf_symbol = cash_etf_ticker.ToString()

            self.cfg.strategy_symbol_weights["Fixed Income"]["strategies"][
                "US Fixed Income"
            ]["symbol weights"][cash_etf_symbol] = cash_weight

            if self.cfg.save_outputs:
                self.US_fixed_income_weights.append(
                    {
                        "symbol": cash_etf_symbol,
                        "date": self.current_date.ToString(),
                        "weight": cash_weight,
                    }
                )

    def Other_Fixed_Income_calculation(self):
        cfg_strategy_section = self.cfg.portfolio["Fixed Income"]["strategies"][
            "Other Fixed Income"
        ]

        symbols_inc_cash = [sy for sy in cfg_strategy_section["symbols"]]
        symbols = [sy for sy in symbols_inc_cash if sy not in [self.cfg.cash_etf]]
        tickers = [s for s in self.symbols if s.ticker in symbols]

        def f_sma(prices, window):
            return np.mean(prices[:window], axis=0)

        def vol(prices, window):
            returns = np.log(prices[:window]) - np.log(prices[1 : (window + 1)])
            return np.std(returns, axis=0) * np.sqrt(250)

        trend_vol_adjust = cfg_strategy_section["trend_vol_adjust"] == "True"  # True

        min_weight = cfg_strategy_section["min_weight"]  # 0
        max_weight = cfg_strategy_section["max_weight"]  # 1
        vol_window = cfg_strategy_section["vol_window"]  # 200

        fast_window = cfg_strategy_section["fast_window"]  # 20
        slow_window = cfg_strategy_section["slow_window"]  # 100

        close_less_window = False

        sy_trend_d = {}
        vol_d = {}

        for ticker in tickers:
            sy = ticker.ToString()

            close = self.close[sy]

            if len(close) <= vol_window:
                close_less_window = True

            sma_short = f_sma(close, fast_window)
            sma_long = f_sma(close, slow_window)

            trend_exist = 1 if sma_short > sma_long else 0

            sy_trend_d[sy] = trend_exist

            if trend_vol_adjust:
                vol_val = vol(close, vol_window)
                vol_val = np.clip(vol_val, 0.03, None)
                vol_d[sy] = vol_val

        if trend_vol_adjust:  # If no trend then 'scale' up vol
            sy_adj_trend_d = {
                sy: (trend / vol_d[sy]) for sy, trend in sy_trend_d.items()
            }
        else:
            sy_adj_trend_d = sy_trend_d.copy()

        total_weighted_trend = sum(sy_adj_trend_d.values())
        total_weight_ex_cash = 0
        raw_weight = 0

        for ticker in tickers:
            sy = ticker.ToString()

            if close_less_window:
                weight = 1 / len(symbols)

            else:
                raw_weight = (
                    sy_adj_trend_d[sy] / total_weighted_trend
                    if total_weighted_trend > 0
                    else 0
                )

            weight = np.clip(raw_weight, min_weight, max_weight)
            total_weight_ex_cash += weight

            self.cfg.strategy_symbol_weights["Fixed Income"]["strategies"][
                "Other Fixed Income"
            ]["symbol weights"][sy] = weight

            if self.cfg.save_outputs:
                self.other_fixed_income_weights.append(
                    {
                        "symbol": sy,
                        "date": self.current_date.ToString(),
                        "weight": weight,
                    }
                )

        cash_weight = 1 - total_weight_ex_cash
        cash_weight = np.clip(cash_weight, 0, 1)

        cash_etf_ticker = [s for s in self.symbols if s.ticker == self.cfg.cash_etf][0]
        cash_etf_symbol = cash_etf_ticker.ToString()

        self.cfg.strategy_symbol_weights["Fixed Income"]["strategies"][
            "Other Fixed Income"
        ]["symbol weights"][cash_etf_symbol] = cash_weight

        if self.cfg.save_outputs:
            self.other_fixed_income_weights.append(
                {
                    "symbol": cash_etf_symbol,
                    "date": self.current_date.ToString(),
                    "weight": cash_weight,
                }
            )

    def US_Equity_calculation(self):
        # symbols = [sy for sy in self.cfg.portfolio['Equity']['strategies']['US Equity']['symbols']]

        cfg_strategy_section = self.cfg.portfolio["Equity"]["strategies"]["US Equity"]

        symbols_inc_cash = [sy for sy in cfg_strategy_section["symbols"]]
        symbols = [sy for sy in symbols_inc_cash if sy != self.cfg.cash_etf]
        tickers = [s for s in self.symbols if s.ticker in symbols]

        weights = cfg_strategy_section[
            "weights"
        ]  # {'IVV_E':0.50, 'ONEQ_E':0.25, 'IWM_E':0.25}  # Fixed weights

        for ticker in tickers:
            sy = ticker.ToString()

            weight = weights.get(ticker.ticker, 0)

            self.cfg.strategy_symbol_weights["Equity"]["strategies"]["US Equity"][
                "symbol weights"
            ][sy] = weight

            if self.cfg.save_outputs:
                self.US_equity_weights.append(
                    {
                        "symbol": sy,
                        "date": self.current_date.ToString(),
                        "weight": weight,
                    }
                )

    def Sector_Equity_calculation(self):
        cfg_strategy_section = self.cfg.portfolio["Equity"]["strategies"][
            "Sector Equity"
        ]

        symbols = [sy for sy in cfg_strategy_section["symbols"]]
        tickers = [s for s in self.symbols if s.ticker in symbols]

        def f_sma(prices, window):
            return np.mean(prices[:window], axis=0)

        min_weight = cfg_strategy_section["min_weight"]  # 0
        max_weight = cfg_strategy_section["max_weight"]  # 0.20

        fast_window = cfg_strategy_section["fast_window"]  # 20
        slow_window = cfg_strategy_section["slow_window"]  # 50

        sma_approach = cfg_strategy_section["approach"] == "sma"

        sy_trend_d = {}
        trend_exist = 0.5

        for ticker in tickers:
            sy = ticker.ToString()
            close = self.close[sy]

            if sma_approach:
                sma_short = f_sma(close, fast_window)
                sma_long = f_sma(close, slow_window)

                trend_exist = 1 if sma_short > sma_long else 0.5

            sy_trend_d[sy] = trend_exist

        n_trends = sum(sy_trend_d.values())

        for ticker in tickers:
            sy = ticker.ToString()

            raw_weight = sy_trend_d[sy] / n_trends if n_trends > 0 else 0
            weight = np.clip(raw_weight, min_weight, max_weight)

            self.cfg.strategy_symbol_weights["Equity"]["strategies"]["Sector Equity"][
                "symbol weights"
            ][sy] = weight

            if self.cfg.save_outputs:
                self.sector_equity_weights.append(
                    {
                        "symbol": sy,
                        "date": self.current_date.ToString(),
                        "weight": weight,
                    }
                )

    def International_Equity_calculation(self):
        ## Idea, default US if international not trending. Otherwise whichever trends more
        ## Otherwise stated: Long S&P 500, 0 to international, UNLESS international trends more
        ## SPY_E = S&P 500 ETF, ACWI_E All country world ETF

        cfg_strategy_section = self.cfg.portfolio["Equity"]["strategies"][
            "International Equity"
        ]

        symbols = [sy for sy in cfg_strategy_section["symbols"]]
        tickers = [s for s in self.symbols if s.ticker in symbols]

        US_symbol = cfg_strategy_section[
            "US_symbol"
        ]  # [sy for sy in symbols if sy!='ACWI_E'][0]
        International_symbol = cfg_strategy_section["International_symbol"]

        # find ticker for US and International symbols
        US_ticker = [s for s in tickers if s.ticker == US_symbol][0]
        International_ticker = [s for s in tickers if s.ticker == International_symbol][
            0
        ]

        US_symbol = US_ticker.ToString()
        International_symbol = International_ticker.ToString()

        def f_sma(prices, window):
            return np.mean(prices[:window], axis=0)

        default_equity = cfg_strategy_section["default_equity"] == "True"  # True

        fast_window = cfg_strategy_section["fast_window"]  # 40
        slow_window = cfg_strategy_section["slow_window"]  # 100

        sy_trend_d = {}
        sy_position_d = {}

        sy_position_d[US_symbol] = 1
        sy_position_d[International_symbol] = 0

        for ticker in tickers:
            sy = ticker.ToString()

            close = self.close[sy]

            sma_short = f_sma(close, fast_window)
            sma_long = f_sma(close, slow_window)

            sy_trend_d[sy] = sma_short / sma_long

        ## Only time when we allocate to internation equity

        if default_equity:
            sy_position_d[US_symbol] = 1
            sy_position_d[International_symbol] = 0

            if sy_trend_d[International_symbol] > sy_trend_d[US_symbol]:
                sy_position_d[US_symbol] = 0
                sy_position_d[International_symbol] = 1

        for ticker in tickers:
            sy = ticker.ToString()

            weight = sy_position_d[sy]

            self.cfg.strategy_symbol_weights["Equity"]["strategies"][
                "International Equity"
            ]["symbol weights"][sy] = weight

            if self.cfg.save_outputs:
                self.international_equity_weights.append(
                    {
                        "symbol": sy,
                        "date": self.current_date.ToString(),
                        "weight": weight,
                    }
                )

    def Factor_Equity_calculation(self):
        ## XMLV_E=S&P midcap low vol, MTUM_E=S&P 500 momentum, IQLT_E=International Quality

        symbols = [
            sy
            for sy in self.cfg.portfolio["Equity"]["strategies"]["Factor Equity"][
                "symbols"
            ]
        ]
        tickers = [s for s in self.symbols if s.ticker in symbols]

        for ticker in tickers:
            sy = ticker.ToString()

            weight = 1 / len(symbols)
            self.cfg.strategy_symbol_weights["Equity"]["strategies"]["Factor Equity"][
                "symbol weights"
            ][sy] = weight

    def update_rebalance_date(self):
        ## Currently done as monthly, start of month, for simplicity
        if self.cfg.rebalance_time_scale == "M":
            if self.current_date.Month != self.previous_days_month:
                self.rebalance = True
                self.previous_days_month = self.current_date.Month

        else:
            raise ValueError("Unsuitable rebalance period")

    def compute_weights(self):
        ## Benchmark approach does not scale by strategy symbols, as only one symbol in each sector
        if self.cfg.approach == "master":
            self.US_Fixed_Income_calculation()
            self.Other_Fixed_Income_calculation()
            self.US_Equity_calculation()
            self.Sector_Equity_calculation()
            self.International_Equity_calculation()
            self.Factor_Equity_calculation()

        ### NEW
        ### Allows the repeating of symbols in the portfolio

        self.symbol_weights_d_2 = {sy.ToString(): 0 for sy in self.symbols}

        for sector, sector_data in self.cfg.strategy_symbol_weights.items():
            sector_weight = sector_data["weight"]

            for strategy, strategy_data in sector_data["strategies"].items():
                strategy_weight = strategy_data["weight"]

                for sy, sy_weight in strategy_data["symbol weights"].items():
                    self.symbol_weights_d_2[sy] += (
                        sector_weight * strategy_weight * sy_weight
                    )
                    self.portfolio_level_weights_l_2.append(
                        {
                            "symbol": sy,
                            "date": self.current_date.ToString(),
                            "weight": self.symbol_weights_d_2[sy],
                        }
                    )

    def run(self):
        # print(self.current_date.ToString())

        self.update_rebalance_date()

        if self.cfg.save_outputs:
            close = self.contextHolder.market_data_loader.get_price_field(
                PriceField.Close
            )
            self.close = {k: to_numpy(close[k]) for k in sorted(close.Keys)}

            for sy, price in self.close.items():
                self.prices.append(
                    {
                        "symbol": sy,
                        "date": self.current_date.ToString(),
                        "price": price[0],
                    }
                )

        if self.rebalance:
            # print("rebalance: " + self.current_date.ToString())

            close = self.contextHolder.market_data_loader.get_price_field(
                PriceField.Close
            )
            self.close = {k: to_numpy(close[k]) for k in sorted(close.Keys)}

            self.compute_weights()

            for symbol, weight in self.symbol_weights_d_2.items():
                self.contextHolder.oms.add_weight_order(
                    self.name, TickerInfo.Deserealize(symbol), weight
                )

            self.rebalance = False

    def run_close(self):
        pass

    def run_open(self):
        pass

    def set_equity(self):
        pass


class MasterConfig(MercuryRunConfig):
    __namespace__ = "Mercury"

    def __init__(self, cfg=None, max_buffer=201, save_outputs=True):
        super().__init__()

        self.end = DateTime.Today.AddDays(-1)
        self.max_buffer_size = max_buffer
        self.save_outputs = save_outputs

        self.require_data_from_all_symbols = False
        self.pl_options.include_commission = False
        self.pl_options.compounded_pl = True

        self.name = "MasterSystemLive"

        self.save_stats_to_file = False
        self.save_outputs_to_file = False
        self.output_folder = "output"

        self.production = False
        self.publish_status = False

        self.cash_etf = "BIL_E"

        self.approach = "master"

        # overwrite default stats
        self.mercuryStatistics = [
            BestDayStat(),
            WorstDayStat(),
            StandardDeviationDayStat(),
            SharpeRatioDayStat(),
            AARStat(),
            MVAVStat(),
            DDMaxDayStat(),
            DDMaxDayOverVolStat(),
            CurrentDDStat(),
            CAGRStat(),
            YTDPct(),
            QTDPctStat(),
            OneYearPctStat(),
            TwoYearPctStat(),
            FiveYearPctStat(),
            TenYearPctStat(),
            TwentyYearPctStat(),
            LongShortRatioStat(),
        ]

        # overwrite from cfg
        if cfg is not None:
            start_date = pd.to_datetime(cfg["Start_Date"])
            self.start = DateTime(start_date.year, start_date.month, start_date.day)
            self.equity = float(cfg["Initial_Equity"])

            self.compounded_pl = cfg["Compounded_pl"]
            self.pl_options.compounded_pl = cfg["Compounded_pl"]

            self.approach = cfg["approach"]
            self.cash_etf = cfg["cash_etf"]

            portfolio = cfg["Portfolio"]
            self.portfolio = portfolio

            self.strategy_symbol_weights = (
                portfolio.copy()
            )  ## MAin object (dict) used for adding weights

            symbols_list = []

            for sector, sector_data in portfolio.items():
                for strategy, strategy_data in sector_data["strategies"].items():
                    strategy_symbols = list(strategy_data["symbols"])
                    strategy_ticker_symbols = [
                        s for s in cfg["symbols"] if s["ticker"] in strategy_symbols
                    ]
                    strategy_tickers = self.get_tickers(strategy_ticker_symbols)

                    if cfg["approach"] == "master":
                        default_weight = (
                            0  # will add the actual allocation in the model
                        )
                    else:
                        default_weight = 1 / len(
                            strategy_symbols
                        )  # benchmark is equal weight of the symbols

                    self.strategy_symbol_weights[sector]["strategies"][strategy][
                        "symbol weights"
                    ] = {sy.ToString(): default_weight for sy in strategy_tickers}

                    symbols_list += strategy_symbols

            # Filter symbols by matching 'id' in symbols_ids
            tickers = [s for s in cfg["symbols"] if s["ticker"] in symbols_list]
            self.symbols = self.get_tickers(tickers)

            self.rebalance = cfg["Rebalance"]
            if self.rebalance == "Monthly":
                self.rebalance_time_scale = "M"  # Simplified with only monthly for now. Can add quarterly in time.
            else:
                raise ValueError("Unsuitable rebalance period")

            self.next_rebal_date = self.start
            self.systems = [MasterSystemLive(self)]

    def parse_config_file(self, config_file):
        with open(config_file) as json_data:
            inputs = json.load(json_data)

        start_date = pd.to_datetime(inputs["Start_Date"])
        self.start = DateTime(start_date.year, start_date.month, start_date.day)
        self.equity = float(inputs["Initial_Equity"])

        self.compounded_pl = inputs["Compounded_pl"]
        self.pl_options.compounded_pl = inputs["Compounded_pl"]

        self.approach = inputs["approach"]
        self.cash_etf = inputs["cash_etf"]

        portfolio = inputs["Portfolio"]
        self.portfolio = portfolio

        self.strategy_symbol_weights = (
            portfolio.copy()
        )  ## MAin object (dict) used for adding weights

        symbols_list = []

        for sector, sector_data in portfolio.items():
            for strategy, strategy_data in sector_data["strategies"].items():
                strategy_symbols = list(strategy_data["symbols"])
                strategy_ticker_symbols = [
                    s for s in inputs["symbols"] if s["ticker"] in strategy_symbols
                ]
                strategy_tickers = self.get_tickers(strategy_ticker_symbols)

                if inputs["approach"] == "master":
                    default_weight = 0  # will add the actual allocation in the model
                else:
                    default_weight = 1 / len(
                        strategy_symbols
                    )  # benchmark is equal weight of the symbols

                self.strategy_symbol_weights[sector]["strategies"][strategy][
                    "symbol weights"
                ] = {sy.ToString(): default_weight for sy in strategy_tickers}

                symbols_list += strategy_symbols

        # Filter symbols by matching 'id' in symbols_ids
        tickers = [s for s in inputs["symbols"] if s["ticker"] in symbols_list]

        self.symbols = self.get_tickers(tickers)

        self.rebalance = inputs["Rebalance"]
        if self.rebalance == "Monthly":
            self.rebalance_time_scale = (
                "M"  # Simplified with only monthly for now. Can add quarterly in time.
            )
        else:
            raise ValueError("Unsuitable rebalance period")

        self.next_rebal_date = self.start
        self.systems = [MasterSystemLive(self)]

    def get_tickers(self, symbols):
        tickers = List[TickerInfo]()
        for symbol in symbols:
            ticker = TickerInfo()
            ticker.ticker = symbol["ticker"]
            ticker.id = symbol["id"]
            ticker.name = symbol["ticker"]
            ticker.country = symbol["country"]
            tickers.Add(ticker)
        return tickers


class MLModelRunner(MercuryRunner):
    __namespace__ = "Mercury"

    def load_model_config(self, modelId):
        cfg_json = None

        cfg_file = os.path.join(
            os.path.dirname(current_path), "config", f"qdeck_settings_{modelId}.json"
        )

        if not os.path.isfile(cfg_file):
            cfg_file = os.path.join(
                os.path.dirname(current_path), "config", "qdeck_settings.json"
            )

        if os.path.isfile(cfg_file):
            with open(cfg_file, "r") as file:
                cfg_json = file.read()

        return cfg_json

    def run_model(self, model_id=0, update_qdeck=0, live=0, config=None):
        cfg_data = None

        if model_id > 0:
            # load configuration from database
            cfg_data_json = self.load_model_config(str(model_id))

            if cfg_data_json is not None:
                cfg_data = json.loads(cfg_data_json)

        elif config is not None:
            # load configuration from file, if provided
            with open(config) as json_data:
                cfg_data = json.load(json_data)

        # build system configuration
        run_config = MasterConfig(cfg_data, 201, False)

        if model_id > 0:
            run_config.model_id = model_id

            run_config.update_qdeck = True
            run_config.production = True

        if live > 0:
            run_config.goLive = True

        # run simulation
        runId = self.run(run_config)

        print("completed! run id: " + str(runId))

        return runId


def f_pnl_by_symbol(runner=None):
    ### Get pnl for each currency in dataframe

    if runner == None:
        return None

    ctx = runner.GetContext()

    # get symbols from config
    symbols = ctx.config.symbols

    # model_data_repository = ctx.model_data_repository
    # symbols_metadata = model_data_repository.symbols_metadata
    # external_model_weights = model_data_repository.external_model_weights
    # weights_by_date_symbol = model_data_repository.weights_by_date_symbol
    # current_date_weights = model_data_repository.current_date_weights
    # wts_for_date = model_data_repository.get_weights_by_date(DateTime.Today)

    # oms = ctx.oms
    # # Dictionary<string, Dictionary<string, double>
    # pos_by_system_by_symbol = oms.position_by_system_and_symbol

    # # Dictionary<string, List<IOrder>>
    # orders_by_symbol = oms.orders_by_symbol

    # market_data_loader = ctx.market_data_loader

    # # example for getting bars, result type is  Dictionary<string, List<Bar>>
    # barsBySymbol = market_data_loader.get_bars()

    pl_manager = ctx.pl_manager
    # pl_by_date is dictionary <string, PLEntry>

    # total PL
    #  can get pl_compounded, dollarPL as well, if required
    pnl = pd.DataFrame(
        [(x.Key.ToString(), x.Value.pl) for x in pl_manager.pl_by_date]
    ).set_index(0)
    pnl.index = pd.to_datetime(pnl.index)

    pnl_list = []
    for sy in symbols:
        pl_manager.process_include(
            sy
        )  # this line re-processes PL for a single symbol - overwrites pl_by_date
        frame = pd.DataFrame(
            [(x.Key.ToString(), x.Value.pl) for x in pl_manager.pl_by_date]
        ).set_index(0)
        frame.index = pd.to_datetime(frame.index)
        frame.columns = [sy]
        pnl_list.append(frame)

    pnl_by_symbol = pd.concat(pnl_list, axis=1)
    pnl_by_symbol["total_pnl"] = pnl  # Add total pnl to pnl by symbol

    return pnl_by_symbol


def main(model_id=0, update_qdeck=0, live=0, config=None):
    mlModelRunner = MLModelRunner(net_logger, runner_config)

    runId = mlModelRunner.run_model(model_id, update_qdeck, live, config)

    # # pnl by symbol
    # pnl_by_symbol = f_pnl_by_symbol(mlModelRunner)

    return runId
