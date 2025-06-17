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
from Mercury import MercuryRunner, MercuryRunConfig, IMercurySystem, TickerInfo  # type: ignore


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


class KeebeckMultiStrategy(IMercurySystem):
    __namespace__ = "Mercury"

    def __init__(self, cfg):
        self.name = "KeebeckMultiStrategy"

        self.cfg = cfg

        self.current_date = cfg.start

        self.indexes = []
        self.symbols = cfg.symbols
        self.previous_days_month = self.current_date.Month
        self.rebalance = True

        self.symbols_weights_d = {}
        self.set_weights()

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

    def update_rebalance_date(self):
        ## Currently done as monthly, start of month, for simplicity
        if self.cfg.rebalance_time_scale == "M":
            if self.current_date.Month != self.previous_days_month:
                self.rebalance = True
                self.previous_days_month = self.current_date.Month

        else:
            raise ValueError("Unsuitable rebalance period")

    def set_weights(self):
        self.symbols_weights_d = self.cfg.symbols_weights_d

    def run(self):
        self.update_rebalance_date()

        if self.rebalance:
            print("Rebalance: ", self.current_date.ToString())

            for symbol, weight in self.symbols_weights_d.items():
                if weight > 0:
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


class KeebeckMultiStrategyConfig(MercuryRunConfig):
    __namespace__ = "Mercury"

    def __init__(self, cfg=None, max_buffer=201, save_outputs=True):
        super().__init__()

        self.end = DateTime.Today.AddDays(-1)
        self.max_buffer_size = max_buffer
        self.save_outputs = save_outputs

        self.require_data_from_all_symbols = False
        self.pl_options.include_commission = False
        self.pl_options.compounded_pl = True

        self.name = "KeebeckMultiStrategy"

        self.save_stats_to_file = False
        self.save_outputs_to_file = False
        self.output_folder = "output"

        self.production = False
        self.publish_status = False

        self.approach = "Keebeck"

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

            portfolio = cfg["Portfolio"]
            self.portfolio = portfolio

            self.strategies = portfolio["Strategies"]

            symbols = []
            symbols_weights_d = {}
            for (
                strategy_name
            ) in self.strategies:  # Get list of symbols, and dict of {sy:weight}
                strategy = self.strategies[strategy_name]

                strategy_symbols = strategy["symbols"]

                strategy_symbol_tickers = [
                    s for s in cfg["symbols"] if s["ticker"] in strategy_symbols
                ]

                existing = [s["ticker"] for s in symbols]
                for s in strategy_symbol_tickers:
                    if s["ticker"] not in existing:
                        symbols.append(s)
                        existing.append(s["ticker"])

                strategy_tickers = self.get_tickers(strategy_symbol_tickers)

                for ticker in strategy_tickers:
                    sy = ticker.ToString()
                    w = strategy["weights"].get(ticker.ticker, 0)

                    if sy in symbols_weights_d:
                        symbols_weights_d[sy] += w
                    else:
                        symbols_weights_d[sy] = w

            self.symbols = self.get_tickers(symbols)
            self.symbols_weights_d = symbols_weights_d

            self.rebalance = cfg["Rebalance"]
            if self.rebalance == "Monthly":
                self.rebalance_time_scale = "M"  # Simplified with only monthly for now. Can add quarterly in time.
            else:
                raise ValueError("Unsuitable rebalance period")

            self.next_rebal_date = self.start
            self.systems = [KeebeckMultiStrategy(self)]

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


class KeebeckMultiStrategyModelRunner(MercuryRunner):
    __namespace__ = "Mercury"

    def __init__(self):
        super().__init__()

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

        runId = 0

        if cfg_data is not None:
            # build system configuration
            run_config = KeebeckMultiStrategyConfig(cfg_data, 201, False)

            if model_id > 0:
                run_config.model_id = model_id

                run_config.update_qdeck = True
                run_config.production = True

            if live > 0:
                run_config.goLive = True

            # run simulation
            runId = self.run(run_config)

            print("completed! run id: ", runId)
        else:
            print("fail to load configuration file! model_id: ", model_id)

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
    kmsModelRunner = KeebeckMultiStrategyModelRunner()

    runId = kmsModelRunner.run_model(model_id, update_qdeck, live, config)

    # # pnl by symbol
    # pnl_by_symbol = f_pnl_by_symbol(mlModelRunner)

    return runId
