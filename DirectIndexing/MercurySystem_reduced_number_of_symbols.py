import numpy as np
import pandas as pd
import datetime
import ta
import json
import os, sys, inspect
import logging

code_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
project_dir = os.path.dirname(code_dir)
sys.path.insert(0, project_dir)

import clr

from System import DateTime  # type: ignore
from System.Collections.Generic import List  # type: ignore

clr.AddReference("Mercury")
from Mercury import MercuryRunner, MercuryRunConfig, IMercurySystem, TickerInfo  # type: ignore
from logger.net_logger import net_logger
from configuration.configuration import QdeckModelRunnerConfiguration

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


class DirectIndexingSystem(IMercurySystem):
    __namespace__ = "Mercury"

    def __init__(self, cfg):
        self.name = "DirectIndexing_reduced_number_of_symbols"

        self.indexes = []
        self.symbols = cfg.symbols

        self.cfg = cfg
        self.current_date = cfg.start
        self.previous_days_month = 0

        self.historic_symbols_list = []

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

    def run_close(self):
        pass

    def is_rebalance_date(self):
        ## Currently done as monthly, start of month, for simplicity
        if self.cfg.rebalance == "Monthly":
            if self.current_date.Month != self.previous_days_month:
                self.previous_days_month = self.current_date.Month
                return True
        else:
            return True

        return False

    def Calculate_weights_reduced_number_of_symbols(self, current_date_weights):
        ## Get list of all symbols since start of model
        ## Was issue where historic symbols were not removed. This forces their removal.
        current_symbols_list = current_date_weights.keys()
        self.historic_symbols_list += current_symbols_list
        self.historic_symbols_list = list(set(self.historic_symbols_list))

        sort_tuple_list = sorted(
            current_date_weights.items(), key=lambda x: x[1], reverse=True
        )  # Largest to smallest value (list of tuples)

        ## Find largest weighted symbols, and scale so sum to 100%

        reduced_sort_dict = {
            key: val for key, val in sort_tuple_list[: self.cfg.number_of_symbols]
        }
        total_weight = sum(reduced_sort_dict.values())

        ## Reset weights to zero default for current_date_weights
        current_date_weights = {}

        for sy in self.historic_symbols_list:
            if sy in reduced_sort_dict.keys():
                current_date_weights[sy] = reduced_sort_dict[sy] / total_weight

            else:
                current_date_weights[sy] = 0

        return current_date_weights

    def run_open(self):
        current_date_s = self.current_date.ToString("yyyy-MM-dd")
        # logging.info("run_open: "+ current_date_s)

        current_date_weights = (
            self.contextHolder.model_data_repository.current_date_weights
        )

        current_date_weights = {k: v.weight for k, v in current_date_weights.items()}

        if len(current_date_weights) == 0:
            logging.info("run_open - no weights available: " + current_date_s)
            return

        trade = self.is_rebalance_date()
        symbols_to_exit = []
        today = DateTime.Today

        if trade:
            logging.info("Rebalance: " + self.current_date.ToString("yyyy-MM-dd"))

            current_date_weights = self.Calculate_weights_reduced_number_of_symbols(
                current_date_weights
            )

            for symbol in current_date_weights:
                self.contextHolder.oms.add_weight_order(
                    # RA: self.name, symbol, current_date_weights[symbol].weight
                    self.name,
                    TickerInfo.Deserealize(symbol),
                    current_date_weights[symbol],
                )

        else:
            current_date_weights = {}

        # if self.current_date.ToString("yyyy-MM-dd") == "2025-01-01":
        #     logging.info(self.current_date.ToString("yyyy-MM-dd"))

        pos_by_symbol = {}
        if self.name in self.contextHolder.oms.position_by_system_and_symbol.keys():
            pos_by_symbol = self.contextHolder.oms.position_by_system_and_symbol[
                self.name
            ]

        if False:
            # trade if no current position
            if not trade:
                if len(pos_by_symbol) == 0 and len(current_date_weights) > 0:
                    trade = True

        if False:
            # trade if existing positions (symbols) do not match new weights (symbols)
            if not trade:
                for ps in pos_by_symbol.keys():
                    if pos_by_symbol[ps] != 0 and ps not in current_date_weights.keys():
                        trade = True

        if True:
            # get symbols to exit, if last bar
            if (
                self.current_date < self.contextHolder.market_data_loader.max_date
                and self.current_date < self.contextHolder.referenceData.config.end
                and self.current_date < today
            ):
                for (
                    sm
                ) in self.contextHolder.market_data_loader.StartAndEndBySymbol.keys():
                    if (
                        self.contextHolder.market_data_loader.StartAndEndBySymbol[
                            sm
                        ].Item2
                        <= self.current_date
                    ):
                        exitQty = (
                            self.contextHolder.oms.get_position_by_system_and_symbol(
                                self.name, sm
                            )
                        )
                        if exitQty != 0:
                            symbols_to_exit.append(sm)

            # # trade if symbols are the same but weights are rebalanced
            # if not trade:
            #     if len(current_date_weights) > 0:
            #         rebal = 0
            #         for cs in current_date_weights.keys():
            #             if (
            #                 round(
            #                     (
            #                         current_date_weights[cs].ideal_weight
            #                         - current_date_weights[cs].weight
            #                     ),
            #                     5,
            #                 )
            #                 <= 0.00001
            #             ):
            #                 rebal += 1

            #         if rebal == len(current_date_weights):
            #             trade = True

        if False:
            if trade:
                if len(current_date_weights) > 0:
                    # prev weight exit
                    if len(pos_by_symbol) > 0:
                        for ps in pos_by_symbol.keys():
                            # logging.info(current_date_weights[ps].weight)

                            if pos_by_symbol[ps] != 0 and (
                                ps not in current_date_weights.keys()
                                # RA: or current_date_weights[ps].weight == 0
                                or current_date_weights[ps] == 0
                            ):
                                symbols_to_exit.append(ps)

                    # new weight entry

                    logging.info(len(current_date_weights))
                    for symbol in current_date_weights.keys():
                        if (
                            symbol not in symbols_to_exit
                            and symbol
                            in self.contextHolder.market_data_loader.StartAndEndBySymbol
                            and self.contextHolder.market_data_loader.StartAndEndBySymbol[
                                symbol
                            ].Item1
                            <= self.current_date
                        ):
                            self.contextHolder.oms.add_weight_order(
                                # RA: self.name, symbol, current_date_weights[symbol].weight
                                self.name,
                                TickerInfo.Deserealize(symbol),
                                current_date_weights[symbol],
                            )

            # entry order for symbols with no position from previous order
            if not trade:
                if len(current_date_weights) > 0 and len(pos_by_symbol) > 0:
                    for symbol in current_date_weights.keys():
                        pos = 0
                        if symbol in pos_by_symbol.keys():
                            pos = pos_by_symbol[symbol]

                        if (
                            pos == 0
                            and symbol not in symbols_to_exit
                            and symbol
                            in self.contextHolder.market_data_loader.StartAndEndBySymbol
                            and self.contextHolder.market_data_loader.StartAndEndBySymbol[
                                symbol
                            ].Item1
                            <= self.current_date
                        ):
                            #####  RA: FIX IN THIS CASE
                            self.contextHolder.oms.add_weight_order(
                                # RA: self.name, symbol, current_date_weights[symbol].weight
                                self.name,
                                TickerInfo.Deserealize(symbol),
                                current_date_weights[symbol],
                            )
        if True:
            if len(symbols_to_exit) > 0:
                for xs in symbols_to_exit:
                    self.contextHolder.oms.add_weight_order(
                        self.name, TickerInfo.Deserealize(xs), 0
                    )

    def run(self):
        pass

    def set_equity(self):
        pass


class DirectIndexingConfig(MercuryRunConfig):
    __namespace__ = "Mercury"

    def __init__(self, cfg=None):
        super().__init__()
        self.name = "DirectIndexing_reduced_number_of_symbols"

        end_date = datetime.datetime.now()
        self.end = DateTime(end_date.year, end_date.month, end_date.day)

        self.max_buffer_size = 400

        self.production = True
        self.publish_status = False

        self.oms_options.clear_orders_at_eod = True
        self.pl_options.compounded_pl = True
        self.pl_options.include_commission = False

        self.logging_options.weight_order_no_bars = True

        self.save_stats_to_file = False
        self.save_outputs_to_file = False
        self.output_folder = "output"

        self.number_of_symbols = 20

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
            # print(cfg["parameters"])

            start_date = pd.to_datetime(cfg["parameters"]["Start_Date"])
            initial_equity = float(cfg["parameters"]["Initial_Equity"])

            self.start = DateTime(start_date.year, start_date.month, start_date.day)
            self.equity = initial_equity

            self.symbols = self.get_tickers(cfg["symbols"])

            self.rebalance = "Monthly"
            if "Rebalance" in cfg["parameters"] and (
                isinstance(cfg["parameters"]["Rebalance"], str)
            ):
                self.rebalance = cfg["parameters"]["Rebalance"]

            if isinstance(cfg["direct_index_id"], int):
                direct_index_id = int(cfg["direct_index_id"])
                self.direct_index_id = direct_index_id

            self.systems = [DirectIndexingSystem(self)]

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


class DirectIndexingModelRunner(MercuryRunner):
    __namespace__ = "Mercury"

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
                # cfg_data["symbols"] = list(cfg_data["symbols"])

        run_config = DirectIndexingConfig(cfg_data)

        run_config.run_time_pst = "100000000"

        if model_id > 0:
            run_config.model_id = model_id

            if update_qdeck > 0:
                run_config.update_qdeck = True

            run_config.production = True

        if live > 0:
            run_config.goLive = True

        # # dev - test
        # run_config.production_dont_set_end_date = True
        # end_date = datetime.datetime.now()
        # run_config.end = DateTime(end_date.year, end_date.month, end_date.day - 2)

        runId = self.run(run_config)

        logging.info("completed! run id: " + str(runId))

        return runId


def main(model_id=0, update_qdeck=0, live=0, config=None):
    runner_config = QdeckModelRunnerConfiguration().get_net_config()

    diModelRunner = DirectIndexingModelRunner(net_logger, runner_config)

    runId = diModelRunner.run_model(model_id, update_qdeck, live, config)

    return runId
