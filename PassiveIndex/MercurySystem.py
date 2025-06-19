import numpy as np
import pandas as pd
import datetime
import ta
import json
import os, sys, inspect
import itertools
from collections import deque, defaultdict
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


class PassiveExposure(IMercurySystem):
    __namespace__ = "Mercury"

    def __init__(self, run_config):
        self.name = "PassiveExposure"

        self.indexes = []
        self.symbols = run_config.symbols
        self.next_rebal_date = run_config.next_rebal_date
        self.prev_rebal_date = run_config.next_rebal_date

        self.cfg = run_config

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

    def set_position_by_weight(self, symbols, weights):
        for symbol, weight in zip(symbols, weights):
            self.contextHolder.oms.add_weight_order(self.name, symbol, weight)

    def update_rebalance_date(self):
        # to trade on the first day of the desired rebalance_time_scale, date returned is the last trade day (ignoring holidays).
        temp_date = self.current_date
        if self.cfg.rebalance_time_scale == "M":
            # will return last trade day of the month
            while temp_date.Day > 20:
                # hacky get to the first of the month
                temp_date = temp_date.AddDays(1)
            while 1 < temp_date.Day and temp_date.Day < 10:
                # hacky get to the first of the month
                temp_date = temp_date.AddDays(-1)
            targetMonth = temp_date.Month + self.cfg.rebalance_freq  # in range 2 to 12+
            # Calculate the target year
            targetYear = temp_date.Year + (targetMonth - 1) // 12  #
            # Find the first day of the month
            self.next_rebal_date = DateTime(targetYear, (targetMonth - 1) % 12 + 1, 1)
            # Shift index back one day to trade on first day of month
            self.next_rebal_date = self.next_rebal_date.AddDays(-1)
            while int(self.next_rebal_date.DayOfWeek) > 5:
                self.next_rebal_date = self.next_rebal_date.AddDays(-1)

        if self.cfg.rebalance_time_scale == "W":
            # will return friday of the week
            if int(temp_date.DayOfWeek) != 5:
                days_to_add = (
                    7 * (self.cfg.rebalance_freq - 1) + 5 - int(temp_date.DayOfWeek)
                )  # if first week is not friday
            else:
                days_to_add = 7 * self.cfg.rebalance_freq
            self.next_rebal_date = temp_date.AddDays(days_to_add)

    def run(self):
        if self.current_date >= self.next_rebal_date:
            logging.info("rebalance:" + str(self.next_rebal_date))
            self.update_rebalance_date()
            self.set_position_by_weight(self.cfg.symbols, self.cfg.weights)

    def run_open(self):
        pass

    def run_close(self):
        pass

    def set_equity(self):
        pass

    def order_filled(self, order):
        pass


class EquityFixedIncomePassiveConfig(MercuryRunConfig):
    __namespace__ = "Mercury"

    def __init__(self, cfg=None, max_buffer=1):
        super().__init__()

        self.require_data_from_all_symbols = True
        self.pl_options.include_commission = False
        self.name = "PassiveExposure"
        self.max_buffer_size = max_buffer

        self.save_stats_to_file = False
        self.save_outputs_to_file = False
        self.output_folder = "output"

        self.production = False
        self.publish_status = False

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
            start_date = pd.to_datetime(cfg["parameters"]["Start_Date"])
            initial_equity = float(cfg["parameters"]["Initial_Equity"])
            rebalance = cfg["parameters"]["Rebalance"]
            compounding = cfg["parameters"]["Compounding"]

            self.start = DateTime(start_date.year, start_date.month, start_date.day)
            self.equity = initial_equity

            compounded_pl = False
            if isinstance(compounding, str):
                compounded_pl = (
                    cfg["parameters"]["Compounding"].lower().capitalize() == "True"
                )
            else:
                compounded_pl = compounding

            self.pl_options.compounded_pl = compounded_pl

            #### Section uses existing config style.  Should update###
            sym_wts = cfg["symbol_weights"]
            sector_wts = cfg["category_weights"]

            sector_wts = {
                key: float(sector_wts[key]) if sector_wts[key] != None else 0
                for key in sector_wts.keys()
            }

            for sector in sym_wts.keys():
                sym_wts[sector] = {
                    key: (
                        float(sym_wts[sector][key])
                        if sym_wts[sector][key] != None
                        else 0
                    )
                    for key in sym_wts[sector].keys()
                }

            weights = {"Weights": {}}

            for (
                val
            ) in sym_wts:  # multiplies sector weight by symbol to get actual weight
                abs_weight = {
                    key: sym_wts[val][key] * sector_wts[val]
                    for key in sym_wts[val].keys()
                }
                weights["Weights"].update(abs_weight)

            self.symbols = self.get_tickers(cfg["symbols"])  # weights["Weights"].keys()
            weights = np.array(list(weights["Weights"].values()))

            if np.sum(weights) > 1:
                weights = weights / np.sum(weights)
            if np.any(weights < 0):
                raise Exception("Currently, no short selling allowed")
            self.weights = weights

            #######################################
            self.rebalance_time_scale = "M"  # default with monthly, weekly will update
            if rebalance == "Monthly":
                self.rebalance_freq = 1
            elif rebalance == "Quarterly":
                self.rebalance_freq = 3
            elif rebalance == "Every 6 Months":
                self.rebalance_freq = 6
            elif rebalance == "Annually":
                self.rebalance_freq = 12
            elif rebalance == "Never":
                self.rebalance_freq = 12 * 100  # (so really every 100 years)
            elif rebalance == "Weekly":
                self.rebalance_freq = 1
                self.rebalance_time_scale = "W"

            else:
                raise Exception(
                    "Rebalance is not in [Weekly, Monthly, Quarterly, Every 6 Months, Annually]"
                )

            self.next_rebal_date = self.start
            self.systems = [PassiveExposure(self)]

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


class PassiveIndexModelRunner(MercuryRunner):
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
                cfg_data["symbols"] = list(cfg_data["symbols"])

        # build system configuration
        run_config = EquityFixedIncomePassiveConfig(cfg_data)

        if model_id > 0:
            run_config.model_id = model_id

            if update_qdeck > 0:
                run_config.update_qdeck = True

            run_config.production = True

        if live > 0:
            run_config.goLive = True

        # run simulation
        runId = self.run(run_config)

        logging.info("completed! run id: " + str(runId))

        return runId


def main(model_id=0, update_qdeck=0, live=0, config=None):
    runner_config = QdeckModelRunnerConfiguration().get_net_config()
    piModelRunner = PassiveIndexModelRunner(net_logger, runner_config)

    runId = piModelRunner.run_model(model_id, update_qdeck, live, config)

    return runId
