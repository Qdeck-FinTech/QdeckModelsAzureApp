import numpy as np
import pandas as pd
import datetime
import ta
import json
import os, sys, inspect
import itertools
from collections import deque, defaultdict

code_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
project_dir = os.path.dirname(code_dir)
sys.path.insert(0, project_dir)

from System import DateTime  # type: ignore
from Mercury import MercuryRunner, MercuryRunConfig, IMercurySystem  # type: ignore

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


class NDWStopLossSystem(IMercurySystem):
    __namespace__ = "Mercury"

    def __init__(self, cfg):
        self.name = "NDWStopLossSystem"

        self.indexes = []
        self.symbols = cfg.symbols

        ## For stop-loss
        self.active_position_fills = {}
        self.active_stop_loss = {}
        self.previous_month = None

        self.symbols_to_exit = []
        self.symbols_weights_to_enter_d = {}

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
        print(
            "order_filled: ",
            order.symbol,
            order.fill_date.ToString("yyyy-MM-dd"),
            order.quantity,
        )

        # store the latest date for the fill
        if order is not None:
            self.active_position_fills[order.symbol] = order.fill_date

    def eval_stop_loss(self, symbol, current_position):
        # ignore if current position = 0
        if current_position == 0:
            return False

        # ignore if already included in active_stop_loss
        if symbol in self.active_stop_loss.keys():
            return False

        # ignore if no active order
        if symbol not in self.active_position_fills.keys():
            return False

        # get open price - current date
        current_bar = self.contextHolder.market_data_loader.get_bars_by_symbol_date(
            symbol, self.current_date
        )

        # get open price on the date of the fill
        order_bar = self.contextHolder.market_data_loader.get_bars_by_symbol_date(
            symbol, self.active_position_fills[symbol]
        )

        if current_bar == None or order_bar == None:
            return False

        current_price = current_bar.open
        entry_price = order_bar.open

        # calc price change
        percent_change = (current_price / entry_price) - 1

        # print(
        #     "eval_stop_loss: ",
        #     self.current_date.ToString("yyyy-MM-dd"),
        #     order_bar.trade_date.ToString("yyyy-MM-dd"),
        #     symbol,
        #     percent_change,
        # )

        # if threshold is breached using config stop_loss_percent
        if (
            self.contextHolder.referenceData.config.stop_loss_percent > 0
            and percent_change
            < -self.contextHolder.referenceData.config.stop_loss_percent
        ):
            return True

        return False

    def run_close(self):
        pass

    def run_open(self):
        current_date_s = self.current_date.ToString("yyyy-MM-dd")
        # print("run_open: ", current_date_s)

        current_date_weights = (
            self.contextHolder.model_data_repository.current_date_weights
        )

        if len(current_date_weights) == 0:
            print("run_open - no weights available: ", current_date_s)
            return

        trade = False
        stop_loss_reset = False

        self.symbols_to_exit = []
        self.symbols_weights_to_enter_d = {}

        today = DateTime.Today

        # if current_date_s == "2025-03-11":
        #     print(current_date_s)

        if self.previous_month == None:
            self.previous_month = self.current_date.Month

        pos_by_symbol = {}
        if self.name in self.contextHolder.oms.position_by_system_and_symbol.keys():
            pos_by_symbol = self.contextHolder.oms.position_by_system_and_symbol[
                self.name
            ]

        # trade if there trade flag
        if len(current_date_weights) > 0:
            trade = any(w.trade for w in current_date_weights.values())

        # trade if no current position
        if not trade:
            if len(pos_by_symbol) == 0 and len(current_date_weights) > 0:
                trade = True

        # trade if existing positions (symbols) do not match new weights (symbols)
        if not trade:
            for ps in pos_by_symbol.keys():
                if pos_by_symbol[ps] != 0 and ps not in current_date_weights.keys():
                    trade = True

        # get symbols to exit, if last bar
        if (
            self.current_date < self.contextHolder.market_data_loader.max_date
            and self.current_date < self.contextHolder.referenceData.config.end
            and self.current_date < today
        ):
            for sm in self.contextHolder.market_data_loader.StartAndEndBySymbol.keys():
                if (
                    self.contextHolder.market_data_loader.StartAndEndBySymbol[sm].Item2
                    <= self.current_date
                ):
                    exitQty = self.contextHolder.oms.get_position_by_system_and_symbol(
                        self.name, sm
                    )
                    if exitQty != 0:
                        # add symbol to exit
                        self.symbols_to_exit.append(sm)

        # trade if missing symbols for position from previous day
        if not trade:
            if len(current_date_weights) > 0 and len(pos_by_symbol) > 0:
                for symbol in current_date_weights.keys():
                    pos = 0
                    if symbol in pos_by_symbol.keys():
                        pos = pos_by_symbol[symbol]

                    if (
                        pos == 0
                        and symbol not in self.symbols_to_exit
                        and symbol not in self.active_stop_loss.keys()
                        and symbol
                        in self.contextHolder.market_data_loader.StartAndEndBySymbol
                        and self.contextHolder.market_data_loader.StartAndEndBySymbol[
                            symbol
                        ].Item1
                        <= self.current_date
                    ):
                        trade = True
                        break

        # trade if symbols are the same but weights are rebalanced
        if not trade:
            if len(current_date_weights) > 0:
                rebal = 0
                for cs in current_date_weights.keys():
                    if (
                        round(
                            (
                                current_date_weights[cs].ideal_weight
                                - current_date_weights[cs].weight
                            ),
                            5,
                        )
                        <= 0.00001
                    ):
                        rebal += 1

                if rebal == len(current_date_weights):
                    trade = True

        # trade if it's a new month and there are active stop-loss
        if not trade and (
            self.current_date.Month != self.previous_month
            and len(self.active_stop_loss) > 0
        ):
            trade = True
            stop_loss_reset = True

        # stop-loss logic
        if not trade:
            if len(pos_by_symbol) > 0:
                for ps in pos_by_symbol.keys():
                    stop_loss_ext = self.eval_stop_loss(ps, pos_by_symbol[ps])
                    if stop_loss_ext:
                        # add symbol to exit
                        self.symbols_to_exit.append(ps)

                        # add symbol to active stop loss
                        self.active_stop_loss[ps] = self.current_date

                        # update the latest stop-loss date
                        self.stop_loss_date = self.current_date

                        print("stop loss: ", ps, current_date_s)

        if trade:
            # reset active stop-loss tracking
            self.active_stop_loss = {}

            if len(current_date_weights) > 0:
                # prev weight exit
                if len(pos_by_symbol) > 0:
                    for ps in pos_by_symbol.keys():
                        if pos_by_symbol[ps] != 0 and (
                            ps not in current_date_weights.keys()
                            or current_date_weights[ps].weight == 0
                        ):
                            self.symbols_to_exit.append(ps)

                # new weight entry
                for symbol in current_date_weights.keys():
                    if (
                        symbol not in self.symbols_to_exit
                        and symbol
                        in self.contextHolder.market_data_loader.StartAndEndBySymbol
                        and self.contextHolder.market_data_loader.StartAndEndBySymbol[
                            symbol
                        ].Item1
                        <= self.current_date
                    ):
                        # ideal weight if stop-loss reset rebalance
                        weight = (
                            current_date_weights[symbol].ideal_weight
                            if stop_loss_reset
                            else current_date_weights[symbol].weight
                        )
                        # add entry order
                        self.symbols_weights_to_enter_d[symbol] = weight

        ## add exit orders
        if len(self.symbols_to_exit) > 0:
            for xs in self.symbols_to_exit:
                self.contextHolder.oms.add_weight_order(self.name, xs, 0)

        ## add entry orders
        for sy, weight in self.symbols_weights_to_enter_d.items():
            self.contextHolder.oms.add_weight_order(self.name, sy, weight)

        # update previous month, if changed
        if self.current_date.Month != self.previous_month:
            self.previous_month = self.current_date.Month

    def run(self):
        pass

    def set_equity(self):
        pass


class NDWStopLossConfig(MercuryRunConfig):
    __namespace__ = "Mercury"

    def __init__(self, cfg=None):
        super().__init__()

        self.name = "NDWStopLoss"
        end_date = datetime.datetime.now()
        self.end = DateTime(end_date.year, end_date.month, end_date.day)

        self.stop_loss_percent = 0.1  # default value

        self.max_buffer_size = 400

        self.production = True
        self.publish_status = False

        self.oms_options.clear_orders_at_eod = True
        self.pl_options.compounded_pl = True
        self.pl_options.include_commission = False

        self.logging_options.weight_order_no_bars = True
        self.logging_options.cash_management = True

        self.output_folder = "output"

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

            self.start = DateTime(start_date.year, start_date.month, start_date.day)
            self.equity = initial_equity

            self.symbols = list(cfg["symbols"])

            if isinstance(cfg["external_model_id"], int):
                external_model_id = int(cfg["external_model_id"])
                self.external_model_id = external_model_id

            stop_loss_percent = 10  # 10% - default
            if "StopLossPercent" in cfg["parameters"] and (
                isinstance(cfg["parameters"]["StopLossPercent"], str)
                or isinstance(cfg["parameters"]["StopLossPercent"], int)
            ):
                stop_loss_percent = int(cfg["parameters"]["StopLossPercent"])

            self.stop_loss_percent = (
                stop_loss_percent / 100
            )  # to get e.g. 10% to 0.1 ie

            self.systems = [NDWStopLossSystem(self)]


class NDWStopLossRunner(MercuryRunner):
    __namespace__ = "Mercury"

    def __init__(self):
        super().__init__()

    def run_model(self, model_id=0, update_qdeck=0, live=0, config=None):
        cfg_data = None
        symbols_metadata_json = None

        if model_id > 0:
            # load configuration from database
            cfg_data_json = self.load_model_config(str(model_id))

            if cfg_data_json is not None:
                cfg_data = json.loads(cfg_data_json)
                symbols_metadata_json = cfg_data["symbols_metadata"]

        elif config is not None:
            # load configuration from file, if provided
            with open(config) as json_data:
                cfg_data = json.load(json_data)
                symbols_metadata_json = json.dumps(cfg_data["symbols_metadata"])
                cfg_data["symbols"] = list(cfg_data["symbols_metadata"].keys())

        # set symbols_metadata
        self.add_metadata_from_json(symbols_metadata_json)

        run_config = NDWStopLossConfig(cfg_data)

        if model_id > 0:
            run_config.model_id = model_id

            if update_qdeck > 0:
                run_config.update_qdeck = True

            run_config.production = True

        if live > 0:
            run_config.goLive = True

        runId = self.run(run_config)

        print("completed! run id: " + str(runId))

        return runId, run_config


def main(model_id=0, update_qdeck=0, live=0, config=None):
    ndwslModelRunner = NDWStopLossRunner()

    runId = ndwslModelRunner.run_model(model_id, update_qdeck, live, config)

    return runId
