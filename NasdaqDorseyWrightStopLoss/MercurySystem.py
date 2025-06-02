import numpy as np
import pandas as pd
import datetime
import ta
import json
import os, sys, inspect

code_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
project_dir = os.path.dirname(code_dir)
sys.path.insert(0, project_dir)

from System import DateTime  # type: ignore
from Mercury import MercuryRunner, MercuryRunConfig, IMercurySystem, PriceField  # type: ignore

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


class DorseyWrightWeightRebalanceTopLossSystem(IMercurySystem):
    __namespace__ = "Mercury"

    def __init__(self, cfg):
        self.name = "DorseyWrightWeightRebalanceStopLoss"

        self.indexes = []
        self.symbols = cfg.symbols

        ## For stoploss
        self.previous_month = None
        self.initial_price_for_stoploss_d = {}
        # self.included_last_month = []
        self.previously_included = []
        # self.previous_positions = {}

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

    def update_stoploss_initial_price_dict(self, sy):
        ## If we are buying and there is not already an initial price, then add the open price as initial price
        ## When we sell then we will remove the symbol

        if sy not in self.initial_price_for_stoploss_d.keys():
            open = self.contextHolder.market_data_loader.get_price_field(
                PriceField.Open
            )
            open = {k: to_numpy(open[k]) for k in sorted(open.Keys)}
            self.initial_price_for_stoploss_d[sy] = open[sy][0]

    def compute_symbols_included_after_trades(self):
        ## Previously we used the positions from the previous day
        ## As the rebalnaces tend to happen on the 1st of the month, along with the stoploss,
        ## then using the previous days price means a mismatch to current portfolio
        ## Instead we are calculating position after todays trades
        ## As all we need is included vs excluded, thats all we calculate
        ## !!! We assume that if in self.symbols_weights_to_enter_d, that the symbol is included !!!

        self.currently_included = []

        for sy in self.contextHolder.config.symbols:
            current_position = self.contextHolder.oms.get_position_by_system_and_symbol(
                self.name, sy
            )
            currently_included = current_position > 0  # has position

            if (
                sy in self.symbols_weights_to_enter_d.keys()
            ):  ## I believe symbol has to be included in this case
                currently_included = True

            if sy in self.symbols_to_exit:
                currently_included = False

            if currently_included:
                self.currently_included.append(sy)

    def stop_loss_calculation(self):
        ## If price is below stop loss, then put in cash for the next month. Re-enter after
        ## We have the initial_price_for_stoploss_d dict. It will start blank and we will add to it
        ## We will remove if symbol exited

        self.removed_due_to_stoploss_l = []  ## Reset each month

        open = self.contextHolder.market_data_loader.get_price_field(PriceField.Open)
        open = {k: to_numpy(open[k]) for k in sorted(open.Keys)}

        for sy in self.currently_included:
            ## Currently included and included at start of last month and not exited since, then run stoploss
            if (sy in self.previously_included) & (
                sy in self.initial_price_for_stoploss_d.keys()
            ):
                current_price = open[sy][0]
                entry_price = self.initial_price_for_stoploss_d[sy]

                percent_change = (current_price / entry_price) - 1

                # if True:
                if (
                    self.contextHolder.referenceData.config.stop_loss_percent > 0
                    and percent_change
                    < -self.contextHolder.referenceData.config.stop_loss_percent
                ):
                    print(str(self.current_date))
                    print(sy)
                    self.removed_due_to_stoploss_l.append(sy)

                    del self.initial_price_for_stoploss_d[sy]
                    self.currently_included.remove(
                        sy
                    )  # so that it is removed when compared with next month (as the previous month)

        self.previously_included = self.currently_included

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
        self.symbols_to_exit = []
        self.symbols_weights_to_enter_d = {}

        today = DateTime.Today

        # if self.current_date.ToString("yyyy-MM-dd") == "2025-01-01":
        #     print(self.current_date.ToString("yyyy-MM-dd"))

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
                        self.symbols_to_exit.append(sm)

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

        if trade:
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
                        self.symbols_weights_to_enter_d[symbol] = current_date_weights[
                            symbol
                        ].weight

        # entry order for symbols with no position from previous order
        if not trade:
            if len(current_date_weights) > 0 and len(pos_by_symbol) > 0:
                for symbol in current_date_weights.keys():
                    pos = 0
                    if symbol in pos_by_symbol.keys():
                        pos = pos_by_symbol[symbol]

                    if (
                        pos == 0
                        and symbol not in self.symbols_to_exit
                        and symbol
                        in self.contextHolder.market_data_loader.StartAndEndBySymbol
                        and self.contextHolder.market_data_loader.StartAndEndBySymbol[
                            symbol
                        ].Item1
                        <= self.current_date
                    ):
                        self.symbols_weights_to_enter_d[symbol] = current_date_weights[
                            symbol
                        ].weight

        ## NORMAL EXITS
        if len(self.symbols_to_exit) > 0:
            for xs in self.symbols_to_exit:
                self.contextHolder.oms.add_weight_order(self.name, xs, 0)

                ### Remove from stoploss dict
                if (
                    xs in self.initial_price_for_stoploss_d.keys()
                ):  ## Helps with ordering of code
                    del self.initial_price_for_stoploss_d[xs]

        #### STOP LOSS LOGIC  ####
        ## Moved here as want to factor proposed days trades into positions

        self.stoploss_trade_day = False

        ## CHECK IF STOPLOSS TRIGGERED if new month then check for stop loss trigger
        if self.current_date.Month != self.previous_month:
            self.stoploss_trade_day = True

            self.compute_symbols_included_after_trades()  # On stoploss days we want to first get position after current days trades
            self.stop_loss_calculation()
            self.previous_month = self.current_date.Month

        ## NORMAL ENTERS - If stop-loss triggered then we won't rebalance as usual, so above needed first
        ## Normal enters
        for sy, weight in self.symbols_weights_to_enter_d.items():
            if (
                sy not in self.removed_due_to_stoploss_l
            ):  # if in stoploss list then do not add it in during the month
                self.contextHolder.oms.add_weight_order(self.name, sy, weight)

                self.update_stoploss_initial_price_dict(sy)

        ## STOP-LOSS EXITS
        ## Ensures that if stoploss is hit then sy will remain at zero weight until the next month
        if self.stoploss_trade_day:
            for sy in self.removed_due_to_stoploss_l:
                if sy not in self.symbols_to_exit:  # symbols already exited
                    self.contextHolder.oms.add_weight_order(self.name, sy, 0)

                    if (
                        sy in self.initial_price_for_stoploss_d.keys()
                    ):  ## Helps with ordering of code
                        del self.initial_price_for_stoploss_d[xs]

    def run(self):
        pass

    def set_equity(self):
        pass


class NDWWeightRebalanceStopLossConfig(MercuryRunConfig):
    __namespace__ = "Mercury"

    def __init__(self, cfg=None):
        super().__init__()

        self.name = "DorseyWrightWeightRebalanceStopLoss"
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

            self.systems = [DorseyWrightWeightRebalanceTopLossSystem(self)]


class NDWStopLossModelRunner(MercuryRunner):
    __namespace__ = "Mercury"

    def __init__(self):
        super().__init__()

    def run_model(self, model_id=0, update_qdeck=0, live=0, config=None):
        # print(model_id)

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

        run_config = NDWWeightRebalanceStopLossConfig(cfg_data)
        run_config.run_time_pst = "100000000"

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
    dwslModelRunner = NDWStopLossModelRunner()

    runId = dwslModelRunner.run_model(model_id, update_qdeck, live, config)

    return runId
