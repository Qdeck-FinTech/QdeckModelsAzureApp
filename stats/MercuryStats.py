import numpy as np
import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta

from System import DateTime
from Mercury import StatisticID, IMercuryStat

from stats.performance import NUM_DAYS_IN_YEAR
from stats.summary_calcs import calc_best_day, calc_worst_day, calc_std_d, calc_sharpe_ratio, \
    calc_annualized_return, calc_mv_av, calc_max_draw_down, calc_draw_down_divided_by_vol, \
    calc_current_draw_down, calc_long_short_ratio, calc_cagr


class BestDayStat(IMercuryStat):
    __namespace__ = 'Mercury'

    def __init__(self):
        super().__init__()
        self.id = int(StatisticID.BEST_DAY)
        self.name = "Best Day"
        self.value = 0


    def get_contextHolder(self):
        return self.contextHolder
    def set_contextHolder(self, contextHolder):
        self.contextHolder = contextHolder


    def get_name(self):
        return self.name
    def set_name(self, name):
        self.name = name

    def get_id(self):
        return self.id
    def set_id(self, id):
        self.id = id

    def get_value(self):
        return self.value
    def set_value(self, value):
        self.value = value


    def process(self):
        pl = self.pnl(self.contextHolder.config.pl_options.compounded_pl)
        self.value = calc_best_day(pl)
        return True

    def pnl(self, compounded = False):
        if compounded:
            pl = pd.DataFrame([(x.Key.ToString("yyyy-MM-dd"),x.Value.pl_compounded) for x in self.contextHolder.pl_manager.pl_by_date]).set_index(0)
        else:
            pl = pd.DataFrame([(x.Key.ToString("yyyy-MM-dd"),x.Value.pl) for x in self.contextHolder.pl_manager.pl_by_date]).set_index(0)

        pl.index = pd.to_datetime(pl.index, format='%Y-%m-%d')
        return pl  



class WorstDayStat(IMercuryStat):
    __namespace__ = 'Mercury'

    def __init__(self):
        super().__init__()
        self.id = int(StatisticID.WORST_DAY)
        self.name = "Worst Day"
        self.value = 0


    def get_contextHolder(self):
        return self.contextHolder
    def set_contextHolder(self, contextHolder):
        self.contextHolder = contextHolder


    def get_name(self):
        return self.name
    def set_name(self, name):
        self.name = name

    def get_id(self):
        return self.id
    def set_id(self, id):
        self.id = id

    def get_value(self):
        return self.value
    def set_value(self, value):
        self.value = value


    def process(self):
        pl = self.pnl(self.contextHolder.config.pl_options.compounded_pl)
        self.value = calc_worst_day(pl)
        return True


    def pnl(self, compounded = False):
        if compounded:
            pl = pd.DataFrame([(x.Key.ToString("yyyy-MM-dd"),x.Value.pl_compounded) for x in self.contextHolder.pl_manager.pl_by_date]).set_index(0)
        else:
            pl = pd.DataFrame([(x.Key.ToString("yyyy-MM-dd"),x.Value.pl) for x in self.contextHolder.pl_manager.pl_by_date]).set_index(0)

        pl.index = pd.to_datetime(pl.index, format='%Y-%m-%d')
        return pl  




class StandardDeviationDayStat(IMercuryStat):
    __namespace__ = 'Mercury'

    def __init__(self):
        super().__init__()
        self.id = int(StatisticID.STD_DAY)
        self.name = "Std (d)"
        self.value = 0


    def get_contextHolder(self):
        return self.contextHolder
    def set_contextHolder(self, contextHolder):
        self.contextHolder = contextHolder


    def get_name(self):
        return self.name
    def set_name(self, name):
        self.name = name

    def get_id(self):
        return self.id
    def set_id(self, id):
        self.id = id

    def get_value(self):
        return self.value
    def set_value(self, value):
        self.value = value


    def process(self):
        pl = self.pnl(self.contextHolder.config.pl_options.compounded_pl)
        self.value = calc_std_d(pl)
        return True


    def pnl(self, compounded = False):
        if compounded:
            pl = pd.DataFrame([(x.Key.ToString("yyyy-MM-dd"),x.Value.pl_compounded) for x in self.contextHolder.pl_manager.pl_by_date]).set_index(0)
        else:
            pl = pd.DataFrame([(x.Key.ToString("yyyy-MM-dd"),x.Value.pl) for x in self.contextHolder.pl_manager.pl_by_date]).set_index(0)

        pl.index = pd.to_datetime(pl.index, format='%Y-%m-%d')
        return pl   



class SharpeRatioDayStat(IMercuryStat):
    __namespace__ = 'Mercury'

    def __init__(self):
        super().__init__()
        self.id = int(StatisticID.SHARPE_DAY)
        self.name = "Sharpe (d)"
        self.value = 0


    def get_contextHolder(self):
        return self.contextHolder
    def set_contextHolder(self, contextHolder):
        self.contextHolder = contextHolder


    def get_name(self):
        return self.name
    def set_name(self, name):
        self.name = name

    def get_id(self):
        return self.id
    def set_id(self, id):
        self.id = id

    def get_value(self):
        return self.value
    def set_value(self, value):
        self.value = value


    def process(self):
        pl = self.pnl(self.contextHolder.config.pl_options.compounded_pl)

        risk_free_rate = self.get_risk_free_rate(pl.index[0], pl.index[len(pl)-1])

        self.value = calc_sharpe_ratio(pl,risk_free_rate, NUM_DAYS_IN_YEAR, self.contextHolder.config.pl_options.compounded_pl )     

        return True


    def pnl(self, compounded = False):
        if compounded:
            pl = pd.DataFrame([(x.Key.ToString("yyyy-MM-dd"),x.Value.pl_compounded) for x in self.contextHolder.pl_manager.pl_by_date]).set_index(0)
        else:
            pl = pd.DataFrame([(x.Key.ToString("yyyy-MM-dd"),x.Value.pl) for x in self.contextHolder.pl_manager.pl_by_date]).set_index(0)

        pl.index = pd.to_datetime(pl.index, format='%Y-%m-%d')
        return pl  


    def get_risk_free_rate(self, startDate=None, endDate=None):
        
        if startDate == None:
            startDate = self.contextHolder.config.start

        if endDate == None:
            endDate = self.contextHolder.config.end            


        start = DateTime(startDate.year,startDate.month,startDate.day)
        end = DateTime(endDate.year,endDate.month,endDate.day)

        return self.contextHolder.referenceData.load_risk_free_rate( start, end )



class AARStat(IMercuryStat):
    __namespace__ = 'Mercury'

    def __init__(self):
        super().__init__()
        self.id = int(StatisticID.AAR)
        self.name = "AAR"
        self.value = 0


    def get_contextHolder(self):
        return self.contextHolder
    def set_contextHolder(self, contextHolder):
        self.contextHolder = contextHolder


    def get_name(self):
        return self.name
    def set_name(self, name):
        self.name = name

    def get_id(self):
        return self.id
    def set_id(self, id):
        self.id = id

    def get_value(self):
        return self.value
    def set_value(self, value):
        self.value = value


    def process(self):
        pl = self.pnl(self.contextHolder.config.pl_options.compounded_pl)

        self.value = calc_annualized_return(pl, NUM_DAYS_IN_YEAR, self.contextHolder.config.pl_options.compounded_pl )     
        return True


    def pnl(self, compounded = False):
        if compounded:
            pl = pd.DataFrame([(x.Key.ToString("yyyy-MM-dd"),x.Value.pl_compounded) for x in self.contextHolder.pl_manager.pl_by_date]).set_index(0)
        else:
            pl = pd.DataFrame([(x.Key.ToString("yyyy-MM-dd"),x.Value.pl) for x in self.contextHolder.pl_manager.pl_by_date]).set_index(0)

        pl.index = pd.to_datetime(pl.index, format='%Y-%m-%d')
        return pl  



class MVAVStat(IMercuryStat):
    __namespace__ = 'Mercury'

    def __init__(self):
        super().__init__()
        self.id = int(StatisticID.MVAV)
        self.name = "MVAV"
        self.value = 0


    def get_contextHolder(self):
        return self.contextHolder
    def set_contextHolder(self, contextHolder):
        self.contextHolder = contextHolder


    def get_name(self):
        return self.name
    def set_name(self, name):
        self.name = name

    def get_id(self):
        return self.id
    def set_id(self, id):
        self.id = id

    def get_value(self):
        return self.value
    def set_value(self, value):
        self.value = value


    def process(self):
        pl = self.pnl(self.contextHolder.config.pl_options.compounded_pl)
        self.value = calc_mv_av(pl)     
        return True


    def pnl(self, compounded = False):
        if compounded:
            pl = pd.DataFrame([(x.Key.ToString("yyyy-MM-dd"),x.Value.pl_compounded) for x in self.contextHolder.pl_manager.pl_by_date]).set_index(0)
        else:
            pl = pd.DataFrame([(x.Key.ToString("yyyy-MM-dd"),x.Value.pl) for x in self.contextHolder.pl_manager.pl_by_date]).set_index(0)

        pl.index = pd.to_datetime(pl.index, format='%Y-%m-%d')
        return pl  




class DDMaxDayStat(IMercuryStat):
    __namespace__ = 'Mercury'

    def __init__(self):
        super().__init__()
        self.id = int(StatisticID.DD_MAX_DAY)
        self.name = "DD Max (d)"
        self.value = 0


    def get_contextHolder(self):
        return self.contextHolder
    def set_contextHolder(self, contextHolder):
        self.contextHolder = contextHolder


    def get_name(self):
        return self.name
    def set_name(self, name):
        self.name = name

    def get_id(self):
        return self.id
    def set_id(self, id):
        self.id = id

    def get_value(self):
        return self.value
    def set_value(self, value):
        self.value = value


    def process(self):
        pl = self.pnl(self.contextHolder.config.pl_options.compounded_pl)
        self.value = calc_max_draw_down(pl, self.contextHolder.config.pl_options.compounded_pl )     
        return True


    def pnl(self, compounded = False):
        if compounded:
            pl = pd.DataFrame([(x.Key.ToString("yyyy-MM-dd"),x.Value.pl_compounded) for x in self.contextHolder.pl_manager.pl_by_date]).set_index(0)
        else:
            pl = pd.DataFrame([(x.Key.ToString("yyyy-MM-dd"),x.Value.pl) for x in self.contextHolder.pl_manager.pl_by_date]).set_index(0)

        pl.index = pd.to_datetime(pl.index, format='%Y-%m-%d')
        return pl  


class DDMaxDayOverVolStat(IMercuryStat):
    __namespace__ = 'Mercury'

    def __init__(self):
        super().__init__()
        self.id = int(StatisticID.DD_MAX_DAY_OVER_VOL)
        self.name = "DD(d) over Vol"
        self.value = 0


    def get_contextHolder(self):
        return self.contextHolder
    def set_contextHolder(self, contextHolder):
        self.contextHolder = contextHolder


    def get_name(self):
        return self.name
    def set_name(self, name):
        self.name = name

    def get_id(self):
        return self.id
    def set_id(self, id):
        self.id = id

    def get_value(self):
        return self.value
    def set_value(self, value):
        self.value = value


    def process(self):
        pl = self.pnl(self.contextHolder.config.pl_options.compounded_pl)
        self.value = calc_draw_down_divided_by_vol(pl, NUM_DAYS_IN_YEAR, self.contextHolder.config.pl_options.compounded_pl )     
        return True


    def pnl(self, compounded = False):
        if compounded:
            pl = pd.DataFrame([(x.Key.ToString("yyyy-MM-dd"),x.Value.pl_compounded) for x in self.contextHolder.pl_manager.pl_by_date]).set_index(0)
        else:
            pl = pd.DataFrame([(x.Key.ToString("yyyy-MM-dd"),x.Value.pl) for x in self.contextHolder.pl_manager.pl_by_date]).set_index(0)

        pl.index = pd.to_datetime(pl.index, format='%Y-%m-%d')
        return pl  



class CurrentDDStat(IMercuryStat):
    __namespace__ = 'Mercury'

    def __init__(self):
        super().__init__()
        self.id = int(StatisticID.DD_CURRENT)
        self.name = "Current DD %"
        self.value = 0


    def get_contextHolder(self):
        return self.contextHolder
    def set_contextHolder(self, contextHolder):
        self.contextHolder = contextHolder


    def get_name(self):
        return self.name
    def set_name(self, name):
        self.name = name

    def get_id(self):
        return self.id
    def set_id(self, id):
        self.id = id

    def get_value(self):
        return self.value
    def set_value(self, value):
        self.value = value


    def process(self):
        pl = self.pnl(self.contextHolder.config.pl_options.compounded_pl)
        self.value = calc_current_draw_down(pl, self.contextHolder.config.pl_options.compounded_pl )     
        return True


    def pnl(self, compounded = False):
        if compounded:
            pl = pd.DataFrame([(x.Key.ToString("yyyy-MM-dd"),x.Value.pl_compounded) for x in self.contextHolder.pl_manager.pl_by_date]).set_index(0)
        else:
            pl = pd.DataFrame([(x.Key.ToString("yyyy-MM-dd"),x.Value.pl) for x in self.contextHolder.pl_manager.pl_by_date]).set_index(0)

        pl.index = pd.to_datetime(pl.index, format='%Y-%m-%d')
        return pl  



class CAGRStat(IMercuryStat):
    __namespace__ = 'Mercury'

    def __init__(self):
        super().__init__()
        self.id = int(StatisticID.CAGR)
        self.name = "CAGR"
        self.value = 0


    def get_contextHolder(self):
        return self.contextHolder
    def set_contextHolder(self, contextHolder):
        self.contextHolder = contextHolder


    def get_name(self):
        return self.name
    def set_name(self, name):
        self.name = name

    def get_id(self):
        return self.id
    def set_id(self, id):
        self.id = id

    def get_value(self):
        return self.value
    def set_value(self, value):
        self.value = value


    def process(self):
        pl = self.pnl(self.contextHolder.config.pl_options.compounded_pl)
        self.value = calc_cagr(pl)

        return True

    def pnl(self, compounded = False):
        if compounded:
            pl = pd.DataFrame([(x.Key.ToString("yyyy-MM-dd"),x.Value.pl_compounded) for x in self.contextHolder.pl_manager.pl_by_date]).set_index(0)
        else:
            pl = pd.DataFrame([(x.Key.ToString("yyyy-MM-dd"),x.Value.pl) for x in self.contextHolder.pl_manager.pl_by_date]).set_index(0)

        pl.index = pd.to_datetime(pl.index, format='%Y-%m-%d')
        return pl  




class YTDPct(IMercuryStat):
    __namespace__ = 'Mercury'

    def __init__(self):
        super().__init__()
        self.id = int(StatisticID.YTDPct)
        self.name = "YTDPct"
        self.value = 0


    def get_contextHolder(self):
        return self.contextHolder
    def set_contextHolder(self, contextHolder):
        self.contextHolder = contextHolder


    def get_name(self):
        return self.name
    def set_name(self, name):
        self.name = name

    def get_id(self):
        return self.id
    def set_id(self, id):
        self.id = id

    def get_value(self):
        return self.value
    def set_value(self, value):
        self.value = value


    def process(self):
        pl = self.pnl(self.contextHolder.config.pl_options.compounded_pl)

        utc_now = datetime.datetime.now(datetime.UTC)

        today = pd.Timestamp.today()

        # Get the beginning of the year
        beginning_of_year = today - pd.offsets.YearBegin()


        # aar = calc_annualized_return(pl[beginning_of_year:], NUM_DAYS_IN_YEAR, self.contextHolder.config.pl_options.compounded_pl ) 

        try:
            ytd = round(pl[beginning_of_year:].sum() * 100, 4)
        except KeyError:
            ytd = np.nan

        self.value = float(ytd.iloc[0])

        return True


    def pnl(self, compounded = False):
        if compounded:
            pl = pd.DataFrame([(x.Key.ToString("yyyy-MM-dd"),x.Value.pl_compounded) for x in self.contextHolder.pl_manager.pl_by_date]).set_index(0)
        else:
            pl = pd.DataFrame([(x.Key.ToString("yyyy-MM-dd"),x.Value.pl) for x in self.contextHolder.pl_manager.pl_by_date]).set_index(0)

        pl.index = pd.to_datetime(pl.index, format='%Y-%m-%d')
        return pl  


class OneYearPctStat(IMercuryStat):
    __namespace__ = 'Mercury'

    def __init__(self):
        super().__init__()
        self.id = int(StatisticID.OneYPct)
        self.name = "1YPct"
        self.value = 0


    def get_contextHolder(self):
        return self.contextHolder
    def set_contextHolder(self, contextHolder):
        self.contextHolder = contextHolder


    def get_name(self):
        return self.name
    def set_name(self, name):
        self.name = name

    def get_id(self):
        return self.id
    def set_id(self, id):
        self.id = id

    def get_value(self):
        return self.value
    def set_value(self, value):
        self.value = value


    def process(self):
        pl = self.pnl(self.contextHolder.config.pl_options.compounded_pl)

        end_date = pl.index.max()
        one_year = round(pl[end_date-relativedelta(years=1):].sum() * 100, 4)

        self.value = float(one_year.iloc[0])

        return True


    def pnl(self, compounded = False):
        if compounded:
            pl = pd.DataFrame([(x.Key.ToString("yyyy-MM-dd"),x.Value.pl_compounded) for x in self.contextHolder.pl_manager.pl_by_date]).set_index(0)
        else:
            pl = pd.DataFrame([(x.Key.ToString("yyyy-MM-dd"),x.Value.pl) for x in self.contextHolder.pl_manager.pl_by_date]).set_index(0)

        pl.index = pd.to_datetime(pl.index, format='%Y-%m-%d')
        return pl  



class TwoYearPctStat(IMercuryStat):
    __namespace__ = 'Mercury'

    def __init__(self):
        super().__init__()
        self.id = int(StatisticID.TwoYPct)
        self.name = "2YPct"
        self.value = 0


    def get_contextHolder(self):
        return self.contextHolder
    def set_contextHolder(self, contextHolder):
        self.contextHolder = contextHolder


    def get_name(self):
        return self.name
    def set_name(self, name):
        self.name = name

    def get_id(self):
        return self.id
    def set_id(self, id):
        self.id = id

    def get_value(self):
        return self.value
    def set_value(self, value):
        self.value = value


    def process(self):
        pl = self.pnl(self.contextHolder.config.pl_options.compounded_pl)

        end_date = pl.index.max()
        two_years = round(pl[end_date-relativedelta(years=2):].sum() * 100, 4)

        self.value = float(two_years.iloc[0])
        return True


    def pnl(self, compounded = False):
        if compounded:
            pl = pd.DataFrame([(x.Key.ToString("yyyy-MM-dd"),x.Value.pl_compounded) for x in self.contextHolder.pl_manager.pl_by_date]).set_index(0)
        else:
            pl = pd.DataFrame([(x.Key.ToString("yyyy-MM-dd"),x.Value.pl) for x in self.contextHolder.pl_manager.pl_by_date]).set_index(0)

        pl.index = pd.to_datetime(pl.index, format='%Y-%m-%d')
        return pl  



class ThreeYearPctStat(IMercuryStat):
    __namespace__ = 'Mercury'

    def __init__(self):
        super().__init__()
        self.id = int(StatisticID.ThreeYPct)
        self.name = "3YPct"
        self.value = 0


    def get_contextHolder(self):
        return self.contextHolder
    def set_contextHolder(self, contextHolder):
        self.contextHolder = contextHolder


    def get_name(self):
        return self.name
    def set_name(self, name):
        self.name = name

    def get_id(self):
        return self.id
    def set_id(self, id):
        self.id = id

    def get_value(self):
        return self.value
    def set_value(self, value):
        self.value = value


    def process(self):
        pl = self.pnl(self.contextHolder.config.pl_options.compounded_pl)

        end_date = pl.index.max()
        three_years = round(pl[end_date-relativedelta(years=3):].sum() * 100, 4)

        self.value = float(three_years.iloc[0])
        return True


    def pnl(self, compounded = False):
        if compounded:
            pl = pd.DataFrame([(x.Key.ToString("yyyy-MM-dd"),x.Value.pl_compounded) for x in self.contextHolder.pl_manager.pl_by_date]).set_index(0)
        else:
            pl = pd.DataFrame([(x.Key.ToString("yyyy-MM-dd"),x.Value.pl) for x in self.contextHolder.pl_manager.pl_by_date]).set_index(0)

        pl.index = pd.to_datetime(pl.index, format='%Y-%m-%d')
        return pl  




class FiveYearPctStat(IMercuryStat):
    __namespace__ = 'Mercury'

    def __init__(self):
        super().__init__()
        self.id = int(StatisticID.FiveYPct)
        self.name = "5YPct"
        self.value = 0


    def get_contextHolder(self):
        return self.contextHolder
    def set_contextHolder(self, contextHolder):
        self.contextHolder = contextHolder


    def get_name(self):
        return self.name
    def set_name(self, name):
        self.name = name

    def get_id(self):
        return self.id
    def set_id(self, id):
        self.id = id

    def get_value(self):
        return self.value
    def set_value(self, value):
        self.value = value


    def process(self):
        pl = self.pnl(self.contextHolder.config.pl_options.compounded_pl)

        end_date = pl.index.max()
        five_years = round(pl[end_date-relativedelta(years=5):].sum() * 100, 4)

        self.value = float(five_years.iloc[0])
        return True


    def pnl(self, compounded = False):
        if compounded:
            pl = pd.DataFrame([(x.Key.ToString("yyyy-MM-dd"),x.Value.pl_compounded) for x in self.contextHolder.pl_manager.pl_by_date]).set_index(0)
        else:
            pl = pd.DataFrame([(x.Key.ToString("yyyy-MM-dd"),x.Value.pl) for x in self.contextHolder.pl_manager.pl_by_date]).set_index(0)

        pl.index = pd.to_datetime(pl.index, format='%Y-%m-%d')
        return pl  





class TenYearPctStat(IMercuryStat):
    __namespace__ = 'Mercury'

    def __init__(self):
        super().__init__()
        self.id = int(StatisticID.TenYPct)
        self.name = "10YPct"
        self.value = 0


    def get_contextHolder(self):
        return self.contextHolder
    def set_contextHolder(self, contextHolder):
        self.contextHolder = contextHolder


    def get_name(self):
        return self.name
    def set_name(self, name):
        self.name = name

    def get_id(self):
        return self.id
    def set_id(self, id):
        self.id = id

    def get_value(self):
        return self.value
    def set_value(self, value):
        self.value = value


    def process(self):
        pl = self.pnl(self.contextHolder.config.pl_options.compounded_pl)

        end_date = pl.index.max()
        ten_years = round(pl[end_date-relativedelta(years=10):].sum() * 100, 4)

        self.value = float(ten_years.iloc[0])
        return True


    def pnl(self, compounded = False):
        if compounded:
            pl = pd.DataFrame([(x.Key.ToString("yyyy-MM-dd"),x.Value.pl_compounded) for x in self.contextHolder.pl_manager.pl_by_date]).set_index(0)
        else:
            pl = pd.DataFrame([(x.Key.ToString("yyyy-MM-dd"),x.Value.pl) for x in self.contextHolder.pl_manager.pl_by_date]).set_index(0)

        pl.index = pd.to_datetime(pl.index, format='%Y-%m-%d')
        return pl  



class TwentyYearPctStat(IMercuryStat):
    __namespace__ = 'Mercury'

    def __init__(self):
        super().__init__()
        self.id = int(StatisticID.TwentyYPct)
        self.name = "20YPct"
        self.value = 0


    def get_contextHolder(self):
        return self.contextHolder
    def set_contextHolder(self, contextHolder):
        self.contextHolder = contextHolder


    def get_name(self):
        return self.name
    def set_name(self, name):
        self.name = name

    def get_id(self):
        return self.id
    def set_id(self, id):
        self.id = id

    def get_value(self):
        return self.value
    def set_value(self, value):
        self.value = value


    def process(self):
        pl = self.pnl(self.contextHolder.config.pl_options.compounded_pl)

        end_date = pl.index.max()
        twenty_years = round(pl[end_date-relativedelta(years=20):].sum() * 100, 4)

        self.value = float(twenty_years.iloc[0])
        return True


    def pnl(self, compounded = False):
        if compounded:
            pl = pd.DataFrame([(x.Key.ToString("yyyy-MM-dd"),x.Value.pl_compounded) for x in self.contextHolder.pl_manager.pl_by_date]).set_index(0)
        else:
            pl = pd.DataFrame([(x.Key.ToString("yyyy-MM-dd"),x.Value.pl) for x in self.contextHolder.pl_manager.pl_by_date]).set_index(0)

        pl.index = pd.to_datetime(pl.index, format='%Y-%m-%d')
        return pl  


class QTDPctStat(IMercuryStat):
    __namespace__ = 'Mercury'

    def __init__(self):
        super().__init__()
        self.id = int(StatisticID.QTDPct)
        self.name = "QTDPct"
        self.value = 0


    def get_contextHolder(self):
        return self.contextHolder
    def set_contextHolder(self, contextHolder):
        self.contextHolder = contextHolder


    def get_name(self):
        return self.name
    def set_name(self, name):
        self.name = name

    def get_id(self):
        return self.id
    def set_id(self, id):
        self.id = id

    def get_value(self):
        return self.value
    def set_value(self, value):
        self.value = value


    def process(self):
        pl = self.pnl(self.contextHolder.config.pl_options.compounded_pl)

        end_date = pl.index.max()
        quarter_start =  pd.Timestamp(end_date) - pd.tseries.offsets.QuarterBegin(startingMonth=1)

        try:
            qtd = round(pl[quarter_start:].sum() * 100, 4)
        except KeyError:
            qtd = np.nan

        self.value = float(qtd.iloc[0])
        return True


    def pnl(self, compounded = False):
        if compounded:
            pl = pd.DataFrame([(x.Key.ToString("yyyy-MM-dd"),x.Value.pl_compounded) for x in self.contextHolder.pl_manager.pl_by_date]).set_index(0)
        else:
            pl = pd.DataFrame([(x.Key.ToString("yyyy-MM-dd"),x.Value.pl) for x in self.contextHolder.pl_manager.pl_by_date]).set_index(0)

        pl.index = pd.to_datetime(pl.index, format='%Y-%m-%d')
        return pl  





class LongShortRatioStat(IMercuryStat):
    __namespace__ = 'Mercury'

    def __init__(self):
        super().__init__()
        self.id = int(StatisticID.LONG_SHORT_RATIO)
        self.name = "LongShort Ratio"
        self.value = 0


    def get_contextHolder(self):
        return self.contextHolder
    def set_contextHolder(self, contextHolder):
        self.contextHolder = contextHolder


    def get_name(self):
        return self.name
    def set_name(self, name):
        self.name = name

    def get_id(self):
        return self.id
    def set_id(self, id):
        self.id = id

    def get_value(self):
        return self.value
    def set_value(self, value):
        self.value = value


    def process(self):

        longTrades = 0
        shortTrades = 0

        positionBySystemAndSymbol = {}

        for order in self.contextHolder.oms.filled_orders: 

            order_symbol = order.ticker.ToString()

            if order.system not in  positionBySystemAndSymbol:
                positionBySystemAndSymbol[order.system] = {}

            if order_symbol not in  positionBySystemAndSymbol[order.system]:
                positionBySystemAndSymbol[order.system][order_symbol] = 0

            # ok now see if we're going from 0 or crossing 0.  for crossing, check if the qty is 
            # opposite sign and Abs() bigger than the position
            if positionBySystemAndSymbol[order.system][order_symbol] == 0 or \
                    ( ( abs( order.quantity ) > abs( positionBySystemAndSymbol[order.system][order_symbol] ) ) and
                      ( np.sign( order.quantity ) != np.sign( positionBySystemAndSymbol[order.system][order_symbol] ) ) ):
                
                #  this is initiating a positon from 0 or crossing 0 - adjust the trade counter
                if( order.quantity > 0 ):
                    longTrades += 1
                else:
                    shortTrades +=1
                    
                
            # regardless adjust the position
            positionBySystemAndSymbol[order.system][order_symbol] += order.quantity
            
        value = 1
        if shortTrades != 0: 
            value =  longTrades / shortTrades

        self.value = float(value)
        return True



