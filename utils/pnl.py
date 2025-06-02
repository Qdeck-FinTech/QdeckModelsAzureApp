import pandas as pd
from collections import defaultdict

def get_symbol_sector_pnl(model,pl_manager,market_data):

    pl_list = []
    sector_pl_lists = defaultdict(list)
    for sy in model.symbols:
        pl_manager.process_include(sy)
        frame = pd.DataFrame([(x.Key.ToString(),x.Value.pl) for x in pl_manager.pl_by_date]).set_index(0)
        frame.index = pd.to_datetime(frame.index)
        frame.columns = [sy]
        pl_list.append(frame)
        sector_pl_lists[market_data.get_symbol_data(sy).sector].append(frame)
        
    pl_by_symbol = pd.concat(pl_list,axis=1)

    pl_by_sector = []
    for sector in sector_pl_lists.keys():
        sector_pl = pd.concat(sector_pl_lists[sector],axis=1).fillna(0.0).sum(axis=1)
        sector_pl.name = sector
        pl_by_sector.append(sector_pl)
        
    pl_by_sector = pd.concat(pl_by_sector,axis=1)
    
    return pl_by_symbol, pl_by_sector


def get_symbol_pnl(model,pl_manager):

    pl_list = []
    for sy in model.symbols:
        pl_manager.process_include(sy)
        frame = pd.DataFrame([(x.Key.ToString(),x.Value.pl) for x in pl_manager.pl_by_date]).set_index(0)
        frame.index = pd.to_datetime(frame.index)
        frame.columns = [sy]
        pl_list.append(frame)
        
    pl_by_symbol = pd.concat(pl_list,axis=1)

    return pl_by_symbol