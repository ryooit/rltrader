import csv
import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
from pandas_datareader import data
import fix_yahoo_finance
fix_yahoo_finance.pdr_override()

chdata = data.get_data_yahoo(
            # tickers list (single tickers accepts a string as well)
            tickers = "SPY",

            # start date (YYYY-MM-DD / datetime.datetime object)
            # (optional, defaults is 1950-01-01)
            start = "2017-01-01",

            # end date (YYYY-MM-DD / datetime.datetime object)
            # (optional, defaults is Today)
            end = "2017-04-30",

            # return a multi-index dataframe
            # (optional, default is Panel, which is deprecated)
            as_panel = False,

            # group by ticker (to access via data['SPY'])
            # (optional, default is 'column')
            group_by = 'ticker',

            # adjust all OHLC automatically
            # (optional, default is False)
            auto_adjust = True,

            # download dividend + stock splits data
            # (optional, default is None)
            # options are:
            #   - True (returns history + actions)
            #   - 'only' (actions only)
            actions = True,

            # How may threads to use?
            threads = 1
        )
df = pd.DataFrame(chdata, columns = ['Open', 'High', 'Low', 'Close', 'Volume'])
df.to_csv('./data/chart_data/spy.csv', header=False)
