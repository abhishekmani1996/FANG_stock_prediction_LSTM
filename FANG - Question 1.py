# -*- coding: utf-8 -*-
"""
FANG Portfolio Construction and Optimization - Q1.ipynb
"""
#Importing all the necessary modules
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import r2_score
import pandas_datareader.data as web


class DataFrameUtils:
    @staticmethod
    def removeRandomEntries(data_frame: object, count: int) -> object:
        """
        This method drops the
        @param data_frame: The data frame object to mutate
        @param count: The number of indices to drop
        @return: The dropped indices
        """
        return np.random.choice(data_frame.index, count, replace=False)


class StockDataDownloader:
    """
    This class takes a list of stock labels and fetches their data using yFinance API
    """

    def __init__(self, stocks: [str]):
        self.stocks = stocks
        self.df = None
        self.start_date = None
        self.end_date = None

    def getStockData(self, start_date: object, end_date: object):
        self.start_date = start_date
        self.end_date = end_date
        return web.DataReader(self.stocks, 'yahoo', start_date, end_date)['Adj Close']
        # self.df.columns = self.stocks

    def getStartDate(self):
        return self.start_date

    def getEndDate(self):
        return self.end_date


####### Solution Below #########

STOCKS: [str] = ['FB', 'AMZN', 'NFLX', 'GOOG']

stockDataDownloader = StockDataDownloader(STOCKS)
start_date = datetime(2020, 1, 1)
end_date = datetime(2020, 12, 31)
data_frame = stockDataDownloader.getStockData(start_date, end_date)
data_frame.columns = STOCKS

data_frame_utils = DataFrameUtils()

remove_n: int = 30  # We have to remove 30 adj close prices from these 4 FANG stocks
drop_indices = data_frame_utils.removeRandomEntries(data_frame, remove_n)

df_random30_drop = data_frame.copy()

for i in drop_indices:
    df_random30_drop.loc[i] = [np.nan] * len(df_random30_drop.columns)

df_random30_drop.info()
pd.set_option('display.max_rows', None)

# Interpolation
df_interpolate = {}
methods = ['linear', 'time']   # We use 2 methods for interpolating the missing data.
for i in methods:
    df_interpolate[i] = df_random30_drop.interpolate(method=i)
    print(df_interpolate[i])

# Finding the R^2 value
results = [(method, r2_score(data_frame, df_interpolate[method])) for method in df_interpolate.keys()]
results_df = pd.DataFrame(np.array(results), columns=['Method', 'R_squared'])
results_df.sort_values(by='R_squared', ascending=False)
