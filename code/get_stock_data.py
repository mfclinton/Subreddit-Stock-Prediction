# https://towardsdatascience.com/downloading-historical-stock-prices-in-python-93f85f059c1f

import pandas as pd
# import yfinance as yf
import datetime
import time
import requests
import io
import pickle
import pandas_datareader as dr


f = open("code/data_labeling/seen.p","rb")
tickers = list(pickle.load(f))
print(len(tickers))

start = datetime.datetime(2014,1,1)
end = datetime.datetime(2021,1,1)

# tickers = list(pd.read_csv("data/ExchangeListings/tickers.csv")["Symbol"])
for t in tickers:
    # print(t)
    try:
        stock = []
        # stock = yf.download("RDS-A",start=start, end=end, progress=False, interval="1d")
        stock = dr.data.DataReader(t, data_source='yahoo', start=start, end=end)
        if len(stock) == 0:
            pass
        else:
            stock["Name"] = t
        stock.to_csv(f"data/stock_data/{t}.csv", index=False)
        # 1/0    
    except Exception:
        print("Unable to to download ", t)

    # 1/0