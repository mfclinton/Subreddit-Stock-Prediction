import os
import pandas as pd
from datetime import datetime, timedelta
import pandas_datareader as dr
import yfinance as yf
import numpy as np


time_interval = timedelta(weeks=1)

blocked_stocks = {}

for filename in os.listdir("data"):
    if not filename.endswith(".jsonl"):
        continue
    print(filename)
    
    cur_path = os.path.abspath(f"data/{filename}")
    subreddit = filename.replace(".jsonl", "")
    print(subreddit)

    data = pd.read_json(cur_path, lines=True)
    data["ticker"] = np.nan
    data["cur_price"] = np.nan
    data["future_price"] = np.nan

    prices = []

    for idx, row in data.iterrows():
        if(len(row["tickers"]) <= 0):
            continue

        ticker, freq = row["tickers"][0]
        # if(ticker in blocked_stocks):
        #     continue

        date = datetime.utcfromtimestamp(row["created_utc"])
        try:
            # stock = yf.download(ticker, progress=False, start=date, end=date + time_interval)
            stock = dr.data.DataReader(ticker, data_source='yahoo', start=date, end=date + time_interval)
            cur_price = stock.iloc[0]["Low"] #Note : is rounded to nearest date
            future_price = stock.iloc[-1]["High"]
            
            data["ticker"] = ticker
            row["cur_price"] = cur_price
            row["future_price"] = future_price

        except Exception:
            error_info = f"{filename} | {ticker} | {idx}"
            print(f"{error_info} failed")
            blocked_stocks[ticker] = error_info

    data.to_csv(f"data/future_{subreddit}_submission.csv", mode="w", header=True, index=False)


print("Blocked Stocks")
print(blocked_stocks)


