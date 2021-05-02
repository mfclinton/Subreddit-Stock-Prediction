import os
import pandas as pd
from datetime import datetime, timedelta
import pandas_datareader as dr
import yfinance as yf
import numpy as np

# https://stackoverflow.com/questions/32237862/find-the-closest-date-to-a-given-date
def closest_date(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))



start = datetime(2014,1,1)
end = datetime(2021,1,1)

time_interval = timedelta(weeks=1)

blocked_stocks = {}
stocks_data = {}
for filename in os.listdir("data"):
    if not filename.endswith(".jsonl"):
        continue
    print(filename)
    
    cur_path = os.path.abspath(f"data/{filename}")
    subreddit = filename.replace(".jsonl", "")
    print(subreddit)

    data = pd.read_json(cur_path, lines=True)
    data["ticker"] = ""
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
            if ticker in stocks_data:
                stock = stocks_data[ticker]
            else:
                stock = dr.data.DataReader(ticker, data_source='yahoo', start=start, end=end)
                stocks_data[ticker] = stock

            cur_date = stock.index.get_loc(date, method="nearest")
            future_date = stock.index.get_loc(date + time_interval, method="nearest")
            cur_price = stock.iloc[cur_date]["Low"] #Note : is rounded to nearest date
            future_price = stock.iloc[future_date]["High"]
            
            data.at[idx, "ticker"] = ticker
            data.at[idx, "cur_price"] = cur_price
            data.at[idx, "future_price"] = future_price

        except Exception:

            error_info = f"{filename} | {ticker} | {idx}"
            print(f"{error_info} failed")
            blocked_stocks[ticker] = error_info

    data.to_csv(f"data/future_{subreddit}_submission.csv", mode="w", header=True, index=False)


print("Blocked Stocks")
print(blocked_stocks)
with open("blacklist.txt", "w") as f:
    f.write("\n".join(list(blocked_stocks.keys())))

