from psaw import PushshiftAPI
import praw
import pprint
from datetime import datetime, timedelta
import pandas as pd
import os
import time

import os
cwd = os.getcwd()
print(cwd)

subreddit_list = [
    "securityanalysis",
    "investing",
    "stocks",
    "stockmarket",
    "economy",
    "globalmarkets",
    "dividends",
    "daytrading",
    "economy",
#     "wallstreetbets",
    "options"
]

for subreddit in subreddit_list:
    csv = f"data/{subreddit}_submission.csv"
    data = pd.read_csv(csv)
    full_texts = [None] * data.shape[0]
    print(subreddit)
    for idx, row in data.iterrows():
        # print(row)
        try:
            full_texts[idx] = row["title"] + "\n\n" + row["text"]

            if(row["title"] == "" or row["title"] == None):
                1/0
        except:
            print(row)
            1/0
        if(idx % 1000 == 0):
            print(idx / data.shape[0])
    
#     insert_idx = len(data.columns)
    insert_idx = 3
    # data.insert(insert_idx, "full_text", full_texts)
    # data.to_csv(f"{subreddit}_submissions.csv", mode="w", header=True, index=False)