from psaw import PushshiftAPI
import praw
import pprint
from datetime import datetime, timedelta
import pandas as pd
import os
import time

import os
import jsonlines
cwd = os.getcwd()
print(cwd)

subreddit_list = [
    # "securityanalysis",
    # "investing",
    # "stocks",
    # "stockmarket",
    # "economy",
    "globalmarkets",
    # "dividends",
    # "daytrading",
    # "economy",
#     "wallstreetbets",
    # "options"
]

for subreddit in subreddit_list:
    csv = f"data/{subreddit}_submission.csv"
    data = pd.read_csv(csv)
    full_texts = [None] * data.shape[0]
    print(subreddit)
    for idx, row in data.iterrows():
        # print(row)
        full_texts[idx] = {"text": row["title"] + "\n" + row["text"], "label": []}
        if(idx % 1000 == 0):
            print(idx / data.shape[0])
    
#     insert_idx = len(data.columns)
    insert_idx = 3

    with jsonlines.open(f"{subreddit}_submissions.jsonl", "w") as writer:
        writer.write_all(full_texts)
    # data.insert(insert_idx, "full_text", full_texts)
    # data.to_csv(f"full_text_{subreddit}_submissions.csv", mode="w", header=True, index=False)