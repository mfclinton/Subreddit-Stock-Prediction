import pandas as pd
import os
import torch
import numpy as np

submission_sentiments = pd.read_csv("data/LabelledData/submissions_with_tickers_labelled.csv")
submission_sentiments.set_index("name", inplace=True)
comment_sentiments = pd.read_csv("data/LabelledData/comments_with_tickers_labelled.csv")

for filename in os.listdir("data"):
    if (not filename.startswith("future_")) or (not filename.endswith("submission.csv")):
        continue
    
    subreddit = filename.replace("_submission.csv", "")
    subreddit = subreddit.replace("future_", "")
    print(filename)
    print(subreddit)

    cur_path = os.path.abspath(f"data/{filename}")
    data = pd.read_csv(cur_path)

    features = []
    for idx, row in data.iterrows():
        if idx % 1000 == 0:
            print(idx)
        name = row["name"]
        ticker = row["ticker"]
        score = row["score"]
        upvote_ratio = row["upvote_ratio"]

        # 1 if goes up, 0 if goes down
        label = 1 if row["cur_price"] <= row["future_price"] else 0

        ticker = None #todo

        sentiment_data = submission_sentiments.loc[name]
        submission_sentiment_vector = [sentiment_data["neg"], sentiment_data["neu"], sentiment_data["pos"]]
        
        # TODO : Comments
        # comments = comment_sentiments.loc[comment_sentiments["name"] == name]

        # TODO : Make more complicated
        feature_vector = submission_sentiment_vector + [float(score), float(upvote_ratio), label]
        features.append(feature_vector)

    data = pd.DataFrame(np.array(features)).to_csv(f"data/{subreddit}_features_labels.csv", index=False)

    # feature_matrix = torch.Tensor(features)
    # print(feature_matrix.size())
    # torch.save(feature_matrix, f"data\{subreddit}_features.pt")
    # torch.save(feature_matrix, f"data\{subreddit}_labels.pt")

    



