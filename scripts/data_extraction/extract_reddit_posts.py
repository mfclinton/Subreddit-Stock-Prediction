from psaw import PushshiftAPI
import praw
import pprint
from datetime import datetime, timedelta
import pandas as pd
import os

class Data_Handler:
    def __init__(self, subreddit, buffer_size=1000):
        self.subreddit = subreddit
        self.submission_buffer = [] 
        self.comment_buffer = []
        self.buffer_size = buffer_size
        
        self.last_submission_name = None
        self.last_comment_id = None

    def Create_CSV(self, initial_date):
        subreddit=self.subreddit
        submission_csv = f"{subreddit}_submissions.csv"
        comments_csv = f"{subreddit}_comments.csv"

        if not os.path.exists(submission_csv):
            with open(submission_csv,"w") as file:
                file.write(",".join(submission_columns))
                file.write("\n")
                
            with open(comments_csv,"w") as file:
                file.write(",".join(comment_columns))
                file.write("\n")
            
            return initial_date
        else:
            last_submission = pd.read_csv(submission_csv).iloc[-1]
            last_comment = pd.read_csv(comments_csv).iloc[-1]
            
            self.last_submission_name = last_submission["name"]
            self.last_comment_id = last_comment["id"]
            
            last_submission_time = datetime.utcfromtimestamp(last_submission["created_utc"])
            last_comment_time = datetime.utcfromtimestamp(last_comment["created_utc"])
            last_time = last_submission_time if last_submission_time < last_comment_time else last_comment_time
            last_time = last_time.replace(hour=0, minute=0, second=0, microsecond=0)
            return last_time
                


    def Write_Data(self, force_flush = False):
        subreddit=self.subreddit
        submission_buffer=self.submission_buffer
        comment_buffer=self.comment_buffer
        buffer_size=self.buffer_size
        
        if buffer_size <= len(submission_buffer) or force_flush:
            
            while self.last_submission_name != None and len(submission_buffer) != 0:
                temp_name = submission_buffer.pop(0)[0] #index 0 is name
#                 print(temp_name, self.last_submission_name)
                if temp_name == self.last_submission_name:
                    print("FOUND IT", temp_name)
                    self.last_submission_name = None
                    
            if len(submission_buffer) != 0:        
                submission_df = pd.DataFrame(submission_buffer, columns = submission_columns)
                submission_df.to_csv(f"{subreddit}_submissions.csv", mode="a", header=False, index=False)
            self.submission_buffer = []
        if buffer_size <= len(comment_buffer) or force_flush:
            
            while self.last_comment_id != None and len(comment_buffer) != 0:
#                 temp_name = comment_buffer[0][0]
                temp_id = comment_buffer.pop(0)[1] #index 1 is id
#                 print(temp_name, temp_id, self.last_comment_id)
                if temp_id == self.last_comment_id:
                    print("FOUND IT", temp_id)
                    self.last_comment_id = None
            
            if len(comment_buffer) != 0:
                comment_df = pd.DataFrame(comment_buffer, columns = comment_columns)
                comment_df.to_csv(f"{subreddit}_comments.csv", mode="a", header=False, index=False)
            self.comment_buffer = []

if __name__=="__main__":
    # Reddit Agent Authorization Data
    client_id = "tpammI9-HYB25Q"
    secret_token = "aDm9bgjz6Vn1QtePAEEzWNJ5wyforg"
    username="AbbieSnoozeAlot"
    password="E2zT9vV7GxWtawK"
    agent="SrsBot/0.0.1"

    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=secret_token,
        user_agent=agent,
        username=username,
        password=password,
    )

    reddit.read_only = True
    api = PushshiftAPI(reddit)

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
        "wallstreetbets",
        "options"
    ]

    deleted_keywords = ["[deleted]", "[removed]", "", None]
    submission_columns = ["name", "text", "score", "upvote_ratio", "created_utc"]
    comment_columns = ["submission_name", "id", "text", "score", "created_utc"]


    initial_date = datetime(2014,1,1)
    end_date = datetime(2021,1,1)
    delta_time = timedelta(days=1)

    buffer_size = 1000
    min_num_interactions = 2
    for sr in subreddit_list:
        cur_date = initial_date
        data_handler = Data_Handler(sr, buffer_size=buffer_size)
        cur_date = data_handler.Create_CSV(initial_date)
        while cur_date < end_date:
    #         if cur_date.day == 0:
    #             print(cur_date)
            print(cur_date)
            
            gen = api.search_submissions(after=cur_date, before=cur_date + delta_time,
                                        subreddit=sr,
                                        filter=["url","author", "title", "subreddit", "selftext"])
            
            
            for submission in gen:
                submission_text = submission.selftext
                submission_score = submission.score
                if(submission_text in deleted_keywords or submission_score < min_num_interactions):
                    continue
                    
                data_handler.submission_buffer.append([submission.name, 
                                        submission_text, 
                                        submission_score, 
                                        submission.upvote_ratio,
                                        submission.created_utc])
                
                submission.comments.replace_more(limit=None)
                for comment in submission.comments:
                    comment_text = comment.body
                    comment_score = comment.score
                    if(comment_text in deleted_keywords or comment_score < min_num_interactions):
                        continue
                    
                    data_handler.comment_buffer.append([submission.name, 
                                        comment.id, 
                                        comment_text, 
                                        comment_score,
                                        comment.created_utc])
            
            data_handler.Write_Data()
            cur_date += delta_time
        
        data_handler.Write_Data(force_flush = True) #Flush remaining buffer
