# Data Description

We have collected data from the following finance related subreddits from 1st January 2014 to 1st January 2021 using the [PRAW API](https://praw.readthedocs.io/en/latest/code_overview/reddit_instance.html).

### Subreddits 
- daytrading
- dividents
- wallstreetbets
- economy
- globalmarkets
- investing
- securityanalysis
- stockmarket
- stocks

For each of the above subreddits we have two csv files one for submissions and one for comments. You will find these files in the [data](../data) folder.

#### Headers for the submission csv:

- `name`: Fullname of the submission.
- `text`: Text of the submission. 
- `score`: Number of upvotes on the submission.
- `upvote_ratio`: The percentage of upvotes from all votes on the submission.
- `created_utc`: Time the submission was created, represented in Unix Time.

#### Headers for the comments csv:
- `submission_name`: Full name of the submission (post). 
- `id`: The ID of the comment.
- `text`: The comment text.
- `score`: Number of upvotes on the comment.
- `created_utc`: Time the comment was created, represented in Unix Time.
