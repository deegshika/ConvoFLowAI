# fetch_reddit_data.py

import praw
from config import CLIENT_ID, CLIENT_SECRET, USER_AGENT

# Initialize Reddit instance
reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    user_agent=USER_AGENT
)

# Fetch comments from a subreddit (e.g., 'MachineLearning')
subreddit = reddit.subreddit('MachineLearning')

print("Top comments from r/MachineLearning:")

for submission in subreddit.hot(limit=5):
    print(f"\nTitle: {submission.title}")
    submission.comments.replace_more(limit=0)
    for comment in submission.comments.list()[:5]:
        print(f" - {comment.body}")

