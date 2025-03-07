import praw
import pandas as pd
import re
from config import CLIENT_ID, CLIENT_SECRET, USER_AGENT

# Initialize Reddit instance
reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    user_agent=USER_AGENT
)

# Function to fetch comments
def fetch_comments(subreddit_name, post_limit=50, comment_limit=50):
    subreddit = reddit.subreddit(subreddit_name)
    data = []
    
    for submission in subreddit.hot(limit=post_limit):
        submission.comments.replace_more(limit=0)
        for comment in submission.comments.list()[:comment_limit]:
            data.append([submission.title, comment.body])
    
    return data

# Preprocess comments
def clean_comment(comment):
    # Remove links, special characters, and extra whitespace
    comment = re.sub(r'http\S+|www\S+', '', comment)
    comment = re.sub(r'[^A-Za-z0-9\s]', '', comment)
    comment = re.sub(r'\s+', ' ', comment).strip()
    return comment

def preprocess_data(data):
    df = pd.DataFrame(data, columns=["Post Title", "Comment"])
    df['Comment'] = df['Comment'].apply(clean_comment)
    # Remove empty comments or those with fewer than 3 words
    df = df[df['Comment'].str.split().str.len() >= 3]
    return df

# Store data in CSV
def store_to_csv(df, filename="reddit_comments_cleaned.csv"):
    df.to_csv(filename, index=False)
    print(f"Cleaned data saved to {filename}")

# Fetch, preprocess, and save data
subreddit_name = "MachineLearning"
raw_data = fetch_comments(subreddit_name)
cleaned_df = preprocess_data(raw_data)
store_to_csv(cleaned_df)

# Summary stats
print(f"Total comments collected: {len(cleaned_df)}")
print(f"Number of unique posts: {cleaned_df['Post Title'].nunique()}")
print("Sample data:")
print(cleaned_df.head())
