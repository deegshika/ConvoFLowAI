import pandas as pd

# Load cleaned data
df = pd.read_csv("reddit_comments_cleaned.csv")

# Define keyword sets
question_keywords = ["how", "what", "why", "when", "where", "which", "who", "does", "is", "can", "will", "should"]
answer_keywords = ["yes", "no", "I think", "according to", "the answer is", "you should"]
opinion_keywords = ["I believe", "in my experience", "I feel", "I prefer", "from my perspective"]

# Function to classify comments
def classify_comment(comment):
    comment_lower = comment.lower()

    if any(word in comment_lower for word in question_keywords):
        return "question"
    elif any(word in comment_lower for word in answer_keywords):
        return "answer"
    elif any(word in comment_lower for word in opinion_keywords):
        return "opinion"
    else:
        return "other"

# Apply classification
df["Label"] = df["Comment"].astype(str).apply(classify_comment)

# Save labeled data
df.to_csv("reddit_comments_labeled.csv", index=False)

# Summary
print("Labeled data saved to reddit_comments_labeled.csv")
print(df["Label"].value_counts())
print(df.head())
