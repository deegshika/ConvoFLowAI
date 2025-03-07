import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import re

# Load the cleaned data
df = pd.read_csv('reddit_comments_cleaned.csv')

# Combine all comments into a single string
all_comments = ' '.join(df['Comment'].astype(str))

# Tokenize words and remove short words (like "a", "is", etc.)
tokens = re.findall(r'\b\w{3,}\b', all_comments.lower())

# Count word frequencies
word_counts = Counter(tokens)

# Generate word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_counts)

# Plot word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most Frequent Words in Reddit Comments')
plt.show()

# Top N words bar plot
top_n = 20
common_words = word_counts.most_common(top_n)
words, counts = zip(*common_words)

plt.figure(figsize=(12, 6))
plt.bar(words, counts, color='skyblue')
plt.title(f'Top {top_n} Most Common Words')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Frequency')
plt.show()

# Comment length distribution
comment_lengths = df['Comment'].apply(lambda x: len(str(x).split()))

plt.figure(figsize=(10, 5))
plt.hist(comment_lengths, bins=30, color='coral', edgecolor='black')
plt.title('Distribution of Comment Lengths')
plt.xlabel('Number of Words')
plt.ylabel('Number of Comments')
plt.show()
