import pandas as pd
import joblib
from collections import Counter
from textblob import TextBlob

# Load trained model and vectorizer
model = joblib.load('comment_classifier.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Load chat data
chat_data = pd.read_csv('test_chat.csv')

# Vectorize chat messages
X_chat_tfidf = vectorizer.transform(chat_data['Message'])

# Predict labels and probabilities
predictions = model.predict(X_chat_tfidf)
confidences = model.predict_proba(X_chat_tfidf).max(axis=1)

# Add predictions and confidence scores to DataFrame
chat_data['Label'] = predictions
chat_data['Confidence'] = confidences

# Toxicity detection using sentiment analysis
def detect_toxicity(message):
    analysis = TextBlob(message)
    return 'toxic' if analysis.sentiment.polarity < -0.5 else 'non-toxic'

chat_data['Toxicity'] = chat_data['Message'].apply(detect_toxicity)

# Message frequency count
message_frequency = Counter(chat_data['User'])

# Count ignored messages (confidence < 0.5)
ignored_messages = chat_data[chat_data['Confidence'] < 0.5]['User'].value_counts().to_dict()

# Calculate fairness scores
total_messages = len(chat_data)
fairness_scores = {user: round(count / total_messages, 2) for user, count in message_frequency.items()}

# Count toxic messages per user
toxic_counts = chat_data[chat_data['Toxicity'] == 'toxic']['User'].value_counts().to_dict()

# Display chat analysis report
print("\n--- Chat Analysis Report ---")
print(chat_data[['User', 'Message', 'Label', 'Confidence', 'Toxicity']])

print("\nMessage Frequency:", dict(message_frequency))
print("Ignored Messages:", ignored_messages)
print("Fairness Scores:", fairness_scores)
print("Toxic Messages Count:", toxic_counts)
