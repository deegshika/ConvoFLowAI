import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import RandomOverSampler
import joblib

# Load the labeled data
df = pd.read_csv('reddit_comments_labeled.csv')

# Features (comments) and labels (question, answer, other)
X = df['Comment']
y = df['Label']

# Visualize class distribution before resampling
plt.figure(figsize=(8, 5))
sns.countplot(x=y, palette='coolwarm')
plt.title('Class Distribution Before Resampling')
plt.show()

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text data into TF-IDF features
vectorizer = TfidfVectorizer(max_features=1500, ngram_range=(1,2), stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Apply oversampling to balance classes
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_train_tfidf, y_train)

# Visualize class distribution after resampling
plt.figure(figsize=(8, 5))
sns.countplot(x=y_resampled, palette='coolwarm')
plt.title('Class Distribution After Resampling')
plt.show()

# Retrain logistic regression model with balanced data
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_resampled, y_resampled)

# Predictions
y_pred = model.predict(X_test_tfidf)

# Evaluation
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, 'comment_classifier.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
print("\nRetrained model with oversampling saved!")
