import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

# Sample mock dataset
data = {
    'text': [
        'I love trying new things and exploring.',
        'I keep everything organized and plan ahead.',
        'I enjoy meeting new people and socializing.',
        'I often feel anxious or worried.',
        'I care deeply about others and am very cooperative.'
    ],
    'trait': ['Openness', 'Conscientiousness', 'Extraversion', 'Neuroticism', 'Agreeableness']
}

df = pd.DataFrame(data)

# Convert text to vectors
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])

# Encode personality traits as labels
y = df['trait']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model and vectorizer
os.makedirs("models", exist_ok=True)
joblib.dump(model, 'models/personality_model.pkl')
joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
print("Model and vectorizer saved.")
