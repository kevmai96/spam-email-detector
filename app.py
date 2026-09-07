import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)
from collections import Counter
import re, warnings

warnings.filterwarnings('ignore')

emails = [
    # spam
    ("WINNER! You have been selected to receive a FREE prize worth $1000. Click here NOW!", 1),
    ("Congratulations! You have won a brand new iPhone. Call us immediately.", 1),
    ("URGENT: Your account has been compromised. Verify your details immediately.", 1),
    ("FREE money waiting for you! Act now and receive $500 cash directly.", 1),
    ("You are a WINNER! Claim your lottery jackpot. Reply with your bank details.", 1),
    ("Hot singles in your area want to meet YOU tonight. Click here for FREE access.", 1),
    ("Limited time offer! Buy cheap Viagra online. No prescription needed.", 1),
    ("Your PayPal account has been suspended. Verify your information NOW.", 1),

    # ham
    ("Hey, are we still meeting for lunch tomorrow?", 0),
    ("Hi, I wanted to follow up on the report I sent you last week.", 0),
    ("Can you please send me the agenda for Thursday's team meeting?", 0),
    ("Just confirming that I received your invoice. Payment will be processed within 5 days.", 0),
    ("Happy birthday! Hope you have a wonderful day.", 0),
    ("The project deadline has been moved to next Friday.", 0),
    ("I have attached the presentation slides for your review.", 0),
    ("Thanks for the great work on the quarterly report.", 0),
]

df = pd.DataFrame(emails, columns=['text', 'label'])
print(f"Dataset ready: {len(df)} emails")

def clean_email(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', 'url', text)
    text = re.sub(r'\S+@\S+', 'email', text)
    text = re.sub(r'\$[\d,]+', 'money', text)
    text = re.sub(r'\b\d+\b', 'number', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text

df['clean_text'] = df['text'].apply(clean_email)

X = df['clean_text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

results = {}

# Naive Bayes
nb = Pipeline([
    ('tfidf', 
        TfidfVectorizer(
            stop_words='english', 
            max_features=3000, 
            ngram_range=(1,2)
        )
    ),
    ('clf', MultinomialNB(alpha=0.1))
])
nb.fit(X_train, y_train)
nb_pred = nb.predict(X_test)
results['Naive Bayes'] = {
    'pipeline': nb,
    'predictions': nb_pred,
    'accuracy': accuracy_score(y_test, nb_pred),
    'precision': precision_score(y_test, nb_pred),
    'recall': recall_score(y_test, nb_pred),
    'f1': f1_score(y_test, nb_pred)
}

# Logistic Regression
lr = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=3000, ngram_range=(1,2))),
    ('clf', LogisticRegression(max_iter=1000, C=1.0, random_state=42))
])
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
results['Logistic Regression'] = {
    'pipeline': lr,
    'predictions': lr_pred,
    'accuracy': accuracy_score(y_test, lr_pred),
    'precision': precision_score(y_test, lr_pred),
    'recall': recall_score(y_test, lr_pred),
    'f1': f1_score(y_test, lr_pred)
}

# Random Forest
rf = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=3000, ngram_range=(1,2))),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
results['Random Forest'] = {
    'pipeline': rf,
    'predictions': rf_pred,
    'accuracy': accuracy_score(y_test, rf_pred),
    'precision': precision_score(y_test, rf_pred),
    'recall': recall_score(y_test, rf_pred),
    'f1': f1_score(y_test, rf_pred)
}

# Linear SVM
svm = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=3000, ngram_range=(1,2))),
    ('clf', LinearSVC(random_state=42, max_iter=2000, C=1.0))
])
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)
results['Linear SVM'] = {
    'pipeline': svm,
    'predictions': svm_pred,
    'accuracy': accuracy_score(y_test, svm_pred),
    'precision': precision_score(y_test, svm_pred),
    'recall': recall_score(y_test, svm_pred),
    'f1': f1_score(y_test, svm_pred)
}

# Best model
best_name = max(results, key=lambda x: results[x]['f1'])
best_pipeline = results[best_name]['pipeline']

print("\nBest model:", best_name)
print("Results:")

for name, r in results.items():
    print(f"{name:<20} | Accuracy: {r['accuracy']*100:.1f}% | F1: {r['f1']*100:.1f}%")

print("\nEnter your own email text to check for spam.")
print("Type 'quit' to exit.\n")

while True:
    try:
        user_email = input("Email text: ").strip()

        if user_email.lower() in ['quit', 'exit', 'q', '']:
            print("Exiting spam checker.")

            break

        clean = clean_email(user_email)
        pred = best_pipeline.predict([clean])[0]
        verdict = "SPAM" if pred == 1 else "LEGIT"

        print(f"Final say: {verdict}\n")
    except KeyboardInterrupt:
        print("\nExiting.")

        break
