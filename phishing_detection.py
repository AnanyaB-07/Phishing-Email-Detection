import re
import os
import pandas as pd
import numpy as np
import requests
import zipfile
import io

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from scipy.sparse import hstack

import matplotlib.pyplot as plt


# ============================================================
# 1. DOWNLOAD DATASET
# ============================================================

DATA_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"

print("Downloading dataset...")

try:
    response = requests.get(DATA_URL, timeout=30)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extractall("dataset")

    print("Dataset downloaded successfully.")

except Exception as e:
    print("Could not download dataset automatically.")
    print("Error:", e)
    exit()


# ============================================================
# 2. LOAD DATASET
# ============================================================

file_path = "dataset/SMSSpamCollection"

if not os.path.exists(file_path):
    print("Dataset file not found.")
    exit()

data = pd.read_csv(
    file_path,
    sep="\t",
    header=None,
    names=["label", "text"],
    encoding="latin-1"
)

print("\nDataset loaded.")
print("Number of messages:", len(data))


# ============================================================
# 3. CONVERT SPAM/HAM TO PHISHING/SAFE
# ============================================================

data["label"] = data["label"].map({
    "spam": "Phishing",
    "ham": "Safe"
})


# ============================================================
# 4. FEATURE EXTRACTION FUNCTIONS
# ============================================================

def count_urls(text):
    urls = re.findall(r'https?://[^\s]+|www\.[^\s]+', str(text), flags=re.IGNORECASE)
    return len(urls)

def count_ip_urls(text):
    urls = re.findall(r'https?://[^\s]+|www\.[^\s]+', str(text), flags=re.IGNORECASE)
    return sum(1 for url in urls if re.search(r'https?://(?:\d{1,3}\.){3}\d{1,3}', url))

def count_suspicious_keywords(text):
    keywords = [
        "urgent","verify","verification","password","account","login","security",
        "confirm","click","suspended","bank","winner","prize","free","claim",
        "limited","immediately","payment"
    ]
    text = str(text).lower()
    return sum(1 for word in keywords if word in text)

def message_length(text):
    return len(str(text))

def uppercase_ratio(text):
    text = str(text)
    if len(text) == 0:
        return 0
    return sum(1 for c in text if c.isupper()) / len(text)


# ============================================================
# 5. APPLY FEATURE EXTRACTION
# ============================================================

data["url_count"] = data["text"].apply(count_urls)
data["ip_url_count"] = data["text"].apply(count_ip_urls)
data["suspicious_keyword_count"] = data["text"].apply(count_suspicious_keywords)
data["msg_length"] = data["text"].apply(message_length)
data["uppercase_ratio"] = data["text"].apply(uppercase_ratio)

print("\nClass distribution:")
print(data["label"].value_counts())

print("\nExample data:")
print(data.head())


# ============================================================
# 6. SPLIT DATA
# ============================================================

X_text = data["text"]
y = data["label"]

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text, y, test_size=0.20, random_state=42, stratify=y
)


# ============================================================
# 7. TF-IDF FEATURES
# ============================================================

print("\nExtracting text features...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=10000,
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train_text)
X_test_tfidf = vectorizer.transform(X_test_text)


# ============================================================
# 8. NUMERIC FEATURES
# ============================================================

train_indices = X_train_text.index
test_indices = X_test_text.index

numeric_features = ["url_count","ip_url_count","suspicious_keyword_count","msg_length","uppercase_ratio"]

X_train_num = data.loc[train_indices, numeric_features].values
X_test_num  = data.loc[test_indices, numeric_features].values


# ============================================================
# 9. COMBINE FEATURES
# ============================================================

X_train = hstack([X_train_tfidf, X_train_num])
X_test  = hstack([X_test_tfidf, X_test_num])


# ============================================================
# 10. TRAIN MODELS
# ============================================================

print("\nTraining Logistic Regression...")
log_model = LogisticRegression(max_iter=1000, class_weight="balanced")
log_model.fit(X_train, y_train)

print("Training Naive Bayes...")
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)


# ============================================================
# 11. EVALUATE MODELS
# ============================================================

def evaluate_model(model, name):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{name} Accuracy: {acc*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Safe","Phishing"]))
    cm = confusion_matrix(y_test, y_pred, labels=["Safe","Phishing"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Safe","Phishing"])
    disp.plot()
    plt.title(f"{name} - Confusion Matrix")
    plt.tight_layout()
    plt.show()

evaluate_model(log_model, "Logistic Regression")
evaluate_model(nb_model, "Naive Bayes")


# ============================================================
# 12. TEST NEW EMAILS
# ============================================================

def predict_email(email, model):
    text_features = vectorizer.transform([email])
    url_features = np.array([[
        count_urls(email),
        count_ip_urls(email),
        count_suspicious_keywords(email),
        message_length(email),
        uppercase_ratio(email)
    ]])
    features = hstack([text_features, url_features])
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    confidence = probability[model.classes_.tolist().index(prediction)] * 100
    return prediction, confidence

print("\n========================================")
print("TESTING NEW EMAILS")
print("========================================")

test_emails = [
    """URGENT! Your bank account has been suspended.
    Verify your password immediately by clicking
    http://192.168.1.10/login""",
    """Hi, our meeting is scheduled for tomorrow at
    10 AM. Please bring the project report."""
]

for email in test_emails:
    pred, conf = predict_email(email, log_model)
    print("\nEmail:")
    print(email.strip())
    print("\nPrediction:", pred)
    print(f"Confidence: {conf:.2f}%")
