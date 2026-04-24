import os
import pickle
import re
import string
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "model.pkl"
VECTORIZER_PATH = PROJECT_ROOT / "vectorizer.pkl"


DATASET = [
    ("Congratulations! You have won a free iPhone. Click now.", "spam"),
    ("Win cash prizes now!!! Reply WIN to claim.", "spam"),
    ("You have been selected for a $1000 Walmart gift card.", "spam"),
    ("URGENT! Your account is compromised. Verify immediately.", "spam"),
    ("Limited offer! Buy now and get 70% off.", "spam"),
    ("Free entry in 2 a weekly competition. Text WIN to 80086.", "spam"),
    ("Call this number to claim your reward.", "spam"),
    ("Exclusive deal just for you. Visit our site today.", "spam"),
    ("Hey, are we still meeting at 6 pm?", "ham"),
    ("Please send me the report by tomorrow morning.", "ham"),
    ("Your OTP is 458921. Do not share it.", "ham"),
    ("Happy birthday! Wishing you a great day.", "ham"),
    ("Can you call me when you are free?", "ham"),
    ("I will be late to the office today.", "ham"),
    ("Let us grab lunch after the meeting.", "ham"),
    ("Reminder: doctor appointment at 4 PM.", "ham"),
    ("Project deadline has been extended to next week.", "ham"),
    ("Thanks for your help yesterday.", "ham"),
    ("Cheap meds available online, no prescription needed!", "spam"),
    ("Get rich quick with this one simple trick.", "spam"),
    ("You have won free movie tickets, reply YES now.", "spam"),
    ("Act now to unlock your bonus reward points.", "spam"),
    ("Final warning: payment failed, update your billing details.", "spam"),
    ("Your package is waiting. Confirm address through this link.", "spam"),
    ("Are you joining the team call at 3 PM?", "ham"),
    ("Please review the pull request when you get time.", "ham"),
    ("Dinner is ready, come downstairs.", "ham"),
    ("Meeting moved to Friday due to client request.", "ham"),
    ("I shared the slides in our group chat.", "ham"),
    ("Can you pick up groceries on your way back?", "ham"),
]


def preprocess_text(text: str) -> str:
    """Lowercase, remove punctuation, and remove stopwords."""
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    tokens = [word for word in text.split() if word not in ENGLISH_STOP_WORDS]
    return " ".join(tokens)


def train_and_save() -> None:
    texts = [item[0] for item in DATASET]
    labels = [item[1] for item in DATASET]

    processed_texts = [preprocess_text(text) for text in texts]

    vectorizer = TfidfVectorizer()
    model = LogisticRegression(max_iter=1000, random_state=42)

    # Fit vectorizer and model separately so they can be stored as two pickle files.
    x_train = vectorizer.fit_transform(processed_texts)
    model.fit(x_train, labels)

    with open(MODEL_PATH, "wb") as model_file:
        pickle.dump(model, model_file)

    with open(VECTORIZER_PATH, "wb") as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Vectorizer saved to: {VECTORIZER_PATH}")


if __name__ == "__main__":
    os.makedirs(PROJECT_ROOT, exist_ok=True)
    train_and_save()
