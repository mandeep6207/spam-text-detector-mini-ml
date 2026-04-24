import pickle
import re
import string
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
VECTORIZER_PATH = BASE_DIR / "vectorizer.pkl"

app = Flask(__name__)


def preprocess_text(text: str) -> str:
    """Lowercase, remove punctuation, and remove stopwords."""
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    tokens = [word for word in text.split() if word not in ENGLISH_STOP_WORDS]
    return " ".join(tokens)


with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)

with open(VECTORIZER_PATH, "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "Please provide a text message."}), 400

    processed_text = preprocess_text(text)
    features = vectorizer.transform([processed_text])
    probabilities = model.predict_proba(features)[0]
    classes = list(model.classes_)

    spam_index = classes.index("spam")
    spam_probability = float(probabilities[spam_index])

    label = "Spam" if spam_probability >= 0.5 else "Not Spam"

    return jsonify(
        {
            "label": label,
            "probability": round(spam_probability * 100, 2),
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
