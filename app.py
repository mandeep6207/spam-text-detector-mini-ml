import pickle
import re
import string
import json
import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
VECTORIZER_PATH = BASE_DIR / "vectorizer.pkl"
MODEL_INFO_PATH = BASE_DIR / "model_info.json"
SPAM_THRESHOLD = 0.5
MAX_TEXT_LENGTH = 5000

app = Flask(__name__)


def preprocess_text(text: str) -> str:
    """Lowercase, remove punctuation, and remove stopwords."""
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    tokens = [word for word in text.split() if word not in ENGLISH_STOP_WORDS]
    return " ".join(tokens)


def get_spam_probability(probabilities, classes) -> float:
    """Return the probability for the spam class."""
    try:
        spam_index = list(classes).index("spam")
    except ValueError as exc:
        raise ValueError("Model does not expose a spam class.") from exc

    return float(probabilities[spam_index])


def get_risk_level(spam_probability: float) -> str:
    """Map a spam probability to a risk level."""
    if spam_probability >= 0.8:
        return "high"
    if spam_probability >= 0.5:
        return "medium"
    return "low"


with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)

with open(VECTORIZER_PATH, "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

MODEL_READY = model is not None and vectorizer is not None


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "model_ready": MODEL_READY,
            "spam_threshold": SPAM_THRESHOLD,
            "max_text_length": MAX_TEXT_LENGTH,
        }
    )


@app.route("/model-info", methods=["GET"])
def model_info():
    if not MODEL_INFO_PATH.exists():
        return jsonify({"error": "Model info not available. Retrain model first."}), 404

    with open(MODEL_INFO_PATH, "r", encoding="utf-8") as info_file:
        info = json.load(info_file)

    info["spam_threshold"] = SPAM_THRESHOLD
    info["max_text_length"] = MAX_TEXT_LENGTH
    info["model_ready"] = MODEL_READY
    return jsonify(info)


@app.route("/predict", methods=["POST"])
def predict():
    if not MODEL_READY:
        return jsonify({"error": "Model artifacts are not loaded."}), 503

    data = request.get_json(silent=True) or {}

    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    raw_text = data.get("text")
    if raw_text is not None and not isinstance(raw_text, str):
        return jsonify({"error": "The text field must be a string."}), 400

    text = (raw_text or "").strip()

    if not text:
        return jsonify({"error": "Please provide a text message."}), 400

    if len(text) > MAX_TEXT_LENGTH:
        return jsonify({"error": "Text message is too long."}), 413

    try:
        processed_text = preprocess_text(text)
        features = vectorizer.transform([processed_text])
        probabilities = model.predict_proba(features)[0]
        spam_probability = get_spam_probability(probabilities, model.classes_)
    except Exception:
        return jsonify({"error": "Prediction failed."}), 500

    label = "Spam" if spam_probability >= SPAM_THRESHOLD else "Not Spam"
    confidence = spam_probability if label == "Spam" else 1 - spam_probability

    risk_level = get_risk_level(spam_probability)

    return jsonify(
        {
            "label": label,
            "probability": round(spam_probability * 100, 2),
            "confidence": round(confidence * 100, 2),
            "risk_level": risk_level,
        }
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    debug_mode = os.getenv("FLASK_DEBUG", "1") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
