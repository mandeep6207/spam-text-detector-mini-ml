import pickle
from pathlib import Path

from sklearn.metrics import classification_report

from train_model import DATASET, preprocess_text


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "model.pkl"
VECTORIZER_PATH = PROJECT_ROOT / "vectorizer.pkl"


def evaluate() -> None:
    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)

    with open(VECTORIZER_PATH, "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    texts = [preprocess_text(text) for text, _ in DATASET]
    labels = [label for _, label in DATASET]

    x_data = vectorizer.transform(texts)
    predictions = model.predict(x_data)

    print("Quick evaluation on sample dataset")
    print(classification_report(labels, predictions, digits=3))


if __name__ == "__main__":
    evaluate()
