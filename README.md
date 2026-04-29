# Spam Text Detector (Mini ML)

A lightweight Flask web app that classifies SMS/text messages as **Spam** or **Not Spam** using NLP + machine learning.

## Features

- Spam/Not Spam prediction with probability
- Confidence score and risk level (`low`, `medium`, `high`)
- Text preprocessing:
  - lowercasing
  - punctuation removal
  - stopword removal
- TF-IDF vectorization + Logistic Regression model
- Quick sample message buttons in UI
- Health endpoint and model metadata endpoint

## Tech Stack

- Python
- Flask
- scikit-learn
- HTML + CSS + Bootstrap

## Project Structure

```text
.
|-- app.py
|-- model.pkl
|-- vectorizer.pkl
|-- model_info.json
|-- requirements.txt
|-- templates/
|   `-- index.html
|-- static/
|   `-- style.css
`-- utils/
    |-- train_model.py
    `-- evaluate_model.py
```

## Setup

1. Create and activate a virtual environment (optional but recommended).
2. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

3. Train model and generate artifacts:

```bash
python utils/train_model.py
```

4. Run app:

```bash
python app.py
```

5. Open in browser:

- http://127.0.0.1:5000

## API

### `POST /predict`

Request body:

```json
{
  "text": "Win a free iPhone now"
}
```

Response:

```json
{
  "label": "Spam",
  "probability": 86.52,
  "confidence": 86.52,
  "risk_level": "high"
}
```

Notes:

- The `text` field must be a string.
- Messages longer than 5000 characters are rejected with `413`.

### `GET /health`

Checks service and model readiness.

Response also includes `spam_threshold` and `max_text_length`.

### `GET /model-info`

Returns model metadata from `model_info.json` plus the active runtime settings.

