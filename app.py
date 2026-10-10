
import os
import json
import logging

import numpy as np
import tensorflow as tf

from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

MODEL_PATH = os.getenv(
    "MODEL_PATH", "artifacts/incident_model.keras"
)
TOKENIZER_PATH = os.getenv(
    "TOKENIZER_PATH", "artifacts/tokenizer.json"
)
LABELS_PATH = os.getenv(
    "LABELS_PATH", "artifacts/labels.json"
)
MAX_LEN = int(os.getenv("MAX_LEN", "30"))

model = None
tokenizer = None
labels = None
load_error = None

# Load all model artifacts during application startup.
try:
    model = tf.keras.models.load_model(MODEL_PATH)

    with open(TOKENIZER_PATH, encoding="utf-8") as f:
        tokenizer = tokenizer_from_json(f.read())

    with open(LABELS_PATH, encoding="utf-8") as f:
        labels = json.load(f)

    app.logger.info("Model artifacts loaded successfully")

except Exception as exc:
    load_error = str(exc)
    app.logger.exception("Failed to load model artifacts")


@app.get("/")
def home():
    return jsonify({
        "application": "IT Incident Classification API",
        "version": "1.0",
        "endpoints": [
            "/health",
            "/model-info",
            "/predict"
        ],
    })


@app.get("/health")
def health():
    if model is None or tokenizer is None or labels is None:
        return jsonify({
            "status": "unhealthy",
            "detail": load_error or "Model unavailable",
        }), 503

    return jsonify({
        "status": "ok",
        "model_loaded": True,
    }), 200


@app.get("/model-info")
def model_info():
    if model is None or labels is None:
        return jsonify({"error": "Model unavailable"}), 503

    return jsonify({
        "model_type": "Bidirectional LSTM",
        "categories": labels,
        "max_sequence_length": MAX_LEN,
    }), 200


@app.post("/predict")
def predict():
    if model is None or tokenizer is None or labels is None:
        return jsonify({"error": "Model is not loaded"}), 503

    if not request.is_json:
        return jsonify({
            "error": "Content-Type must be application/json"
        }), 415

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Request body must be a JSON object"
        }), 400

    text = data.get("text")

    if not isinstance(text, str) or not text.strip():
        return jsonify({
            "error": "'text' must be a non-empty string"
        }), 400

    if len(text) > 2000:
        return jsonify({
            "error": "Text must be 2000 characters or fewer"
        }), 400

    # Apply the same preprocessing used during training.
    sequences = tokenizer.texts_to_sequences([text.strip()])
    padded = pad_sequences(
        sequences,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post",
    )

    probabilities = model.predict(padded, verbose=0)[0]
    best_index = int(np.argmax(probabilities))

    scores = sorted(
        [
            {
                "category": labels[i],
                "confidence": round(float(probabilities[i]), 4),
            }
            for i in range(len(labels))
        ],
        key=lambda item: item["confidence"],
        reverse=True,
    )

    app.logger.info(
        "Prediction completed: category=%s",
        labels[best_index],
    )

    return jsonify({
        "predicted_category": labels[best_index],
        "confidence": round(
            float(probabilities[best_index]), 4
        ),
        "probabilities": scores,
    }), 200


if __name__ == "__main__":
    # Debug mode is disabled.
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),
        debug=False,
    )
