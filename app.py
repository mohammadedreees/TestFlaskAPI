from flask import Flask, request, jsonify
import os
import torch
import librosa
import numpy as np
import tensorflow as tf
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import json
import tempfile

app = Flask(__name__)

# -----------------------------
# Load model and labels
# -----------------------------
MODEL_PATH = "quran_reciter_wav2vec.keras"
LABELS_PATH = "labels.json"

model = tf.keras.models.load_model(MODEL_PATH)

with open(LABELS_PATH) as f:
    labels = json.load(f)
label_map = {int(v): k for k, v in labels.items()}

# -----------------------------
# Load wav2vec2
# -----------------------------
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
wav2vec.eval()

# -----------------------------
# Embedding extraction
# -----------------------------
def extract_embedding(file_path):
    audio, _ = librosa.load(file_path, sr=16000)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = wav2vec(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding

# -----------------------------
# Prediction function
# -----------------------------
def predict_reciter(file_path):
    emb = extract_embedding(file_path)
    probs = model.predict(emb[np.newaxis, :])[0]
    pred_id = np.argmax(probs)
    return {
        "predicted_reciter": label_map[pred_id],
        "confidence": round(float(probs[pred_id]) * 100, 2)
    }

# -----------------------------
# Flask route
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        result = predict_reciter(tmp_path)
    finally:
        os.remove(tmp_path)

    return jsonify(result)

# -----------------------------
# Run locally
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)

