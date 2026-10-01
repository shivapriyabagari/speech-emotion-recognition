from flask import Flask, render_template, request
import librosa
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load("models/emotion_model.pkl")
scaler = joblib.load("models/scaler.pkl")

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "audio" not in request.files:
        return "No audio file uploaded"

    file = request.files["audio"]

    if file.filename == "":
        return "No file selected"

    file_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )

    file.save(file_path)

    # Load audio
    audio, sample_rate = librosa.load(
        file_path,
        sr=None
    )

    # Extract MFCC
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40
    )

    # Convert to feature vector
    features = np.mean(mfcc.T, axis=0)

    # Scale features
    features = scaler.transform([features])

    # Predict emotion
    prediction = model.predict(features)[0]

    # Confidence
    probabilities = model.predict_proba(features)
    confidence = np.max(probabilities) * 100

    return render_template(
        "result.html",
        emotion=prediction,
        confidence=round(confidence, 2)
    )


if __name__ == "__main__":
    app.run(debug=True)