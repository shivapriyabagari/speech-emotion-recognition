import librosa
import numpy as np
import joblib

# Load trained model and scaler
model = joblib.load("models/emotion_model.pkl")
scaler = joblib.load("models/scaler.pkl")

emotion_labels = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# Test audio file
audio_path = "dataset/raw/Audio_Speech_Actors_01-24_16k/Actor_01/03-01-01-01-01-01-01.wav"

# Load audio
audio, sample_rate = librosa.load(audio_path, sr=None)

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

# Predict
prediction = model.predict(features)

print("Predicted Emotion:", prediction[0])

# Probability
probabilities = model.predict_proba(features)

confidence = np.max(probabilities) * 100

print("Confidence:", round(confidence, 2), "%")