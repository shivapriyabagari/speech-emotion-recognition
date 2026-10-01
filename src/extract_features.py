import os
import librosa
import numpy as np
import pandas as pd

DATASET_PATH = "dataset/raw/Audio_Speech_Actors_01-24_16k"

features = []

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

for actor in os.listdir(DATASET_PATH):

    actor_path = os.path.join(DATASET_PATH, actor)

    if not os.path.isdir(actor_path):
        continue

    for filename in os.listdir(actor_path):

        if not filename.endswith(".wav"):
            continue

        file_path = os.path.join(actor_path, filename)

        # Extract emotion from filename
        parts = filename.split("-")
        emotion_code = parts[2]
        emotion = emotion_labels[emotion_code]

        # Load audio
        audio, sample_rate = librosa.load(file_path, sr=None)

        # Extract MFCC
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=40
        )

        # Convert to fixed-size feature vector
        mfcc_features = np.mean(mfcc.T, axis=0)

        features.append(
            [*mfcc_features, emotion]
        )

        print("Processed:", filename)

# Create column names
columns = [f"MFCC_{i+1}" for i in range(40)]
columns.append("emotion")

# Create DataFrame
df = pd.DataFrame(features, columns=columns)

# Save CSV
df.to_csv("dataset/features.csv", index=False)

print("\nFeature extraction completed!")
print("Total audio files:", len(df))
print("CSV saved as: dataset/features.csv")