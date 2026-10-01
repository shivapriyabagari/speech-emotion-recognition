import librosa

audio_path = "dataset/raw/Audio_Speech_Actors_01-24_16k/Actor_01/03-01-01-01-01-01-01.wav"

audio, sample_rate = librosa.load(audio_path, sr=None)

print("Audio loaded successfully!")
print("Sample rate:", sample_rate)
print("Audio samples:", len(audio))
print("Duration:", len(audio) / sample_rate, "seconds")