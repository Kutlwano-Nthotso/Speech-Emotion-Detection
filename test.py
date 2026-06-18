import librosa
import os

# pick one audio file
file_path = "data/Actor_01/03-01-01-01-01-01-01.wav"

audio, sr = librosa.load(file_path)

print("Loaded successfully")
print("Sample rate:", sr)
print("Audio length:", len(audio))