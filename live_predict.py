import numpy as np
import librosa
import sounddevice as sd
import joblib
from scipy.io.wavfile import write

# Load model + scaler
model = joblib.load("emotion_model.pkl")
scaler = joblib.load("scaler.pkl")

SAMPLE_RATE = 22050
DURATION = 4
FILE_NAME = "recorded_audio.wav"


def extract_features(file_path):
    audio, sr = librosa.load(file_path)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    features = np.hstack((
        np.mean(mfcc.T, axis=0),
        np.mean(mfcc_delta.T, axis=0),
        np.mean(mfcc_delta2.T, axis=0)
    ))

    return features


def record_audio():
    print("\n🎤 Speak now...")

    recording = sd.rec(int(DURATION * SAMPLE_RATE),
                       samplerate=SAMPLE_RATE,
                       channels=1)

    sd.wait()

    write(FILE_NAME, SAMPLE_RATE, recording)
    print("✅ Recording saved!")


def predict():
    features = extract_features(FILE_NAME)
    features = scaler.transform([features])

    prediction = model.predict(features)[0]

    print("\n====================")
    print("🎯 Emotion:", prediction)
    print("====================\n")


while True:
    input("Press ENTER to record (Ctrl+C to stop)...")
    record_audio()
    predict()