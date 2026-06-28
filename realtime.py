import numpy as np
import sounddevice as sd
import librosa
import joblib
import scipy.io.wavfile as wav

# Load model + scaler
model = joblib.load("emotion_model.pkl")
scaler = joblib.load("scaler.pkl")

SAMPLE_RATE = 22050
DURATION = 4  # seconds


def extract_features(audio, sr):
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
    print("\n🎤 Recording... Speak now!")

    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32'
    )

    sd.wait()

    return np.squeeze(audio)


def predict_emotion(audio):
    features = extract_features(audio, SAMPLE_RATE)
    features = scaler.transform([features])

    prediction = model.predict(features)[0]
    probs = model.predict_proba(features)[0]

    confidence = np.max(probs)

    print("\n========================")
    print("Emotion:", prediction)
    print("Confidence:", round(confidence, 2))
    print("========================\n")


# MAIN LOOP
while True:
    input("Press ENTER to record (Ctrl+C to exit)")
    
    audio = record_audio()
    predict_emotion(audio)