import numpy as np
import librosa
import joblib
import sys
import os


# -----------------------------
# Load trained model + scaler
# -----------------------------
model = joblib.load("emotion_model.pkl")
scaler = joblib.load("scaler.pkl")


# -----------------------------
# Emotion decoder (optional)
# -----------------------------
def decode_emotion(pred):
    return pred


# -----------------------------
# Feature extraction
# -----------------------------
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


# -----------------------------
# Predict function
# -----------------------------
def predict(file_path):

    features = extract_features(file_path)
    features = scaler.transform([features])

    prediction = model.predict(features)[0]
    probs = model.predict_proba(features)[0]

    confidence = np.max(probs)
    emotion = model.classes_[np.argmax(probs)]

    print("\n========================")
    print("File:", file_path)
    print("Emotion:", emotion)
    print("Confidence:", round(confidence, 2))
    print("========================\n")


# -----------------------------
# MAIN LOGIC
# -----------------------------
if __name__ == "__main__":

    if len(sys.argv) > 1:

        file_path = sys.argv[1]

        if os.path.exists(file_path):
            predict(file_path)
        else:
            print("File not found:", file_path)

    else:

        print("Emotion Prediction Mode")
        print("Enter path to .wav file or type 'exit'")

        while True:
            file_path = input(">> ")

            if file_path.lower() == "exit":
                break

            if os.path.exists(file_path):
                predict(file_path)
            else:
                print("File not found. Try again.")