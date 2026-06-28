
import os
import glob
import librosa
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


# -----------------------------
# 1. EMOTION FUNCTION
# -----------------------------
def get_emotion(file_name):
    try:
        code = file_name.split("-")[2]

        mapping = {
            "01": "neutral",
            "02": "calm",
            "03": "happy",
            "04": "sad",
            "05": "angry",
            "06": "fear",
            "07": "disgust",
            "08": "surprise"
        }

        return mapping.get(code, "unknown")

    except:
        return "unknown"


# -----------------------------
# 2. DATASET SETUP
# -----------------------------
data_path = "data"

X = []
y = []

print("Starting dataset processing...")


# -----------------------------
# 3. LOAD AUDIO FILES
# -----------------------------
files = glob.glob(os.path.join(data_path, "**", "*.wav"), recursive=True)

for file_path in files:

    print("Processing:", file_path)

    file_name = os.path.basename(file_path)

    audio, sr = librosa.load(file_path)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    mfcc_combined = np.hstack((
        np.mean(mfcc.T, axis=0),
        np.mean(mfcc_delta.T, axis=0),
        np.mean(mfcc_delta2.T, axis=0)
    ))

    X.append(mfcc_combined)

    emotion = get_emotion(file_name)
    y.append(emotion)

print("DONE")
print("Samples:", len(X))


# -----------------------------
# 4. PREPARE DATA
# -----------------------------
X = np.array(X)
y = np.array(y)

scaler = StandardScaler()
X = scaler.fit_transform(X)


# -----------------------------
# 5. TRAIN MODEL
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    max_depth=None
)

model.fit(X_train, y_train)


# -----------------------------
# 6. EVALUATE MODEL
# -----------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

import joblib

joblib.dump(model, "emotion_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model saved successfully!")