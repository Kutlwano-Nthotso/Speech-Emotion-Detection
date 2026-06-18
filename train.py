import os
import librosa
import numpy as np

data_path = "data"

X = []
y = []

print("Starting dataset processing...")

for actor in os.listdir(data_path):
    print("Processing:", actor)

    actor_path = os.path.join(data_path, actor)

    if os.path.isdir(actor_path):

        for file in os.listdir(actor_path):

            if file.endswith(".wav"):

                file_path = os.path.join(actor_path, file)

                audio, sr = librosa.load(file_path)

                mfcc = librosa.feature.mfcc(
                    y=audio,
                    sr=sr,
                    n_mfcc=13
                )

                mfcc_mean = np.mean(mfcc.T, axis=0)

                X.append(mfcc_mean)
                y.append(actor)

print("DONE")
print("Samples:", len(X))