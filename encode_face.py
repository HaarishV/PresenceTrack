# ------------------- Ignore Warnings -------------------
import warnings
warnings.filterwarnings("ignore")


import os
import face_recognition
import numpy as np
import pickle

data_path = "faces"
known_encodings = []
known_names = []

for folder in os.listdir(data_path):
    folder_path = os.path.join(data_path, folder)
    if not os.path.isdir(folder_path):
        continue

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        image = face_recognition.load_image_file(img_path)
        encodings = face_recognition.face_encodings(image)

        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(folder)

data = {"encodings": known_encodings, "names": known_names}

with open("face_encodings.pkl", "wb") as f:
    pickle.dump(data, f)

print(f"✅ Encoded {len(known_names)} faces and saved as face_encodings.pkl")
