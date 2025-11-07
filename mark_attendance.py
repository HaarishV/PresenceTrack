# ------------------- Ignore Warnings -------------------
import warnings
warnings.filterwarnings("ignore")

import cv2
import face_recognition
import pickle
from ultralytics import YOLO
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
import torch
import numpy as np
import os
import pandas as pd


# ------------------- Firebase Setup -------------------
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# ------------------- Load Known Faces -------------------
with open("face_encodings.pkl", "rb") as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_names = data["names"]

# ------------------- GPU Check -------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ------------------- YOLOv8 Face Detection -------------------
model = YOLO("yolov8n-face.pt")  
#model.fuse()
model.to(device)

# ------------------- Attendance Tracking -------------------
marked_students = set()
attendance_records = []

def mark_attendance(name):
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")
    reg_no, actual_name = name.split("_", 1)
    attendance_ref = db.collection("attendance").document(date)

    doc = attendance_ref.get()
    if doc.exists:
        existing_data = doc.to_dict()
        if reg_no in existing_data:
            print(f"⏳ {actual_name} already marked at {existing_data[reg_no]['time']}")
            marked_students.add(name)
            return

    # Mark in Firebase
    attendance_ref.set({
        reg_no: {"name": actual_name, "time": time, "status": "Present"}
    }, merge=True)
    print(f"✅ {actual_name} ({reg_no}) marked present at {time}")
    marked_students.add(name)

    # Keep in memory for Excel export
    attendance_records.append({
        "Date": date,
        "Time": time,
        "Register No": reg_no,
        "Name": actual_name,
        "Status": "Present"
    })

def export_attendance_to_excel():
    if not attendance_records:
        print("No attendance to export.")
        return
    
    # Ensure Attendance folder exists
    os.makedirs("Attendance", exist_ok=True)
    
    # File name based on current date
    date_str = datetime.now().strftime('%Y-%m-%d')
    file_name = f"Attendance/Attendance_{date_str}.xlsx"
    
    # If file exists, append new records; otherwise create new
    if os.path.exists(file_name):
        df_existing = pd.read_excel(file_name)
        df_new = pd.DataFrame(attendance_records)
        df_combined = pd.concat([df_existing, df_new]).drop_duplicates(subset=["Register No"])
        df_combined.to_excel(file_name, index=False)
    else:
        df = pd.DataFrame(attendance_records)
        df.to_excel(file_name, index=False)
    
    print(f"✅ Attendance exported to {file_name}")

# ------------------- Start Camera -------------------
cam = cv2.VideoCapture(0)
print("📡 Starting real-time attendance... Press 'q' to quit.")

while True:
    ret, frame = cam.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # YOLO detection
    results = model(rgb_frame, verbose=False)[0]
    boxes = results.boxes.xyxy.cpu().numpy()  # (x1, y1, x2, y2)

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)

        # Skip very small boxes
        if (x2 - x1) < 20 or (y2 - y1) < 20:
            continue

        # face_recognition expects (top, right, bottom, left)
        face_enc = face_recognition.face_encodings(
            rgb_frame, known_face_locations=[(y1, x2, y2, x1)]
        )

        name = "Unknown"
        if face_enc:
            face_enc = face_enc[0]
            distances = face_recognition.face_distance(known_encodings, face_enc)
            if len(distances) > 0:
                best_idx = np.argmin(distances)
                if distances[best_idx] < 0.5:
                    name = known_names[best_idx]
                    if name not in marked_students:
                        mark_attendance(name)

        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, name, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("PresenceTrack", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

# ------------------- Export to Excel -------------------
export_attendance_to_excel()
