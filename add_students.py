# ------------------- Ignore Warnings -------------------
import warnings
warnings.filterwarnings("ignore")


import cv2
import os
import firebase_admin
from firebase_admin import credentials, firestore

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

reg_no = input("Enter Register Number: ").strip()
name = input("Enter Name: ").strip()
dept = input("Enter Department: ").strip()

folder_name = f"{reg_no}_{name}"
path = f"faces/{folder_name}"
os.makedirs(path, exist_ok=True)

cam = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

count = 0
print("📸 Capturing 20 face samples... Look around slightly to help model learn angles.")

while True:
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (250, 250))
        cv2.imwrite(f"{path}/{count}.jpg", face)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"Captured: {count}/20", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("Capturing Faces", frame)
    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 20:
        break

cam.release()
cv2.destroyAllWindows()

db.collection("students").document(reg_no).set({
    "reg_no": reg_no,
    "name": name,
    "department": dept
})

print(f"✅ {name} ({reg_no}, {dept}) added to database and face data saved.")
