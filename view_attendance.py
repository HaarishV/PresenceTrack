import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Input date to view attendance
date = input("Enter the date to view attendance (YYYY-MM-DD): ").strip()

attendance_ref = db.collection("attendance").document(date)
doc = attendance_ref.get()

if doc.exists:
    data = doc.to_dict()
    if not data:
        print(f"No attendance records found for {date}.")
    else:
        print(f"📅 Attendance for {date}:")
        print("-" * 40)
        for reg_no, info in data.items():
            name = info.get("name", "Unknown")
            time = info.get("time", "Unknown")
            status = info.get("status", "Unknown")
            print(f"{reg_no} | {name} | {status} at {time}")
        print("-" * 40)
else:
    print(f"No attendance found for {date}.")
