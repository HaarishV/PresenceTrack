# PresenceTrack

PresenceTrack is a real-time attendance system that leverages computer vision and deep learning to automate attendance marking. Using a combination of YOLOv8 for fast face detection and face_recognition for identification, it provides a highly accurate and efficient solution for classrooms, offices, and events. Attendance data is stored securely in Firebase Firestore.

## Features

- Real-time face detection using YOLOv8

- Accurate face recognition with face encodings

- Automatic attendance marking in Firebase Firestore

- GPU support for faster processing

- Supports multiple users simultaneously

- Easy addition of new students with face data capture

## Installation
<h3>1.Create a virtual environment and activate it:</h3>

```
conda create -n track python=3.10
conda activate track
```
<h3>Install dependencies:</h3>

```
pip install -r requirements.txt
```
<h3>Add your Firebase service account key:</h3>
Place serviceAccountKey.json in the project root. (Do not push it to GitHub)

## Usage
### Step 1: Add Students
Run the script to capture faces and save student data:
```
python add_students.py
```
### Step 2: Encode Faces
```
python encode.py
```
This will generate face_encodings.pkl used for recognition.

### Step 3: Mark Attendance
```
python mark_attendance.py
```
- Real-time detection and marking will start.

- Press q to quit.

- Attendance is stored in Firebase Firestore by date.

### Step 4: View Attendance
```
python view.attendance.py
```

