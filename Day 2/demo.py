import face_recognition
import cv2
import torch
import os
import numpy as np



path = os.path.join(os.path.dirname(__file__), 'known_faces', 'obama.jpg')
img = cv2.imread(path)


# === Step 2: Load YOLOv5 Model ===

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)
results = model(img)
results.print()
results.show()

# === Step 3: Start Webcam for Face Authentication ===
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    raise RuntimeError("[ERROR] Webcam not detected or not accessible.")

face_authenticated = False
print("[INFO] Starting face authentication...")

while True:
    ret, frame = video_capture.read()
    if not ret:
        continue

    # Resize and convert to RGB
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB).astype('uint8')

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces([known_encoding], face_encoding) # type: ignore
        if True in matches:
            print("[ACCESS GRANTED] Face matched.")
            face_authenticated = True
            break

    if face_authenticated:
        break

# === Step 4: Real-Time Object Detection ===
print("[INFO] Starting real-time object detection... Press 'q' to quit.")

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Convert BGR to RGB and ensure uint8 format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype('uint8')

    # YOLOv5 Detection
    results = model(rgb_frame)

    # Render detection boxes
    annotated_frame = results.render()[0]

    # Print object info to console
    for *box, conf, cls in results.xyxy[0]:
        label = model.names[int(cls)]
        print(f"[DETECTED] {label} ({conf:.2f})")

    # Show result in window
    cv2.imshow("Object Detection (Press 'q' to quit)", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === Step 5: Cleanup ===
video_capture.release()
cv2.destroyAllWindows()
