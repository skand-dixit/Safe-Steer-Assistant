import cv2
import numpy as np
import tensorflow as tf
import time
import threading
import winsound
from collections import deque

# Load models
behavior_model = tf.keras.models.load_model('distraction_detection_model.h5')
eye_model = tf.keras.models.load_model('drowsiness_detection_model.h5')

# Load Haarcascade classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Class labels for driver behavior
class_names = {
    0: 'DangerousDriving',
    1: 'Distracted',
    2: 'Drinking',
    3: 'SafeDriving',
    4: 'SleepyDriving',
    5: 'Yawn'
}

# Parameters
input_size_behavior = (224, 224)
img_size_eye = 224
color_type = 3  # RGB
alert_behavior_classes = [0, 1, 2, 4, 5]
alert_threshold_behavior = 0.70

# Cooldown configuration
last_drowsy_beep_time = 0
last_behavior_beep_time = 0
beep_cooldown_seconds = 1

# Eye state history buffer for smoothing
eye_state_history = deque(maxlen=10)  # Store last 10 predictions

# Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open webcam.")
    exit()

# Beeping threads
def beep_drowsy():
    winsound.Beep(1000, 200)  # long beep

def beep_behavior():
    winsound.Beep(1500, 150)

# Function to draw enhanced text boxes
def draw_text_box(img, text, position, box_color, text_color=(255, 255, 255), font_scale=0.7, thickness=2):
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x, y = position
    box_coords = ((x - 5, y - text_height - 10), (x + text_width + 5, y + baseline + 5))

    overlay = img.copy()
    cv2.rectangle(overlay, box_coords[0], box_coords[1], box_color, -1)
    alpha = 0.6  # Transparency factor
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)

print("Monitoring started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    current_time = time.time()

    # Driver Behavior Prediction
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_frame = cv2.resize(rgb_frame, input_size_behavior)
    input_frame = resized_frame.astype('float32') / 255.0
    input_frame = np.expand_dims(input_frame, axis=0)

    prediction = behavior_model.predict(input_frame, verbose=0)[0]
    pred_class = np.argmax(prediction)
    confidence = prediction[pred_class]
    behavior_label = class_names[pred_class]

    # Display behavior prediction
    draw_text_box(frame, f"Behavior: {behavior_label} ({confidence:.2f})", (10, 30),
                  (0, 255, 0) if pred_class == 3 else (0, 0, 255))

    # Eye State Detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    current_eye_states = []

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            eye_img = roi_color[ey:ey+eh, ex:ex+ew]
            eye_img = cv2.resize(eye_img, (img_size_eye, img_size_eye))
            eye_img = cv2.cvtColor(eye_img, cv2.COLOR_BGR2RGB)
            eye_img = np.array(eye_img) / 255.0
            eye_img = eye_img.reshape(1, img_size_eye, img_size_eye, color_type)

            eye_pred = eye_model.predict(eye_img, verbose=0)
            is_drowsy = 0 if eye_pred[0][0] > 0.5 else 1  # 1 = drowsy
            current_eye_states.append(is_drowsy)

    # Add average state of this frame to history
    if current_eye_states:
        frame_drowsy_status = 1 if sum(current_eye_states) >= 2 else 0
        eye_state_history.append(frame_drowsy_status)

    # Determine smoothed status
    drowsy_status = 'Non-Drowsy'
    if eye_state_history.count(1) >= 6:  # At least 6 of last 10 frames drowsy
        drowsy_status = 'Drowsy'
        draw_text_box(frame, "ALERT: DROWSINESS DETECTED", (10, 135), (0, 0, 255))
        if current_time - last_drowsy_beep_time > beep_cooldown_seconds:
            threading.Thread(target=beep_drowsy, daemon=True).start()
            last_drowsy_beep_time = current_time

    # Display drowsy status
    draw_text_box(frame, f"Drowsiness: {drowsy_status}", (10, 65),
                  (0, 255, 0) if drowsy_status == 'Non-Drowsy' else (0, 0, 255))

    # Alert for Driver Behavior
    if pred_class in alert_behavior_classes and confidence > alert_threshold_behavior:
        draw_text_box(frame, f"ALERT: {behavior_label.upper()}", (10, 100), (0, 0, 255))
        if current_time - last_behavior_beep_time > beep_cooldown_seconds:
            threading.Thread(target=beep_behavior, daemon=True).start()
            last_behavior_beep_time = current_time

    # Show frame
    cv2.imshow('Driver Monitoring System', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
