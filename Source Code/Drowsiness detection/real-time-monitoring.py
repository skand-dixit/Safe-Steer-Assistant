import cv2
import numpy as np
import tensorflow as tf
import time
import threading
import winsound  # For Windows audible alert

# Load the trained model
model = tf.keras.models.load_model('drowsiness_detection_model.h5')

# Load Haarcascade models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Parameters
img_size = 224
color_type = 3  # RGB
fps_target = 15
alert_threshold = 0.8  # Confidence threshold for 'Closed' detection

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

prev_time = 0
beep_on = False

def continuous_beep():
    while beep_on:
        winsound.Beep(2500, 500)  # 500ms beep
        time.sleep(0.1)  # Short gap

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Control the FPS
    current_time = time.time()
    if (current_time - prev_time) < 1.0 / fps_target:
        continue
    prev_time = current_time

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    status = 'Open'

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        eye_predictions = []

        for (ex, ey, ew, eh) in eyes:
            eye_img = roi_color[ey:ey+eh, ex:ex+ew]
            eye_img = cv2.resize(eye_img, (img_size, img_size))
            eye_img = cv2.cvtColor(eye_img, cv2.COLOR_BGR2RGB)
            eye_img = np.array(eye_img) / 255.0
            eye_img = eye_img.reshape(1, img_size, img_size, color_type)

            prediction = model.predict(eye_img, verbose=0)
            pred_class = 'Open' if prediction[0][0] > 0.5 else 'Closed'
            eye_predictions.append(pred_class)

            # Draw rectangle around the eye
            color = (0, 255, 0) if pred_class == 'Open' else (0, 0, 255)
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), color, 2)

        if eye_predictions.count('Closed') >= 2:  # Both eyes closed
            status = 'Closed'
            if not beep_on:
                beep_on = True
                threading.Thread(target=continuous_beep, daemon=True).start()
        else:
            beep_on = False  # Eyes open, stop beep

    # Display status on frame
    label = f"Status: {status}"
    color = (0, 255, 0) if status == 'Open' else (0, 0, 255)
    cv2.putText(frame, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Display the resulting frame
    cv2.imshow('Drowsiness Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

beep_on = False
time.sleep(0.5)  # Let any beeping thread finish
cap.release()
cv2.destroyAllWindows()
