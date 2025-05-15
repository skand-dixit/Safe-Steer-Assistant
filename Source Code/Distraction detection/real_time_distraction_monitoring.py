import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model # type: ignore
import winsound  

# Load model
model = load_model('distraction_detection_model.h5')

# Class labels
class_names = {
    0: 'DangerousDriving',
    1: 'Distracted',
    2: 'Drinking',
    3: 'SafeDriving',
    4: 'SleepyDriving',
    5: 'Yawn'
}

# Parameters
input_size = (224, 224)
alert_classes = [0, 1, 2, 4, 5]  # Classes other than SafeDriving

# Initialize webcam
cap = cv2.VideoCapture(0)

print("Starting real-time monitoring... Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_frame = cv2.resize(rgb_frame, input_size)
    input_frame = resized_frame.astype('float32') / 255.0
    input_frame = np.expand_dims(input_frame, axis=0)

    # Predict
    prediction = model.predict(input_frame)[0]
    pred_class = np.argmax(prediction)
    confidence = prediction[pred_class]
    label = class_names[pred_class]

    # Display results
    display_text = f"{label} ({confidence:.2f})"
    color = (0, 255, 0) if pred_class == 3 else (0, 0, 255)  # Green for safe, red for alert

    cv2.putText(frame, display_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow('Driver Monitoring', frame)

    # Beep if not SafeDriving
    if pred_class in alert_classes and confidence > 0.70:
        winsound.Beep(1000, 200)  # Beep: frequency=1000Hz, duration=200ms

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
