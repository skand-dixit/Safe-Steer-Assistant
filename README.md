🚗 Safe Steer Assistant

Safe Steer Assistant is a real-time driver monitoring system that detects drowsiness and distraction using deep learning and computer vision techniques. It uses trained CNN models to monitor the driver's facial cues and provides audible alerts to enhance road safety.

📁 Project Structure
SafeSteerAssistant/
│
├── Distraction detection/
│   ├── best_driver_model.keras
│   ├── best_finetuned_driver_model.keras
│   ├── Distraction Detection Model Training.ipynb
│   ├── distraction_detection_model.h5
│   └── real_time_distraction_monitoring.py
│
├── Drowsiness detection/
│   ├── Drowsiness Detection Model Training.ipynb
│   ├── drowsiness_detection_model.h5
│   ├── haarcascade_eye_tree_eyeglasses.xml
│   ├── haarcascade_eye.xml
│   ├── haarcascade_frontalface_default.xml
│   └── real-time-monitoring.py
│
├── Final Module/
│   ├── distraction_detection_model.h5
│   ├── drowsiness_detection_model.h5
│   └── real_time_monitoring.py
│
└── requirements.txt

🔍 Description

Safe Steer Assistant is divided into three major components:

Drowsiness Detection:

Trained CNN model to detect eye closure.

Uses Haarcascade classifiers for eye and face detection.

Real-time drowsiness detection in real-time-monitoring.py.


Distraction Detection:

CNN model trained to recognize attention loss or head turns.

Model training in Distraction Detection Model Training.ipynb.

Real-time inference in real_time_distraction_monitoring.py.


Final Module:

Unified script: real_time_monitoring.py

Loads both models and provides consolidated real-time monitoring and alerting.


✅ Features

Real-time webcam-based monitoring

Drowsiness detection via eye state

Distraction detection via attention classification

Alerts using audio cues to prevent accidents

Trained models in .h5 formats

Modular structure for training and deployment


💻 Installation
1. Clone the repository
git clone https://github.com/your-username/Safe-Steer-Assistant.git
cd Safe-Steer-Assistant

3. Install dependencies
pip install -r requirements.txt

5. Run the final monitoring script
cd "Final Module"
python real_time_monitoring.py

⚙️ Requirements
Python 3.7+
OpenCV
TensorFlow / Keras
NumPy
imutils
pygame or playsound (for audio alerts)
All dependencies are listed in requirements.txt

📊 Performance
Model	Accuracy
Drowsiness Detection	~95%
Distraction Detection	~92%

Tested on real-time webcam input at ~15–20 FPS.

🚀 Future Scope
Infrared camera support for night detection
Deployment on edge devices (like Raspberry Pi)
Integration with vehicle braking systems
Multi-face and multi-camera support
