
# Hand Gesture Recognition using Python and OpenCV

## Overview
A real-time hand gesture recognition project using **OpenCV** and **Keras (CNN)**.  
The system can detect gestures like **fist, palm, thumbs up**, etc., in real-time through a webcam.

## Folder Structure
hand_gesture_recognition/
├── gestures/ # Folder with gesture images
├── hand_gesture.py # Real-time detection script
├── train_model.py # Script to train CNN
├── utils.py # Helper functions
└── README.md

## Steps to Run
1. Install dependencies:
2. bash
pip install opencv-python tensorflow numpy
Prepare data:

Create folders inside gestures/ named after your gesture classes (e.g., fist, palm, thumbs_up).

Add images of hands performing each gesture.

Train model:

python train_model.py


Run real-time detection:

python hand_gesture.py


Press ESC to exit webcam.

Notes

Ensure good lighting for better recognition.

ROI rectangle in the webcam frame is where the hand should be placed.
