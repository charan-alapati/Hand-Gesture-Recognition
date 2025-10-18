import cv2
import numpy as np

def preprocess_frame(frame, img_size):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (img_size, img_size))
    normalized = resized / 255.0
    return np.expand_dims(normalized, axis=(0,-1))
