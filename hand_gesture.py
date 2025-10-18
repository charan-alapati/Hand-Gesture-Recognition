import cv2
import numpy as np
from tensorflow.keras.models import load_model
from utils import preprocess_frame

MODEL_PATH = "hand_gesture_model.h5"
IMG_SIZE = 64

model = load_model(MODEL_PATH)
classes = ["fist", "palm", "thumbs_up"]  # Update based on your folder labels

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    roi = cv2.flip(frame, 1)
    x, y, w, h = 100, 100, 300, 300
    cv2.rectangle(roi, (x, y), (x+w, y+h), (0,255,0), 2)
    hand = roi[y:y+h, x:x+w]
    pred = preprocess_frame(hand, IMG_SIZE)
    pred_class = classes[np.argmax(model.predict(pred))]
    cv2.putText(roi, pred_class, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.imshow("Hand Gesture Recognition", roi)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
