import cv2
import torch
import numpy as np

# Load a pre-trained YOLOv5 model from torch hub
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Small YOLO model

# Open webcam or video (replace 0 with 'traffic.mp4' if you have video file)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame)

    # Draw results
    cv2.imshow('Smart Traffic AI', np.squeeze(results.render()))

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

