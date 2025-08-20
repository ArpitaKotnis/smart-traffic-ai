import cv2
import numpy as np

# Load a traffic video (you can download a sample or use webcam)
cap = cv2.VideoCapture("traffic.mp4")  # replace with 0 for webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Simple thresholding (for car detection placeholder)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    cv2.imshow("Traffic Frame", frame)
    cv2.imshow("Threshold", thresh)

    if cv2.waitKey(30) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
