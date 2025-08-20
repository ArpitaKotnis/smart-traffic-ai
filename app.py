import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile

st.title("🚦 Smart Traffic AI System")
st.write("Upload a traffic video to detect vehicles.")

# Upload video
video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

# Load YOLO model
model = YOLO("yolov5s.pt")

if video_file:
    # Save file temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    tfile_path = tfile.name

    # Correct way to open video with OpenCV
    cap = cv2.VideoCapture(tfile_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO prediction on frame
        results = model(frame)
        
        # You can draw boxes or labels on the frame if needed
        annotated_frame = results[0].plot()  # annotated frame

        # Display frame in Streamlit
        st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))

    cap.release()
