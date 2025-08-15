# import cv2
# from ultralytics import YOLO

# model = YOLO("C:/Users/Anisha.S/Documents/helmet-detection-yolov8/venv/dataset/runs/detect/train2/weights/best.pt")

# cap = cv2.VideoCapture(0)  # or "video.mp4"

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     results = model(frame)
#     annotated_frame = results[0].plot()

#     cv2.imshow("Helmet Detection", annotated_frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os

# Title
st.title("🪖 Helmet Detection using YOLOv8")
st.write("Upload an image or use your webcam to detect helmets.")

# Load YOLO model
model_path = "C:/Users/Anisha.S/Documents/helmet-detection-yolov8/venv/dataset/runs/detect/train2/weights/best.pt"  # Change path if needed
model = YOLO(model_path)

# Option 1: Upload image
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save to temp file
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())
    temp_path = temp_file.name

    # Read and run detection
    img = cv2.imread(temp_path)
    results = model(img)

    # Annotate image
    annotated_img = results[0].plot()

    # Convert BGR to RGB for Streamlit
    annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

    st.image(annotated_img, caption="Detection Result", use_column_width=True)
    os.unlink(temp_path)

# Option 2: Webcam mode
if st.button("Start Webcam"):
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated_frame = results[0].plot()
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        stframe.image(annotated_frame, channels="RGB")

    cap.release()
