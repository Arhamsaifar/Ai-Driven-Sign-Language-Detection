import streamlit as st
import cv2
from keras.models import load_model
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector

st.set_page_config(page_title="Sign Language Detection", layout="wide")

# Title and Description
st.markdown("<h1 style='text-align:center; color:white;'>🤟 Real-Time Sign Language Detection</h1>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; color:white; font-size:18px;'>
This web app uses your webcam and an AI model to detect hand gestures corresponding to American Sign Language (A–N).<br>
It utilizes OpenCV, TensorFlow, and CVZone to classify live hand signs.<br>
Start the detection to see real-time predictions of your hand gestures.
</div>
""", unsafe_allow_html=True)

# Sidebar: Settings
st.sidebar.markdown("## ⚙️ Settings")
show_confidence = st.sidebar.checkbox("Show Confidence", True)
show_accuracy = st.sidebar.checkbox("Show Accuracy", True)

# Sidebar: Actions
st.sidebar.markdown("## 🧭 ACTIONS")
start_detection = st.sidebar.button("▶️ Start Detection")
stop_detection = st.sidebar.button("⏹️ Stop Detection")

# Initialize model and variables
model = load_model("Model/keras_model.h5")
labels = open("Model/labels.txt").read().strip().split("\n")
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300

# Real-time display elements
frame_display = st.empty()
output_display = st.empty()
confidence_display = st.empty()
accuracy_display = st.empty()

cap = None

# Store session detection toggle
if "detecting" not in st.session_state:
    st.session_state.detecting = False

if start_detection:
    st.session_state.detecting = True
    cap = cv2.VideoCapture(0)

if stop_detection:
    st.session_state.detecting = False
    if cap:
        cap.release()
        cap = None

# Begin detection loop
if st.session_state.detecting:
    cap = cv2.VideoCapture(0)
    while st.session_state.detecting:
        success, img = cap.read()
        if not success:
            st.warning("Webcam not accessible.")
            break

        hands, img = detector.findHands(img)
        prediction_letter = "-"
        confidence = 0.0

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            height, width, _ = img.shape

            x1 = max(0, x - offset)
            y1 = max(0, y - offset)
            x2 = min(x + w + offset, width)
            y2 = min(y + h + offset, height)
            imgCrop = img[y1:y2, x1:x2]
            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            imgWhite_resized = cv2.resize(imgWhite, (224, 224))
            imgWhite_normalized = imgWhite_resized.astype(np.float32) / 255.0
            img_input = np.expand_dims(imgWhite_normalized, axis=0)

            prediction = model.predict(img_input)[0]
            index = np.argmax(prediction)
            confidence = prediction[index]
            prediction_letter = labels[index]

            # Draw only bounding box (no label text)
            cv2.rectangle(img, (x - offset, y - offset),
                          (x + w + offset, y + h + offset), (255, 0, 255), 4)

        frame_display.image(img, channels="BGR")
        output_display.markdown(f"<h3 style='color:#00bfff;'>Detected Output:</h3><h2 style='color:white'>{prediction_letter}</h2>", unsafe_allow_html=True)
        if show_confidence:
            confidence_display.markdown(f"<h4 style='color:#00cc99;'>Confidence Detected:</h4><p style='color:white'>{confidence * 100:.2f}%</p>", unsafe_allow_html=True)
        if show_accuracy:
            accuracy_display.markdown(f"<h4 style='color:#ffaa00;'>Accuracy Detected:</h4><p style='color:white'>{confidence * 100:.2f}%</p>", unsafe_allow_html=True)

    if cap:
        cap.release()