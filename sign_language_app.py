import streamlit as st
import cv2
import numpy as np
import math
from keras.models import load_model
from cvzone.HandTrackingModule import HandDetector

# Page config
st.set_page_config(page_title="Sign Language Detection", layout="wide")

# Load model and labels
model = load_model("Model/keras_model.h5")
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N"]

# Setup
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300

# Custom fonts via HTML (minimal improvement within Streamlit constraints)
font_css = """
<style>
body {
    font-family: 'Segoe UI', sans-serif;
}
h1, h3, p {
    font-family: 'Segoe UI', sans-serif;
}
</style>
"""
st.markdown(font_css, unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("## ⚙️ Options")
show_confidence = st.sidebar.checkbox("Show Confidence %", value=True)
show_accuracy = st.sidebar.checkbox("Show Accuracy %", value=True)

st.sidebar.markdown("## 🚀 ACTIONS")
start_detection = st.sidebar.button("▶️ Start Detection")
stop_detection = st.sidebar.button("⛔ Stop Detection")

# Title & Description
st.markdown("""
<h1 style='color:white;'>🤟 Real-Time Sign Language Detection</h1>
<p style='color:white; font-size:16px;'>
This real-time web app detects American Sign Language (A–N) using your webcam. <br>
It leverages AI with OpenCV, TensorFlow, and CVZone to recognize hand gestures.<br>
Click the <b>Start Detection</b> button to begin. You'll see the live camera feed and detection results below.<br>
You can also toggle confidence and accuracy metrics from the sidebar.
</p>
<p style='color:gray; font-size:12px;'>Note: Your camera will activate once you click the Start Detection button.</p>
""", unsafe_allow_html=True)

# Output placeholders
col1, col2 = st.columns(2)
with col1:
    output_box = st.empty()
    confidence_box = st.empty()
    accuracy_box = st.empty()
with col2:
    frame_display = st.image([])

# App state values
if 'predicted_label' not in st.session_state:
    st.session_state['predicted_label'] = "-"
if 'confidence' not in st.session_state:
    st.session_state['confidence'] = "-"
if 'accuracy' not in st.session_state:
    st.session_state['accuracy'] = "-"

# Update display boxes
output_box.markdown(f"### Detected Output:\n**{st.session_state['predicted_label']}**")
confidence_box.markdown(f"### Confidence Detected:\n**{st.session_state['confidence']}%**")
accuracy_box.markdown(f"### Accuracy Detected:\n**{st.session_state['accuracy']}%**")

# Detection loop
if start_detection and not stop_detection:
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Stream", cv2.WINDOW_NORMAL)

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            st.warning("Failed to read from webcam.")
            break

        imgOutput = img.copy()
        hands, img = detector.findHands(img)
        prediction_text = "-"
        confidence = "-"
        accuracy = "-"

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

            # Resize to 224x224 for MobileNet
            imgWhite_resized = cv2.resize(imgWhite, (224, 224))
            imgWhite_normalized = imgWhite_resized.astype(np.float32) / 255.0
            imgWhite_input = np.expand_dims(imgWhite_normalized, axis=0)

            prediction = model.predict(imgWhite_input)[0]
            index = np.argmax(prediction)
            confidence = float(prediction[index] * 100)
            prediction_text = labels[index]
            accuracy = confidence  # Same as confidence here unless custom accuracy logic is added

            # Draw bounding box only (no text label above hand)
            cv2.rectangle(imgOutput,
                          (x - offset, y - offset),
                          (x + w + offset, y + h + offset),
                          (255, 0, 255), 4)

        # Update session state and UI
        st.session_state['predicted_label'] = prediction_text
        st.session_state['confidence'] = f"{confidence:.2f}" if show_confidence and prediction_text != "-" else "-"
        st.session_state['accuracy'] = f"{accuracy:.2f}" if show_accuracy and prediction_text != "-" else "-"

        output_box.markdown(f"### Detected Output:\n**{st.session_state['predicted_label']}**")
        confidence_box.markdown(f"### Confidence Detected:\n**{st.session_state['confidence']}%**")
        accuracy_box.markdown(f"### Accuracy Detected:\n**{st.session_state['accuracy']}%**")

        frame_display.image(imgOutput, channels="BGR")

        if stop_detection:
            break

    cap.release()
