import cv2
from cvzone.HandTrackingModule import HandDetector
from keras.models import load_model

import numpy as np
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
model = load_model("Model/keras_model.h5")


offset = 20
imgSize = 300
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N"]

# ✅ Create resizable OpenCV window before the loop
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read from webcam")
        break

    imgOutput = img.copy()

    # ✋ Detect hands from the raw webcam feed
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Clamp bounding box within image boundaries
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

        # Resize imgWhite to 224x224 (Keras expects this for MobileNet)
        imgWhite_resized = cv2.resize(imgWhite, (224, 224))
        imgWhite_normalized = imgWhite_resized.astype(np.float32) / 255.0
        imgWhite_input = np.expand_dims(imgWhite_normalized, axis=0)

        # Get confidence scores
        prediction = model.predict(imgWhite_input)[0]
        index = np.argmax(prediction)
        confidence = prediction[index]

        cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                      (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
        label = f"{labels[index]} ({confidence * 100:.2f}%)"

        # Measure the size of the label text
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 1.2, 2)

        # Draw background rectangle based on text size
        cv2.rectangle(imgOutput,
                      (x - offset, y - offset - 50),
                      (x - offset + text_width + 20, y - offset - 50 + text_height + 20),
                      (255, 0, 255),
                      cv2.FILLED)

        # Draw the label text inside the purple rectangle
        cv2.putText(imgOutput, label,
                    (x - offset + 10, y - offset - 50 + text_height + 5),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1.2,
                    (255, 255, 255),
                    2)

        cv2.rectangle(imgOutput, (x - offset, y - offset),
                      (x + w + offset, y + h + offset), (255, 0, 255), 4)

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    # ✅ Resize display output at the very end
    desired_width = 1280
    desired_height = 720
    resizedOutput = cv2.resize(imgOutput, (desired_width, desired_height))

    cv2.imshow("Image", resizedOutput)
    cv2.waitKey(1)
