# 🤖 **AI-Driven Real-Time Sign Language Detection**

## 📌 Overview
This project aims to bridge communication gaps by recognizing American Sign Language (ASL) letters and numbers in **real-time** using **computer vision** and **machine learning** techniques. The system leverages **MediaPipe** for hand tracking, **OpenCV** for video input, and a **RandomForest classifier** trained on hand landmark data.

It’s designed with accessibility in mind—especially for people of determination—and is a step toward creating inclusive, intelligent interfaces.

---

## 🛠 Features

- Real-time ASL letter and number recognition via webcam
- Hand tracking and landmark extraction using **MediaPipe**
- Classification using **RandomForest** trained on landmark features
- Display of the predicted letter/number on the screen
- Structured for future improvements (e.g., CNN models, UI layer)

---

## 🖥️ System Requirements

- Python 3.9 
- OpenCV
- MediaPipe
- NumPy
- Scikit-learn (for RandomForest Classifier)
- Webcam

---

## 🧱 Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/sign-language-detector.git
   cd sign-language-detector
   
2. **Install Required Packages**
    ```bash
   pip install opencv-python mediapipe numpy scikit-learn
   
## 🧠 Dataset
- Hand landmark coordinates were collected using MediaPipe.


- The dataset includes 26 ASL alphabets (A-Z) and digits (0-9).


- Features were extracted from 21 hand landmarks (x, y, z coordinates).


- Data was stored in a CSV format and labeled accordingly.


- Preprocessing included normalization and shuffling to prevent bias.

## 🏗️ Model Architecture
- Classifier Used: RandomForest from **sklearn.ensemble**


- Input: 21 hand landmarks → 63 features (x, y, z for each point)


- Training: Trained on labeled CSV data for alphabets and numbers


- Testing: Evaluated with accuracy metrics and confusion matrix

## 🎮 How It Works
- Captures live feed from webcam using OpenCV.


- Detects and tracks hand using MediaPipe.


- Extracts hand landmark features (x, y, z).


- Loads pre-trained RandomForest model.


- Classifies gesture into a corresponding letter or number.

 
- Displays result in real-time on screen.

## 🔬 Accuracy and Challenges
- Achieved solid accuracy (will update exact % after full testing).


- Low latency for prediction.


- Struggles with similar signs like ‘J’ and ‘Z’ due to limited training samples.


- Future plans include CNN-based enhancement and dataset expansion.

## 📈 Future Enhancements
- Replace RandomForest with Convolutional Neural Network for better visual recognition


- Include dynamic gestures (words, expressions)


- Add voice output for real-time speech generation


- Develop a mobile version of the tool


- Add user-friendly GUI

## 🗂️ Project Structure
├── keras_model.h5 # Trained Keras model
├── labels.txt # Label mapping (0 = A, 1 = B, etc.)
├── test.py # Main real-time prediction script
├── dataCollection.py # Tool to collect hand gesture images
├── Data/ # Folders containing gesture images (A, B, C...)
└── README.md # Project documentation

## 💡 How to Use

1. **🧪 Test the Trained Model (Real-Time ASL Detection) A pre-trained model (`keras_model.h5`) and label map (`labels.txt`) are already provided. To run the real-time detection system:**
   ```bash
   python test.py
   
This will:

Launch your webcam

Detect your hand

Predict the ASL letter (A–N as of now)

Display the prediction live on-screen

2. **📦Collect Your Own Dataset:
If you'd like to collect more hand gesture data:**
   ```bash
   python dataCollection.py
   
How it works:

It activates your webcam.

Shows a cropped and resized hand image.

When you press the S key, the image is saved in the appropriate folder (e.g., Data/E).

You can change the folder name in the code to collect different letters/numbers.

## 👥 Contributing
If you'd like to contribute or suggest improvements:

1. Fork the repo

2. Create a new branch

3. Push your changes

4. Submit a Pull Request

## 📃 License
This project is open-source and available under the MIT License.

## 📬 Contact
For any questions or suggestions, feel free to reach out or open an issue in the repository.