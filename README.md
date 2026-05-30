# 👁️ Real-Time Gaze Estimation and Iris Tracking System

## 📌 Overview

Real-Time Gaze Estimation and Iris Tracking System is a computer vision and deep learning project that combines MediaPipe Face Mesh and a Convolutional Neural Network (CNN) to detect facial landmarks, track iris movements, and estimate user gaze direction in real time through a webcam.

The system detects both eyes, extracts iris landmark coordinates, verifies eye visibility, and predicts gaze coordinates using a custom-trained neural network based on the MPIIFaceGaze dataset.

---

# 🚀 Features

## 👁️ Real-Time Iris Tracking

* Detects both left and right iris positions.
* Displays live iris coordinates.
* Draws visual markers on both irises.
* Tracks eye movement continuously.

## 🎯 Gaze Estimation

* Uses a CNN model built with PyTorch.
* Predicts gaze coordinates (x, y).
* Supports training on MPIIFaceGaze dataset.
* Generates custom trained gaze models.

## 📹 Webcam Integration

* Real-time video stream processing.
* Continuous frame analysis.
* Low latency detection and tracking.

## 🧠 Face Landmark Detection

* Uses MediaPipe Face Mesh.
* Detects 468+ facial landmarks.
* Extracts iris-specific landmarks.
* Provides accurate eye localization.

## 👀 Eye Visibility Validation

* Detects whether eyes are visible.
* Displays warning when eyes are closed.
* Handles face disappearance from webcam.
* Prevents false gaze estimations.

## ⚡ Real-Time Processing

* Live webcam inference.
* Efficient image preprocessing.
* Optimized for CPU execution.
* Supports future GPU acceleration.

---

# 🏗️ System Architecture

Webcam Input

  ↓

Face Detection

  ↓

Face Mesh Landmark Extraction

  ↓

Iris Landmark Detection

  ↓

Eye Visibility Verification

  ↓

CNN-Based Gaze Estimation

  ↓

Coordinate Prediction

  ↓

Real-Time Visualization

---

# 📂 Project Structure

```text
GAZE-ESTIMATION/
│
├── MPIIFaceGaze_preprocessed/
│   ├── Image/
│   │   ├── p00/
│   │   ├── p01/
│   │   └── ...
│   │
│   └── Label/
│       ├── p00.label
│       ├── p01.label
│       └── ...
│
├── modaltraining.py
├── gaze_model.pth
├── requirements.txt
└── README.md
```

---

# 🛠️ Technologies Used

## Programming Language

* Python 3.11+

## Deep Learning Framework

* PyTorch

## Computer Vision

* OpenCV

## Facial Landmark Detection

* MediaPipe Face Mesh

## Scientific Computing

* NumPy

## Machine Learning Utilities

* Scikit-Learn

## Data Processing

* Pandas

---

# 📚 Dataset

## MPIIFaceGaze Dataset

The project utilizes the MPIIFaceGaze dataset for training and evaluating gaze estimation models.

### Dataset Contents

* Face Images
* Left Eye Images
* Right Eye Images
* Head Pose Information
* Gaze Direction Labels
* Facial Landmark Information

### Advantages

* Real-world data collection
* Multiple users
* Different lighting conditions
* Different head poses
* Large-scale dataset

---

# 🧠 Deep Learning Model

## GazeCNN Architecture

### Input Layer

* 224 × 224 RGB Image

### Convolution Layer 1

* 32 Filters
* Kernel Size: 3 × 3
* ReLU Activation

### Max Pooling Layer

* Pool Size: 2 × 2

### Convolution Layer 2

* 64 Filters
* Kernel Size: 3 × 3
* ReLU Activation

### Max Pooling Layer

* Pool Size: 2 × 2

### Fully Connected Layer

* 128 Neurons

### Output Layer

* 2 Neurons
* Gaze X Coordinate
* Gaze Y Coordinate

---

# 📊 Functionalities

## Face Detection

Detects the user's face from the webcam stream.

## Iris Localization

Identifies precise iris positions using MediaPipe landmarks.

## Coordinate Extraction

Calculates iris center coordinates.

## Eye State Monitoring

Checks whether eyes are open or visible.

## Gaze Prediction

Predicts gaze direction using the trained CNN model.

## Model Training

Allows custom training using MPIIFaceGaze data.

## Model Saving

Exports trained model weights into `.pth` files.

## Real-Time Visualization

Displays live tracking information and markers.

---

# 🔧 Installation

```bash
git clone https://github.com/your-repository/gaze-estimation.git

cd gaze-estimation

pip install -r requirements.txt
```

---

# ▶️ Running the Project

```bash
python modaltraining.py
```

---

# 📦 Requirements

```txt
torch
torchvision
opencv-python
mediapipe
numpy
scikit-learn
pandas
```

---

# 🎯 Applications

* Eye Tracking Systems
* Human Computer Interaction
* Accessibility Solutions
* Assistive Technologies
* Driver Monitoring Systems
* Attention Monitoring
* E-Learning Analytics
* Medical Research
* Gaming Interaction
* Smart User Interfaces

---

# 🔮 Future Enhancements

* Screen Point Calibration
* Cursor Control Using Eye Gaze
* Blink Detection
* Eye Fatigue Monitoring
* Multi-Person Tracking
* GPU Acceleration
* TensorRT Optimization
* Mobile Application Support
* Web-Based Gaze Tracking
* Attention Score Prediction

---

# 📈 Skills Demonstrated

## Artificial Intelligence

* Deep Learning
* Computer Vision
* Neural Networks

## Machine Learning

* Dataset Processing
* Model Training
* Model Evaluation

## Computer Vision

* Image Processing
* Facial Landmark Detection
* Iris Tracking

## Software Development

* Python Programming
* Modular Design
* Real-Time Systems

## Data Engineering

* Data Preprocessing
* Feature Extraction
* Dataset Management

---

# 👨‍💻 Author

Anish Kumar

Computer Science Engineering Student

Focused on Artificial Intelligence, Computer Vision, Full Stack Development, and Cloud Technologies.

---

# ⭐ Project Goal

To build a robust real-time gaze estimation system capable of accurately tracking user eye movement and predicting gaze direction using computer vision and deep learning techniques.
