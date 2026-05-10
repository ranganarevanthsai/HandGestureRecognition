# 🤟 HandGestureRecognition

> Real-time American Sign Language (ASL) recognition using Deep Learning and Computer Vision with Text + Speech output.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?logo=opencv)
![Flask](https://img.shields.io/badge/Flask-Web%20App-000000?logo=flask)
![License](https://img.shields.io/badge/License-MIT-green)

---

# 📌 Table of Contents

- Project Overview
- Features
- Tech Stack
- System Workflow
- MediaPipe Hand Landmarks
- Dataset Information
- CNN Model
- Performance
- Project Structure
- Installation
- Requirements
- Usage
- Screenshots
- Deployment
- Future Improvements
- Contributors
- License
- Contact

---

# 📖 Project Overview

HandGestureRecognition is a real-time ASL recognition system that captures hand gestures from a webcam, extracts hand landmarks using MediaPipe, classifies signs using a CNN model, converts predictions into text, and supports Text-to-Speech (TTS) for voice output.

This project demonstrates:
- Computer Vision
- Deep Learning
- Human-Computer Interaction
- Accessible communication technology

---

# ✨ Features

- Real-time hand gesture detection
- ASL gesture classification using CNN
- MediaPipe hand landmark extraction
- Text output generation
- Text-to-Speech conversion
- Flask-based web interface
- Dataset-driven training workflow

---

# 🧰 Tech Stack

| Category | Technologies |
|---|---|
| Language | Python |
| Deep Learning | TensorFlow, Keras |
| Computer Vision | OpenCV, MediaPipe |
| Numerical Computing | NumPy |
| Web Framework | Flask |
| Speech Engine | pyttsx3 |

---

# 🏗️ System Workflow

1. Webcam captures real-time video
2. MediaPipe detects hand landmarks
3. Image preprocessing is applied
4. CNN model predicts gesture class
5. Output is converted into text
6. Text is converted into speech
7. Flask displays results on web interface

---

# 🖐️ MediaPipe Hand Landmarks

MediaPipe detects 21 hand landmarks for accurate gesture recognition.

Advantages:
- Works in different lighting conditions
- Reduces background dependency
- Improves recognition accuracy
- Provides structured hand feature extraction

---

# 🗂️ Dataset Information

- ASL A-Z gesture dataset
- Multiple gesture variations and angles
- Used for CNN model training and testing
- Includes processed gesture representations

---

# 🧠 CNN Model

The CNN model performs multiclass classification for ASL gestures.

### Model Workflow
- Convolution Layers
- Pooling Layers
- Fully Connected Layers
- Softmax Output Layer

### Model File
```bash
cnn8grps_rad1_model.h5
