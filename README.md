# 🔩 RotorAI

**AI/ML-Based Video Analysis for Motor Inspection**

RotorAI is an intelligent system that uses computer vision and machine learning to **detect rust, cracks, and other surface defects** in electric motors from **video footage**.  
It aims to assist engineers and maintenance teams by automating defect detection and providing **3D visualization** of problem areas.

---

## 🚀 Features

- 🎥 **Video-based defect detection** using YOLO and OpenCV
- 🧠 **Deep learning models** built with TensorFlow, Keras, and PyTorch
- ⚡ **Real-time inference** from live camera feeds or uploaded video
- 🧩 **3D Gaussian Splatting visualization** to highlight detected defects on motor models
- 🌐 **Web-based dashboard** for monitoring results and inspection reports
- ☁️ **Cloud deployment** enabling scalable industrial inspection workflows

---

# 🧠 Project Goals

- Collect and preprocess image/video datasets of motors (normal, rusted, cracked)
- Train and evaluate deep learning models for **defect detection and classification**
- Build a **real-time video processing pipeline** for automated inspection
- Map detected defect regions onto a **3D reconstructed motor surface**
- Develop an **interactive web interface** for engineers to monitor inspections

---

# 🧰 Tech Stack

## AI / Machine Learning
- Python
- TensorFlow / Keras
- PyTorch
- YOLO (Object Detection)
- OpenCV

## 3D Visualization
- 3D Gaussian Splatting

## Backend
- Flask API
- Google Cloud Run

## Frontend
- JavaScript
- Node.js
- Vercel

---

# 🏗 System Architecture

1. **Video Input**
   - Live camera feed or uploaded inspection footage

2. **Frame Processing**
   - OpenCV extracts frames from video streams

3. **Object Detection**
   - YOLO detects potential motor defects (rust, cracks, corrosion)

4. **Model Classification**
   - TensorFlow / PyTorch models classify defect types

5. **3D Visualization**
   - Detected defects are mapped onto a **3D Gaussian Splatting motor model**

6. **Backend API**
   - Flask handles inference requests and serves results

7. **Frontend Dashboard**
   - JavaScript + Node.js interface for visualization and monitoring

8. **Deployment**
   - Backend → Google Cloud Run  
   - Frontend → Vercel
