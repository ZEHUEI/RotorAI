<div align="center">

<img src="https://img.shields.io/badge/-%F0%9F%94%A9%20RotorAI-000000?style=for-the-badge" alt="RotorAI" />

### AI-Powered Motor Defect Detection & 3D Inspection Platform

Detect **rust**, **cracks**, **corrosion**, and surface defects from video footage using Computer Vision and Deep Learning.

<br/>

![Python](https://img.shields.io/badge/Python_3.11+-3776AB?style=flat-square&logo=python&logoColor=white)
![YOLO](https://img.shields.io/badge/YOLO-Object_Detection-E31E24?style=flat-square)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-3D8B37?style=flat-square&logo=opencv&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep_Learning-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Neural_Networks-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Google Cloud](https://img.shields.io/badge/Google_Cloud_Run-4285F4?style=flat-square&logo=googlecloud&logoColor=white)

<br/>

**Video Analysis · Deep Learning · Real-Time Detection · 3D Visualization**

</div>

---

## Overview

RotorAI is an intelligent motor inspection platform that automates the detection of surface defects using Computer Vision, Deep Learning, and 3D Reconstruction.

Instead of manually reviewing inspection footage, engineers can upload a video or stream a live camera feed and receive automated defect detection results in real time. Detected defects are mapped onto a reconstructed **3D motor model** for intuitive inspection and maintenance planning.

---

## Detected Defects

| Defect | Description |
|---|---|
| 🟠 Rust | Surface oxidation |
| 🔴 Cracks | Structural damage |
| 🟣 Corrosion | Material degradation |
| 🔵 Surface Wear | Deterioration over time |
| 🟢 Structural Damage | Physical deformation |

---

## AI Pipeline

```
Video Input  ──►  Frame Extraction  ──►  YOLO Detection  ──►  Classification  ──►  3D Splatting  ──►  Dashboard
 Camera/Upload       OpenCV                YOLOv11             TF / PyTorch       Gaussian            Reports
```

---

## Features

<table>
<tr>
<td width="50%">

### 🎥 Video Inspection
Analyze uploaded footage, live camera feeds, and maintenance recordings automatically.

</td>
<td width="50%">

### ⚡ Real-Time Inference
Live bounding-box overlay, confidence scoring, and instant detection feedback.

</td>
</tr>
<tr>
<td width="50%">

### 🧩 3D Visualization
Gaussian Splatting projects detected defects directly onto a photorealistic reconstructed motor model.

</td>
<td width="50%">

### 📊 Inspection Reports
Detection history, defect statistics, and annotated 3D models in one interactive dashboard.

</td>
</tr>
</table>

---

## Tech Stack

**AI / Machine Learning**

| Technology | Purpose |
|---|---|
| Python 3.10 | Core language |
| YOLOv11 | Object detection |
| OpenCV | Video processing |
| TensorFlow | Classification |
| PyTorch | Deep learning |

**3D Visualization**

| Technology | Purpose |
|---|---|
| Gaussian Splatting | 3D reconstruction & defect projection |

**Backend & Frontend**

| Technology | Purpose |
|---|---|
| Flask | REST API |
| Google Cloud Run | Cloud deployment |
| JavaScript / Node.js | Frontend logic & runtime |
| Vercel | Frontend hosting |

---

## Project Goals

- [x] Build motor inspection dataset
- [x] Train YOLO defect detector
- [x] Implement classification pipeline
- [x] Develop real-time inference system
- [x] Integrate Gaussian Splatting
- [x] Deploy cloud infrastructure
- [x] Create interactive dashboard
- [x] Generate inspection reports

---

## Future Improvements

**📈 Predictive Maintenance** — Estimate component lifespan before failure occurs.

**🌡️ Thermal Inspection** — Combine thermal imaging with visual analysis for deeper diagnostics.

**📱 Mobile Application** — Perform inspections directly from mobile devices in the field.

**⚙️ Edge Deployment** — Run RotorAI on industrial edge devices without cloud dependency.

**📝 AI Report Generation** — Automatically summarise findings and recommend maintenance actions.

**🔗 Digital Twin Integration** — Connect live inspections to real-time digital twin systems.

---

<div align="center">

**🔩 RotorAI — Automating Industrial Motor Inspection with AI**

⭐ Star this repository if you find it useful. ❤️

</div>
