<div align="center">

# 🔩 RotorAI

### AI-Assisted Surface Defect Inspection & 3D Gaussian Splatting

**Computer Vision · Deep Learning · Ray-Based 3D Mapping · Interactive Visualization**

[![Website](https://img.shields.io/badge/Live%20Website-RotorAI-111827?style=flat-square)](https://rotor-ai.vercel.app/)
[![Models](https://img.shields.io/badge/Models-Hugging%20Face-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/MeNoobHeHe/RotorAI)
![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-16-000000?style=flat-square&logo=nextdotjs&logoColor=white)
![3DGS](https://img.shields.io/badge/3D-Gaussian%20Splatting-7C3AED?style=flat-square)

[**Explore the Website**](https://rotor-ai.vercel.app/) · [**3D Viewer**](https://rotor-ai.vercel.app/gaussian) · [**Development Blog**](https://rotor-ai.vercel.app/blog) · [**Model Repository**](https://huggingface.co/MeNoobHeHe/RotorAI)

</div>

---

## Overview

**RotorAI** is a research-oriented motor inspection project exploring how **2D defect detection** can be combined with **3D Gaussian Splatting (3DGS)** to make surface damage easier to locate and inspect spatially.

It brings together two complementary workflows:

1. **2D inspection:** A TensorFlow U-Net segmentation pipeline analyzes still images for rust and searches for crack-like features **within detected rust regions**. A separate YOLOv8/OpenCV workflow processes video frames for low-latency visual inspection.
2. **3D inspection:** Offline reconstruction and ray-based defect mapping produce a Gaussian-splat motor model and associated annotations. The website loads these **precomputed assets** into an interactive browser-based 3D viewer.

The project is also an ongoing investigation into the geometry of Gaussian representations, including **Mahalanobis-distance masking** and **ray–ellipsoid intersections**. These are **research directions in progress**, not features currently claimed to be implemented in the deployed viewer.

> **Deployment status:** The former Google Cloud Run inference service has been shut down. The website, project write-ups, and precomputed 3D demonstration remain available; image/video inference on the public site is not currently operational.

## What RotorAI Does

| Component | Approach | Status |
| --- | --- | --- |
| Image segmentation | TensorFlow U-Net with a ResNet50 encoder, image enhancement, and post-processing | Implemented; backend currently offline |
| Rust-constrained crack analysis | Black-hat morphology and connected-component filtering inside rust regions | Implemented; deliberately scope-limited |
| Video inspection | YOLOv8 with OpenCV and frame-skipping optimization | Implemented; backend currently offline |
| 3D reconstruction | Offline 3D Gaussian Splatting workflow | Implemented as precomputed assets |
| 2D-to-3D defect mapping | Ray-based projection using a ray–sphere approximation | Demonstrated with precomputed annotations |
| Interactive 3D inspection | Three.js and Gaussian splat rendering | Available on the website |
| Covariance-aware mapping | Mahalanobis distance, anisotropic ellipsoid support, smooth Gaussian masks | **Under investigation** |

## System Architecture

The live 2D inference workflow and the offline 3D workflow are **separate by design**. Uploading a new image does **not** automatically regenerate a 3D scene.

```text
                    2D INSPECTION (INFERENCE BACKEND)

         Still image                              Video frames
             |                                        |
             v                                        v
       CLAHE + resize                            OpenCV processing
             |                                        |
             v                                        v
      TensorFlow U-Net                                YOLOv8
             |                                        |
             v                                        v
       Rust segmentation                         Region detections
             |                                        |
             v                                        v
  Rust / crack post-processing                   Visual annotations
             |                                        |
             +------------------+---------------------+
                                |
                                v
                    2D detection outputs


                    3D INSPECTION (OFFLINE + WEB)

           Multiview reconstruction inputs
                          |
                          v
                 3D Gaussian Splatting
                          |
                          v
             Ray-based defect projection
                          |
                          v
              Precomputed PLY + defect JSON
                          |
                          v
             Three.js interactive 3D viewer
```

### Image Analysis

- **Segmentation:** U-Net with a ResNet50 encoder estimates a corrosion/rust candidate mask.
- **Preprocessing:** Aspect-ratio-preserving resizing and CLAHE improve consistency before inference.
- **Rust refinement:** Color-space filtering, morphological processing, and connected-component rules refine the segmentation.
- **Crack analysis:** Black-hat morphology identifies dark, elongated candidates **only where the rust mask permits**. This stage is **not** intended to detect cracks on otherwise clean material.
- **Output:** Masks, overlays, and detected-pixel counts for inspection.

### Video Analysis

- YOLOv8 localizes candidate regions within incoming frames.
- OpenCV and ROI post-processing refine rust/crack annotations.
- Frame skipping trades detection frequency for processing speed during continuous inspection.
- The original GPU-accelerated setup achieved **30+ FPS in development testing**; that performance is **not a benchmark of the currently published CPU-configured backend**. Measured throughput depends on hardware, resolution, and inference settings.

### 3D Defect Visualization

- A motor is represented as a collection of 3D Gaussians, rather than only as a conventional triangle mesh.
- Ray-based mapping associates selected 2D detections with positions in a reconstructed 3D scene.
- The website loads a **precomputed** Gaussian-splat `.ply` model and defect annotation `.json` data.
- Users can explore the scene spatially using the browser viewer.

The current 3D presentation is a **demonstration of mapping and visualization**, not a real-time cloud reconstruction service.

## Research: Mahalanobis Distance & Ellipsoid Rendering

**Status: ongoing exploration — not yet part of the deployed RotorAI pipeline.**

The existing ray–sphere approximation is useful for an initial spatial mapping pipeline, but Gaussian splats are generally **anisotropic**: their shape and orientation are described by a covariance matrix, not a single radius.

The next research direction is to use that covariance information when reasoning about where a viewing ray passes through a Gaussian and how strongly a point should contribute to a defect mask.

### 1. Mahalanobis Distance

For a Gaussian with center $\boldsymbol{\mu}$ and positive-definite covariance $\boldsymbol{\Sigma}$, the squared Mahalanobis distance of a point $\mathbf{x}$ is

$$
d_M^2(\mathbf{x}) = (\mathbf{x}-\boldsymbol{\mu})^T
\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu}).
$$

Unlike Euclidean distance, this measures distance relative to a Gaussian's **scale and orientation**. A possible smooth weighting function is

$$
w(\mathbf{x}) = \exp\left(-\frac{1}{2}d_M^2(\mathbf{x})\right).
$$

This is a candidate building block for a **continuous, covariance-aware mask** instead of a hard binary boundary. Whether it improves projection stability and training behavior must be established experimentally.

### 2. Ray–Ellipsoid Intersection

A covariance-aligned ellipsoid at Mahalanobis radius $k$ satisfies

$$
(\mathbf{x}-\boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}
(\mathbf{x}-\boldsymbol{\mu}) = k^2.
$$

Substituting a ray $\mathbf{r}(t)=\mathbf{o}+t\mathbf{d}$ gives a quadratic in $t$:

$$
At^2 + Bt + C = 0,
$$

where $\mathbf{q}=\mathbf{o}-\boldsymbol{\mu}$ and

$$
\begin{aligned}
A &= \mathbf{d}^T\boldsymbol{\Sigma}^{-1}\mathbf{d},\\
B &= 2\mathbf{d}^T\boldsymbol{\Sigma}^{-1}\mathbf{q},\\
C &= \mathbf{q}^T\boldsymbol{\Sigma}^{-1}\mathbf{q}-k^2.
\end{aligned}
$$

Real roots with $t\ge 0$ identify candidate forward intersections with the chosen ellipsoidal level set. Because a Gaussian has no literal finite surface, **$k$ is a selected threshold**, not an intrinsic physical boundary.

### 3. Proposed Evaluation

The proposed comparison is between **ray–sphere approximation**, **covariance-aware ray–ellipsoid intersection**, and **smooth Mahalanobis-weighted projection**. Evaluation questions include:

- Does orientation-aware mapping improve alignment for elongated or rotated Gaussians?
- How sensitive are predictions to camera pose, covariance scale, and mask thresholds?
- How do smooth weights behave near the chosen ellipsoid boundary?
- What is the runtime cost relative to the simpler ray–sphere method?

A ray–ellipsoid hit test alone does **not** guarantee differentiability or stable gradients. Smooth weighting and any optimization behavior require separate testing.

## Testing & Validation

The project includes experimental work involving **Monte Carlo methods**. Randomized trials are useful for assessing sensitivity and robustness, but the exact claim depends on which quantities were sampled and measured.

For transparent, repeatable reporting, future published experiment summaries should include:

| Item | What to report |
| --- | --- |
| Experiment | Component or hypothesis being tested |
| Sampling | Variables, ranges, probability distributions, and random seed |
| Trial count | Number of Monte Carlo iterations |
| Metrics | Error, robustness, runtime, or other relevant measurements |
| Baseline | Method used for comparison |
| Results | Summary statistics and plots, including failure cases |

**No quantitative Monte Carlo results are claimed in this README until the underlying experiment settings and outputs are documented.** Numerical research validation and software regression tests serve different purposes; both remain useful as the project evolves.

## Why Precompute the 3D Pipeline?

On-demand 3D reconstruction can be expensive and slow. A continuously available workflow with background queues, dedicated compute workers, GPU resources, and storage would introduce recurring infrastructure costs that are not justified for this research showcase.

RotorAI therefore separates **offline reconstruction and annotation** from **interactive frontend visualization**. This retains a meaningful 3D demonstration without claiming that each visitor can train a new 3DGS scene in real time.

## Technology Stack

| Layer | Tools |
| --- | --- |
| Deep learning | Python, TensorFlow, U-Net/ResNet50, PyTorch, YOLOv8 |
| Image processing | OpenCV, NumPy, CLAHE, morphological operations |
| Reconstruction & mapping | 3D Gaussian Splatting, Nerfstudio/COLMAP workflows, ray-based projection |
| Browser 3D | Three.js, `@mkkellogg/gaussian-splats-3d` |
| Backend | Flask, REST API, Docker |
| Frontend | Next.js, React, Tailwind CSS, Framer Motion |
| Hosting | Vercel; previously Google Cloud Run and Cloud Storage |

## Repository Layout

```text
RotorAI/
├── Frontend/rotorai/       # Next.js frontend and interactive 3D viewer
├── backend/                # Flask API and backend dependencies
│   ├── AIController.py
│   └── requirements.txt
├── Phase1/                 # Image segmentation and post-processing
│   └── PostProcess.py
├── Phase2/                 # YOLO video detection logic
│   └── YOLOapi.py
├── Phase3/                 # 3D research and development files
├── Phase4/                 # Further 3D integration/development files
├── Outcomes/               # Project outputs
├── Dockerfile              # Backend container definition
└── README.md
```

## Getting Started

### Frontend (local development)

Requires a recent Node.js release compatible with Next.js 16 (Node.js **20.9+**).

```bash
git clone https://github.com/ZEHUEI/RotorAI.git
cd RotorAI/Frontend/rotorai
npm install
npm run dev
```

Visit **http://localhost:3000**. Precomputed 3D assets may be viewed independently of the inference backend, provided the assets are present in the project's public directory.

### Inference backend (requires model access)

The Flask backend is **not a one-command offline demo**: its current startup path retrieves model weights from Google Cloud Storage and requires valid access to those assets.

```bash
# From the repository root, in a Python 3.10 virtual environment
pip install -r backend/requirements.txt
```

Configure environment variables for your development environment:

```dotenv
API_SECRET=<your-secret-value>
GOOGLE_APPLICATION_CREDENTIALS=<path-to-your-gcp-credentials>
```

The Next.js proxy uses the corresponding server-side API secret and backend address:

```dotenv
# Frontend/rotorai/.env.local
API_SECRET=<same-secret-value>
NEXT_PUBLIC_BACKEND_URL=http://localhost:8080
```

**Never commit real secrets or service-account credentials.** The `API_SECRET` value must match on the proxy and backend. `NEXT_PUBLIC_BACKEND_URL` contains only the service URL, not the secret.

Once the Python environment and cloud model access are configured, the Flask application can be started from the repository root with:

```bash
python -m backend.AIController
```

> **Reproducibility note:** Local environment compatibility and access to trained weights are prerequisites. The historical cloud bucket may not be available to external contributors. The [Hugging Face model page](https://huggingface.co/MeNoobHeHe/RotorAI) is the public model reference; download or startup adaptations may be needed for local inference.

## Current Limitations

- Public image/video inference is unavailable because the former GCP backend has been discontinued.
- The public 3D demo uses precomputed assets; it does not reconstruct arbitrary new uploads.
- Crack detection in the image post-processing stage is limited to detected rust regions.
- Performance varies with deployment hardware and inference settings.
- The Mahalanobis and ellipsoid-based methods are active research topics, **not validated production features**.
- Model access, environment dependencies, and evaluation datasets may require additional setup before independent reproduction.

## Roadmap

- [x] Develop an image-based rust segmentation pipeline
- [x] Add rust-constrained crack analysis
- [x] Integrate YOLOv8 video inspection
- [x] Build a browser-based 3D Gaussian Splatting viewer
- [x] Demonstrate ray-based 2D-to-3D defect mapping with precomputed annotations
- [x] Deploy an inference backend during development (subsequently discontinued)
- [ ] Prototype covariance-aware Mahalanobis masking
- [ ] Compare ray–sphere and ray–ellipsoid intersection methods
- [ ] Benchmark projection error, robustness, and runtime under documented test conditions
- [ ] Publish reproducible experimental configurations and results
- [ ] Improve multi-stream inference isolation and automated regression coverage

## Project Links

- **Website:** https://rotor-ai.vercel.app/
- **Interactive 3D viewer:** https://rotor-ai.vercel.app/gaussian
- **Development timeline:** https://rotor-ai.vercel.app/blog
- **Model page:** https://huggingface.co/MeNoobHeHe/RotorAI

---

<div align="center">

**RotorAI — exploring the connection between defect detection, 3D geometry, and Gaussian-based rendering.**

</div>
