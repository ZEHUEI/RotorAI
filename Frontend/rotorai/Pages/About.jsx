'use client';
import React from 'react';
import FullBleedDivider from '@/Components/FullBleedDivider';
import AboutCard from '@/Components/AboutCard';
import TechStackCard from '@/Components/TechStackCard';

const Features = [
  {
    id: 1,
    text1: 'Hybrid Image & Video Defect Detection',
    text2:
      'Supports both high-resolution image analysis and real-time video inspection. Static images are processed using a TensorFlow segmentation pipeline, while live video streams leverage YOLOv8 (PyTorch) for low-latency detection.',
  },
  {
    id: 2,
    text1: 'Deep Learning-Based Classification',
    text2:
      'Combines U-Net (ResNet50 encoder) for pixel-level segmentation with YOLOv8 for real-time object detection, enabling robust classification of rust and crack defects.',
  },
  {
    id: 3,
    text1: 'RaySplat-Based 3D Defect Projection',
    text2:
      'Projects 2D detections into 3D space using RaySplat intersection techniques. A ray-sphere approximation is applied to accurately map defects onto curved surfaces, improving spatial alignment over traditional planar projections.',
  },
  {
    id: 4,
    text1: '3D Gaussian Splatting Visualisation',
    text2:
      'Integrates projected defect data with precomputed 3D Gaussian Splatting models, allowing users to inspect defect locations interactively in a spatially coherent 3D environment.',
  },
  {
    id: 5,
    text1: 'Real-Time Inference Pipeline',
    text2:
      'Optimised using GPU acceleration (CUDA), frame skipping, and efficient memory handling to achieve smooth real-time detection (>30 FPS) without UI blocking.',
  },
  {
    id: 6,
    text1: 'Cloud-Native & Scalable Architecture',
    text2:
      'Deployed via Docker and Google Cloud Run using a stateless API design, enabling horizontal scalability, fault tolerance, and continuous deployment.',
  },
];

const Stack = [
  {
    id: 1,
    text1: 'AI & MACHINE LEARNING',
    text2: '• Python 3.x',
    text3: '• TensorFlow (U-Net, ResNet50) – Image segmentation',
    text4: '• PyTorch (YOLOv8) – Real-time object detection',
  },
  {
    id: 2,
    text1: 'COMPUTER VISION & PROCESSING',
    text2: '• OpenCV – Video capture and preprocessing',
    text3: '• CLAHE & Black-Hat Morphology – Image enhancement',
    text4: '• NumPy / Scikit-image – Data processing',
  },
  {
    id: 3,
    text1: '3D RECONSTRUCTION & SPATIAL MAPPING',
    text2: '• 3D Gaussian Splatting (Gsplat)',
    text3: '• RaySplat Intersection (Ray-Sphere Projection)',
    text4: '• Nerfstudio / COLMAP – Scene reconstruction',
  },
  {
    id: 4,
    text1: 'VISUALISATION',
    text2: '• Three.js / WebGL',
    text3: '• @mkkellogg/gaussian-splats-3d',
    text4: '• Interactive 3D viewer (WASD controls)',
  },
  {
    id: 5,
    text1: 'BACKEND & CLOUD',
    text2: '• Flask REST API (Stateless)',
    text3: '• Docker – Containerisation',
    text4: '• Google Cloud Run & Cloud Storage',
  },
  {
    id: 6,
    text1: 'FRONTEND',
    text2: '• React / Next.js',
    text3: '• Tailwind CSS & Framer Motion',
    text4: '• Vercel – CI/CD Deployment',
  },
];

//add link for /About
const About = () => {
  return (
    <div className="mt-10">
      <section>
        <div className="font-ocr text-gray-400/50 text-2xl mb-5">[ ABOUT ]</div>

        <p className="font-Inter text-2xl leading-relaxed max-w-4xl">
          RotorAI is an intelligent inspection system that leverages advanced
          computer vision and machine learning techniques to automatically
          detect surface defects, including rust, cracks, and material
          degradation, across a wide range of structures and components using
          image and video data.
        </p>

        <p className="font-Inter text-gray-300 leading-relaxed max-w-3xl mt-10">
          Designed to support engineers and maintenance teams, RotorAI reduces
          reliance on manual inspection processes by delivering accurate defect
          detection, real-time analysis, and spatially-aware 3D visualisation.
          This enables more efficient condition monitoring, improved maintenance
          planning, and better-informed decision-making in industrial and
          general inspection scenarios.
        </p>
      </section>

      <div className="w-full h-[0.1px] mt-15 mb-15 bg-white/20" />

      <section>
        <div className="font-ocr text-gray-400/50 text-2xl mb-10">
          [ FEATURES ]
        </div>

        {/*mapping */}
        <div className="grid grid-cols-2 gap-10 auto-cols-fr">
          {Features.map((item) => (
            <div key={item.id}>
              <AboutCard text1={item.text1} text2={item.text2} />
            </div>
          ))}
        </div>
      </section>

      <div className="w-full h-[0.1px] mt-15 mb-15 bg-white/20" />
      <section>
        <div className="font-ocr text-gray-400/50 text-2xl mb-5">
          [ TECHNOLOGY STACK ]
        </div>

        {/*mapping */}
        <div className="grid grid-cols-2 gap-10 auto-cols-fr border-0 border-amber-200">
          {Stack.map((item) => (
            <div key={item.id}>
              <TechStackCard
                text1={item.text1}
                text2={item.text2}
                text3={item.text3}
                text4={item.text4}
              />
            </div>
          ))}
        </div>
      </section>

      <FullBleedDivider />
    </div>
  );
};

export default About;
