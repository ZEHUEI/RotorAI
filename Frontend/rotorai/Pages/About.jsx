'use client';
import React from 'react';
import FullBleedDivider from '@/Components/FullBleedDivider';
import AboutCard from '@/Components/AboutCard';
import TechStackCard from '@/Components/TechStackCard';

const Features = [
  {
    id: 1,
    text1: 'Video-Based Defect Detection',
    text2:
      'Analyzes live camera feeds or uploaded videos using TensorFlow and OpenCV to identify visible defects on motor surfaces.',
  },
  {
    id: 2,
    text1: 'Deep Learning-Powered Analysis',
    text2:
      'Utilizes a trained neural network capable of distinguishing between normal, rusted, and cracked motor components with high accuracy.',
  },
  {
    id: 3,
    text1: '3D Defect Visualization',
    text2:
      'Maps detected defect regions onto a 3D motor model, allowing users to visually inspect affected areas from multiple angles.',
  },
  {
    id: 4,
    text1: 'Real-Time Processing',
    text2:
      'Supports near real-time inference for fast feedback during inspections, enabling immediate action in operational environments.',
  },
  {
    id: 5,
    text1: 'Scalable & Industry-Ready',
    text2:
      'Built with modular architecture to allow seamless integration into industrial maintenance systems and future expansion.',
  },
];
const Stack = [
  {
    id: 1,
    text1: 'AI & MACHINE LEARNING',
    text2: '• Python 3.x',
    text3: '• Tensorflwow, Keras- (CLAHE and Balckhat Morphology)',
    text4: '• OpenCV / YOLO - Image and video processing',
  },
  {
    id: 2,
    text1: '3D VISUALIZATION',
    text2: '• 3D Gaussian Splatting',
  },
  {
    id: 3,
    text1: 'BACKEND & CLOUD DEPLOYMENT',
    text2: '• FLASK API',
    text3: '• Docker',
    text4: '• GCR - (Google Cloud Run)',
  },
  {
    id: 4,
    text1: 'FRONTEND DEPLOYMENT',
    text2: '• JavaScript',
    text3: '• NODE.js',
    text4: '• Vercel',
  },
];

//add link for /About
const About = () => {
  return (
    <div className="mt-10">
      <section>
        <div className="font-ocr text-gray-400/50 text-2xl mb-5">[ ABOUT ]</div>

        <p className="font-Inter text-2xl leading-relaxed max-w-4xl">
          RotorAI is an intelligent inspection system that leverages computer
          vision and machine learning to automatically detect surface defects-
          such as rust, cracks, and structural wear-in electric motors using
          video footage.
        </p>

        <p className="font-Inter text-gray-300 leading-relaxed max-w-3xl mt-10">
          Designed to support engineers and maintenance teams, RotorAI reduces
          reliance on manual inspections by delivering accurate defect
          detection, real-time analysis, and 3D visual insights into problem
          areas, improving maintenance efficiency and decision-making.
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
