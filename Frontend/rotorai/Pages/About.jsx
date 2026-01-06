'use client';
import React from 'react';
import FullBleedDivider from '@/Components/FullBleedDivider';
import { p } from 'framer-motion/client';

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
    text1: 'Backend & AI',
    text2: 'Python 3.x',
    text3: 'TensorFlow / Keras - Deep learning model development',
    text4: 'OpenCV / YOLO - Image and video processing',
  },
  {
    id: 2,
    text1: 'Visualization',
    text2: 'Matplotlib - Data and result visualization',
    text3:
      'Gaussian Splatting - PHD Research ~ grid plotting then 3D modeling with defects',
    text4: '',
  },
  {
    id: 3,
    text1: '',
    text2: '',
    text3: '',
    text4: '',
  },
];

//add link for /About
const About = () => {
  return (
    <div className="mt-10">
      <section>
        <div className="font-ocr text-gray-400/50 text-2xl mb-5">[ ABOUT ]</div>

        <p className="font-Inter text-xl leading-relaxed max-w-3xl">
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
        <div className="font-ocr text-gray-400/50 text-2xl mb-5">
          [ FEATURES ]
        </div>

        {/*mapping */}
        <div className="space-y-10 max-w-3xl">
          {Features.map((item) => (
            <div key={item.id}>
              {item.text1 && (
                <p className="font-Inter text-xl mb-2">{item.text1}</p>
              )}

              <ul className="font-Inter text-gray-300 space-y-1">
                {item.text2 && <li>• {item.text2}</li>}
              </ul>
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
        <div className="space-y-10 max-w-3xl">
          {Stack.map((item) => (
            <div key={item.id}>
              {item.text1 && (
                <p className="font-Inter text-xl mb-2">{item.text1}</p>
              )}

              <ul className="font-Inter text-gray-300 space-y-1">
                {item.text2 && <li>• {item.text2}</li>}
                {item.text3 && <li>• {item.text3}</li>}
                {item.text4 && <li>• {item.text4}</li>}
              </ul>
            </div>
          ))}
        </div>
      </section>

      <FullBleedDivider />
    </div>
  );
};

export default About;
