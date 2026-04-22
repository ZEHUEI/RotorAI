'use client';
import React, { useEffect, useRef } from 'react';
import * as GaussianSplats3D from '@mkkellogg/gaussian-splats-3d';

const Gaussian = () => {
  const containerRef = useRef(null);

  useEffect(() => {
    if (!containerRef.current) return;

    const viewer = new GaussianSplats3D.Viewer({
      cameraUp: [0, -1, 0],
      initialCameraPosition: [0, 1, 3],
      initialCameraLookAt: [0, 0, 0],
      sphericalHarmonicsDegree: 0, // Lowering this saves your laptop's GPU
    });

    viewer.init();
    containerRef.current.appendChild(viewer.renderer.domElement);

    // Make sure 'model_compressed.ply' is in your PUBLIC folder
    viewer
      .addSplatScene('/models/testingmodel.compressed.ply', {
        progressiveLoad: true,
        streamView: true,
      })
      .then(() => viewer.start())
      .catch(console.error);

    return () => viewer.dispose();
  }, []);

  return (
    <div style={{ width: '100vw', height: '100vh', background: '#000' }}>
      <div ref={containerRef} style={{ width: '100%', height: '100%' }} />
    </div>
  );
};

export default Gaussian;
