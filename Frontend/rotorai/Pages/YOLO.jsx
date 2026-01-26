'use client';
import React, { useState, useEffect, useRef } from 'react';

const YOLO = () => {
  const videoRef = useRef(null);

  useEffect(() => {
    const startCam = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: true,
          audio: false,
        });

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (error) {
        console.error('Cam deniced', error);
      }
    };

    startCam();

    return () => {
      if (videoRef.current?.srcObject) {
        const tracks = videoRef.current.srcObject.getTracks();
        tracks.forEach((track) => track.stop());
      }
    };
  }, []);

  return (
    <div className="flex flex-col items-center justify-center h-screen bg-black text-white">
      <h1 className="mb-4 text-xl">Camera Test</h1>
      <video
        ref={videoRef}
        autoPlay
        playsInline
        className="w-[400px] rounded-lg border-2 border-white"
      />
    </div>
  );
};

export default YOLO;
