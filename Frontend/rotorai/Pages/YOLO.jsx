'use client';
import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';

const YOLO = () => {
  const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL;
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const isProcessing = useRef(false);

  useEffect(() => {
    isProcessing.current = true;
    let stream;
    const startCam = async () => {
      try {
        const mediaStream = await navigator.mediaDevices.getUserMedia({
          video: { width: 1280, height: 720 },
          audio: false,
        });

        stream = mediaStream;
        if (videoRef.current) {
          videoRef.current.srcObject = mediaStream;
          videoRef.current.onloadedmetadata = () => {
            detectFrame();
          };
        }
      } catch (error) {
        console.error('Cam deniced', error);
      }
    };

    startCam();

    return () => {
      isProcessing.current = false; // Stop the loop
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }

      if (videoRef.current) {
        videoRef.current.srcObject = null;
      }
    };
  }, []);

  const detectFrame = async () => {
    if (!videoRef.current || !canvasRef.current) return;

    //invis canvas
    const video = videoRef.current;
    if (video.readyState !== 4) {
      requestAnimationFrame(detectFrame);
      return;
    }

    //match dimensions
    if (canvasRef.current.width !== video.videoWidth) {
      canvasRef.current.width = video.videoWidth;
      canvasRef.current.height = video.videoHeight;
    }

    const offScreenCanvas = document.createElement('canvas');
    offScreenCanvas.width = video.videoWidth;
    offScreenCanvas.height = video.videoHeight;
    const ctx = offScreenCanvas.getContext('2d');
    ctx.drawImage(video, 0, 0);

    const imageBase64 = offScreenCanvas.toDataURL('image/jpeg', 0.8); // 0.8 quality for speed

    try {
      const res = await axios.post(`/api/detect`, {
        image: imageBase64,
      });

      const boxes = res.data;
      drawBoxes(boxes);
    } catch (err) {
      console.error('Detection error:', err);
    }

    // Loop: Call function again continuously
    requestAnimationFrame(detectFrame);
  };

  const drawBoxes = (boxes) => {
    if (!canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    // 1. Clear the canvas completely (Transparent)
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // 2. Draw new boxes
    boxes.forEach((b) => {
      // Increase line width for better visibility
      ctx.lineWidth = 3;

      if (b.label === 'Rust') {
        ctx.strokeStyle = '#00FF00';
        ctx.fillStyle = '#00FF00';
      } else if (b.label === 'Cracks') {
        ctx.strokeStyle = '#00FFFF';
        ctx.fillStyle = '#00FFFF';
      } else {
        ctx.strokeStyle = 'white';
        ctx.fillStyle = 'white';
      }

      ctx.strokeRect(b.x, b.y, b.w, b.h);

      ctx.font = 'bold 18px Arial';
      ctx.fillText(b.label, b.x, b.y - 10);
    });
  };

  return (
    <div className="flex flex-col items-center justify-center h-screen-full bg-black mt-10">
      <div className="relative w-[1200px] rounded-xl overflow-hidden shadow-2xl shadow-white/10 ring-2 ring-white/10">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="w-full h-full object-cover overflow-hidden shadow-2xl shadow-white/10 ring-2 ring-white/10"
        />
        <canvas
          ref={canvasRef}
          className="absolute top-0 left-0 w-full h-full pointer-events-none overflow-hidden shadow-2xl shadow-white/10 ring-2 ring-white/10"
        />
      </div>
    </div>
  );
};

export default YOLO;
