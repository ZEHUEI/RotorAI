'use client';
import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { video } from 'framer-motion/client';

const YOLO = () => {
  const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL;
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

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

  useEffect(() => {
    const loop = setInterval(async () => {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');

      if (!video.videoWidth) return;

      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      ctx.drawImage(video, 0, 0);

      const image = canvas.toDataURL('image/jpeg');

      const res = await axios.post(`${BACKEND_URL}/detect`, {
        image: image,
      });

      const boxes = res.data;

      boxes.forEach((b) => {
        if (b.label === 'Rust') {
          ctx.strokeStyle = 'lime';
          ctx.fillStyle = 'lime';
        } else if (b.label === 'Cracks') {
          ctx.strokeStyle = 'cyan';
          ctx.fillStyle = 'cyan';
        } else {
          ctx.strokeStyle = 'white';
          ctx.fillStyle = 'white';
        }

        ctx.strokeRect(b.x, b.y, b.w, b.h);
        ctx.fillText(b.label, b.x, b.y - 5);
      });
    }, 200);

    return () => clearInterval(loop);
  }, [BACKEND_URL]);

  return (
    <div className="flex flex-col items-center justify-center h-screen-full bg-black text-white mt-10">
      <video
        ref={videoRef}
        autoPlay
        playsInline
        className="w-[1200px] rounded-lg border-2 border-white"
      />
      <canvas
        ref={canvasRef}
        className="absolute w-[1200px] rounded-lg border-2"
      />
    </div>
  );
};

export default YOLO;
