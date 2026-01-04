'use client';
import React, { useState, useEffect } from 'react';

const BlogImage = () => {
  return (
    <div className="h-full w-full flex items-center justify-center">
      <img
        src="/blogimg/Genesis.png"
        alt="Genesis"
        className="h-full w-full object-contain border-0 border-amber-300 scale-110"
      />
    </div>
  );
};

export default BlogImage;
