'use client';
import React from 'react';

const ImageCard = ({ image, title }) => {
  return (
    <>
      <img
        src={image}
        alt={title}
        className="object-cover w-full h-64 rounded-t-lg"
      />
    </>
  );
};

const BlogPosts = ({ title, image, text, date }) => {
  return (
    <div className="w-full h-[450px] border-0 border-amber-300 rounded-lg bg-[#0a0a0a] backdrop-blur-3xl shadow-xl shadow-white/10 ring-2 ring-white/10 transition-all p-2">
      <div className="w-full hover:scale-[1.02] transition">
        <ImageCard image={image} title={title} />
        <h2 className="text-xl font-AT mb-2">{title}</h2>
        <span className="text-sm font-ocr">{date}</span>
        <p className="text-base text-gray-300 font-Inter mt-4">{text}</p>
      </div>
    </div>
  );
};

export default BlogPosts;
