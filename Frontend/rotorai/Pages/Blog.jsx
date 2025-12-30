'use client';
import React from 'react';
import FullBleedDivider from '@/Components/FullBleedDivider';
// add /Blog

const Data = [];

const Blog = () => {
  return (
    <div className="mt-10">
      <div className="font-ocr text-gray-400/50 text-2xl mb-5">
        [ WHAT'S NEW ]
      </div>

      <div className="w-full h-[300px] border-2 border-amber-300"></div>

      <div className="w-full h-[0.1px] mt-15 mb-15 bg-white/20" />

      <div className="w-full h-[200px] border-2 border-amber-300">
        Newest news
      </div>

      <div className="w-full h-[0.1px] mt-15 mb-15 bg-white/20" />

      {/*Nx3 grid */}
      <div className="grid grid-cols-3 items-center justify-between gap-12">
        <div className="w-full h-[450px] border-2 border-amber-300" />
        <div className="w-full h-[450px] border-2 border-amber-300" />
        <div className="w-full h-[450px] border-2 border-amber-300" />
        <div className="w-full h-[450px] border-2 border-amber-300" />
        <div className="w-full h-[450px] border-2 border-amber-300" />
        <div className="w-full h-[450px] border-2 border-amber-300" />
        <div className="w-full h-[450px] border-2 border-amber-300" />
        <div className="w-full h-[450px] border-2 border-amber-300" />
        <div className="w-full h-[450px] border-2 border-amber-300" />
      </div>

      <FullBleedDivider />
    </div>
  );
};

export default Blog;
