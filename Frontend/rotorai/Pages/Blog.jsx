'use client';
import React, { useState, useRef, useEffect } from 'react';
import FullBleedDivider from '@/Components/FullBleedDivider';
import BlogPosts from '@/Components/BlogPosts';
import { useRouter } from 'next/navigation';

const Data = [
  {
    id: 1,
    title: 'Project Initiation – RotorAI Concept',
    image: './blogimg/Blue.jpg',
    text: 'Concept of an AI system for automated rust and defect detection.',
    date: '15 SEPTEMBER 2025',
  },
  {
    id: 2,
    title: 'Prototype – Rust Detection (v0.1)',
    image: './blogimg/Silver.jpg',
    text: 'First prototype for rust detection using deep learning.',
    date: '9 OCTOBER 2025',
  },
  {
    id: 3,
    title: 'RotorAI v1 Release',
    image: './blogimg/Leviticus.png',
    text: 'Initial release with basic image-based defect detection.',
    date: '30 OCTOBER 2025',
  },
  {
    id: 4,
    title: 'v1.5 – Dataset Expansion',
    image: './blogimg/Deuteronomy.png',
    text: 'Improved rust detection; crack limitations identified.',
    date: '16 NOVEMBER 2025',
  },
  {
    id: 5,
    title: 'Dataset Strategy Shift',
    image: './blogimg/Kings.png',
    text: 'Switched datasets to improve generalisation.',
    date: '25 NOVEMBER 2025',
  },
  {
    id: 6,
    title: 'RotorAI v2 – Dual Detection',
    image: './blogimg/Chronicles.png',
    text: 'Introduced combined rust and crack detection.',
    date: '10 DECEMBER 2025',
  },
  {
    id: 7,
    title: 'Real-Time Detection',
    image: './blogimg/Psalms.jpg',
    text: 'Enabled real-time detection with YOLOv8.',
    date: '5 JANUARY 2026',
  },
  {
    id: 8,
    title: '3D Gaussian Splatting',
    image: './blogimg/Romans.png',
    text: 'Added 3D spatial visualisation.',
    date: '20 FEBRUARY 2026',
  },
  {
    id: 9,
    title: 'RaySplat Mapping',
    image: './blogimg/Thessalonians.jpg',
    text: 'Mapped detections into 3D using RaySplat.',
    date: '13 MARCH 2026',
  },
  {
    id: 10,
    title: 'Final System',
    image: '/blogimg/Genesis.png',
    text: 'Completed full AI, real-time, and 3D system.',
    date: '29 APRIL 2026',
  },
];

const ButtonHelper = ({}) => {
  return (
    <>
      <div className="border-2 rounded-3xl w-fit px-2 font-Inter text-sm transition-all group-hover:bg-white group-hover:text-black mt-2">
        <div className="p-2">OPEN</div>
      </div>
    </>
  );
};

const BigFeaturePost = ({ post }) => {
  if (!post) return null;
  const router = useRouter();
  return (
    <div
      className="group w-full h-[300px] border-0 border-amber-300 hover:cursor-pointer transition-all hover:scale-[1.02] hover:shadow-xl"
      onClick={() => router.push(`/blog/posts?id=10`)}
    >
      {/*3 columns using flex */}
      <div className="flex flex-row h-full font-Inter ">
        <div className="flex items-start justify-center text-lg border-0 border-amber-300 mr-5">
          <div className="text-lg font-ocr text-gray-400/50">
            {post.date || '-'}
          </div>
        </div>
        <div className="relative flex flex-col items-start justify-starts text-lg border-0 border-amber-300 px-15 mr-20 w-155 max-w-full">
          <div className="text-xl">{post.title}</div>
          <div className="text-gray-300 text-base mt-10">{post.text}</div>
          <div className=" absolute bottom-4 right-4">
            <ButtonHelper />
          </div>
        </div>
        <div className="flex items-center justify-center border-0 border-amber-300 h-full">
          <div className="h-full w-full flex items-center justify-center">
            <img
              src={post.image}
              alt={post.title}
              className="h-full w-full object-contain border-0 border-amber-300 scale-110 rounded-lg"
            />
          </div>
        </div>
      </div>
    </div>
  );
};

const Blog = () => {
  const sorted = [...Data].sort((a, b) => b.id - a.id);
  const [latestPost, ...restPosts] = sorted;
  return (
    <div className="mt-10">
      <div className="font-ocr text-gray-400/50 text-2xl mb-5">
        [ WHAT'S NEW ]
      </div>

      <div className="w-full h-[200px] border-0 border-amber-300 font-Inter text-lg">
        <div className="grid grid-cols-2">
          <div className="flex items-start justify-start text-5xl py-30">
            Latest News
          </div>
          <div className="flex items-center justify-center text-lg py-30">
            Read about our latest product
            <br /> and research annoucements.
          </div>
        </div>
      </div>

      <div className="w-full h-[0.1px] mt-15 mb-15 bg-white/20" />

      <BigFeaturePost post={latestPost} />

      <div className="w-full h-[0.1px] mt-15 mb-15 bg-white/20" />

      {/*Nx3 grid map this*/}
      <div className="grid grid-cols-3 items-center justify-between gap-6">
        {[...Data]
          .reverse()
          .slice(1)
          .map((item) => (
            <div key={item.id}>
              <BlogPosts
                title={item.title}
                image={item.image}
                text={item.text}
                date={item.date}
                postID={item.id}
              />
            </div>
          ))}
        {/* <div className="w-full h-[450px] border-2 border-amber-300" />
        <div className="w-full h-[450px] border-2 border-amber-300" />
        <div className="w-full h-[450px] border-2 border-amber-300" />
        <div className="w-full h-[450px] border-2 border-amber-300" />
        <div className="w-full h-[450px] border-2 border-amber-300" />
        <div className="w-full h-[450px] border-2 border-amber-300" />
        <div className="w-full h-[450px] border-2 border-amber-300" />
        <div className="w-full h-[450px] border-2 border-amber-300" />
        <div className="w-full h-[450px] border-2 border-amber-300" /> */}
      </div>

      <FullBleedDivider />
    </div>
  );
};

export default Blog;
