'use client';
import React, { useState, useRef, useEffect } from 'react';
import FullBleedDivider from '@/Components/FullBleedDivider';
import BlogPosts from '@/Components/BlogPosts';
// add /Blog

const Data = [
  {
    id: 1,
    title: 'Announcing RotorAI',
    image: './blogimg/Blue.jpg',
    text: 'Rotor AI is an Vision Learning Model that detects erosion where naked eye could not.',
    date: '2 OCOTBER 2025',
  },
  {
    id: 2,
    title: 'Introducing Rust detection',
    image: './blogimg/Silver.jpg',
    text: 'Rust Detection is out and ready for testing.',
    date: '9 NOVEMBER 2025',
  },
  {
    id: 3,
    title: 'Open Release of V1',
    image: './blogimg/Leviticus.png',
    text: 'RotorAI v1 is out.',
    date: '30 NOVEMBER 2025',
  },
  {
    id: 4,
    title: 'Annoucing V1.5',
    image: './blogimg/Deuteronomy.png',
    text: 'Better Rust detection with DACL10k dataset but cracks cannot be detected.',
    date: '16 DECEMBER 2025',
  },
  {
    id: 5,
    title: 'V1.5 Review and Future Vision',
    image: './blogimg/Kings.png',
    text: 'Scrapping DACL10k dataset and using 3 Roboflow dataset of corrosion.',
    date: '25 DECEMBER 2026',
  },
  {
    id: 6,
    title: 'Annoucing V2',
    image: './blogimg/Chronicles.png',
    text: 'V2 is out with Corrosion detection and Cracks detection.',
    date: '16 JANUARY 2026',
  },
  {
    id: 7,
    title: 'V2 Review and Future Vision',
    image: './blogimg/Psalms.jpg',
    text: 'TBA',
    date: '23 JANUARY 2026',
  },
  {
    id: 8,
    title: 'Romans',
    image: './blogimg/Romans.png',
    text: 'TBA',
    date: 'DD MMOONNTTHH YEAR',
  },
  {
    id: 9,
    title: 'Thessalonians',
    image: './blogimg/Thessalonians.jpg',
    text: 'TBA',
    date: 'DD MMOONNTTHH YEAR',
  },
  {
    id: 10,
    title: 'Placeholder title',
    image: '/blogimg/Genesis.png',
    text: 'Sed ut perspiciatis estiae consequatur, vel illum qui dolorem eum fugiat quo voluptas nulla pariatur?',
    date: 'DD MMOONNTTHH YEAR',
  },
];

const BigFeaturePost = ({ post }) => {
  if (!post) return null;
  return (
    <div
      className="w-full h-[300px] border-0 border-amber-300 hover:cursor-pointer"
      // onClick={}
    >
      {/*3 columns using flex */}
      <div className="flex flex-row h-full font-Inter ">
        <div className="flex items-start justify-center text-lg border-0 border-amber-300 mr-5">
          <div className="text-lg font-ocr text-gray-400/50">
            {post.date || '-'}
          </div>
        </div>
        <div className="flex flex-col items-start justify-starts text-lg border-0 border-amber-300 px-15 mr-20 w-155 max-w-full">
          <div className="text-xl">{post.title}</div>
          <div className="text-gray-300 text-base mt-10">{post.text}</div>
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
