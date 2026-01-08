'use client';
import React from 'react';
import FullBleedDivider from '@/Components/FullBleedDivider';
import BlogImage from '@/Components/BlogImage';
import BlogPosts from '@/Components/BlogPosts';
// add /Blog

const Data = [
  {
    id: 1,
    title: 'Genesis',
    image: './blogimg/Genesis.png',
    text: '',
    date: '',
  },
  {
    id: 2,
    title: 'Exodus',
    image: './blogimg/Genesis.png',
    text: '',
    date: '',
  },
  {
    id: 3,
    title: 'Leviticus',
    image: './blogimg/Genesis.png',
    text: '',
    date: '',
  },
  {
    id: 4,
    title: 'Deuteronomy',
    image: './blogimg/Genesis.png',
    text: '',
    date: '',
  },
  {
    id: 5,
    title: 'Kings',
    image: './blogimg/Genesis.png',
    text: '',
    date: '',
  },
  {
    id: 6,
    title: 'Chronicles',
    image: './blogimg/Genesis.png',
    text: '',
    date: '',
  },
  {
    id: 7,
    title: 'Psalms',
    image: './blogimg/Genesis.png',
    text: '',
    date: '',
  },
  {
    id: 8,
    title: 'Romans',
    image: './blogimg/Genesis.png',
    text: '',
    date: '',
  },
  {
    id: 9,
    title: 'Thessalonians',
    image: './blogimg/Genesis.png',
    text: '',
    date: '',
  },
  {
    id: 10,
    title: 'Revelation',
    image: './blogimg/Genesis.png',
    text: '',
    date: '',
  },
];

const Blog = () => {
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

      <div
        className="w-full h-[300px] border-0 border-amber-300 hover:cursor-pointer"
        // onClick={}
      >
        {/*3 columns using flex */}
        <div className="flex flex-row h-full font-Inter ">
          <div className="flex items-start justify-center text-lg border-0 border-amber-300 mr-5">
            <div className="text-lg font-ocr text-gray-400/50">
              December 20, 2025
            </div>
          </div>
          <div className="flex flex-col items-start justify-starts text-lg border-0 border-amber-300 px-15 mr-20">
            <div className="text-xl">
              Introducing Grok Business and Grok Enterprise
            </div>
            <div className="text-gray-300 text-base mt-10">
              THe best assistant in the world is now Enterprise ready.
            </div>
          </div>
          <div className="flex items-center justify-center border-0 border-amber-300 h-full">
            <BlogImage />
          </div>
        </div>
      </div>

      <div className="w-full h-[0.1px] mt-15 mb-15 bg-white/20" />

      {/*Nx3 grid map this*/}
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
