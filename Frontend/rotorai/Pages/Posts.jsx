'use client';
import { useRouter, useSearchParams } from 'next/navigation';
import { ArrowLeft } from 'lucide-react';
import React from 'react';

const data = [
  {
    id: 1,
    title: 'Introducing RotorAI: AI-Powered Motor Inspection',
    content: './blogimg/Blue.jpg',
    text: 'RotorAI is an AI-powered inspection system designed to detect rust, cracks, and surface defects in electric motors using video analysis. By combining computer vision techniques with deep learning models such as YOLO, TensorFlow, and PyTorch, the system can identify defects that may not be visible to the naked eye. RotorAI processes live camera feeds or uploaded videos, automatically highlights potential damage, and maps detected areas onto a 3D motor visualization using Gaussian Splatting. The goal of RotorAI is to help engineers and maintenance teams perform faster, more accurate inspections while reducing manual effort and improving industrial maintenance workflows.',
    date: '2 OCOTBER 2025',
  },
  {
    id: 2,
    title: 'Announcing RUST Detection',
    content: './blogimg/Blue.jpg',
    text: 'Rotor AI is an Vision Learning Model that detects erosion where naked eye could not.',
    date: 'CBA',
  },
  {
    id: 3,
    title: 'Open Release Of V1',
    content: './blogimg/Blue.jpg',
    text: 'Rotor AI is an Vision Learning Model that detects erosion where naked eye could not.',
    date: 'CBA',
  },
  {
    id: 4,
    title: 'Announcing V1.5',
    content: './blogimg/Blue.jpg',
    text: 'Rotor AI is an Vision Learning Model that detects erosion where naked eye could not.',
    date: 'CBA',
  },
  {
    id: 5,
    title: 'V1.5 Review & Future Vision',
    content: './blogimg/Blue.jpg',
    text: 'Rotor AI is an Vision Learning Model that detects erosion where naked eye could not.',
    date: 'CBA',
  },
  {
    id: 6,
    title: 'Announcing V2',
    content: './blogimg/Blue.jpg',
    text: 'Rotor AI is an Vision Learning Model that detects erosion where naked eye could not.',
    date: 'CBA',
  },

  {
    id: 7,
    title: 'V2 Review & Future Vision',
    content: './blogimg/Blue.jpg',
    text: 'Rotor AI is an Vision Learning Model that detects erosion where naked eye could not.',
    date: 'CBA',
  },
  {
    id: 8,
    title: 'CBA',
    content: './blogimg/Blue.jpg',
    text: 'Rotor AI is an Vision Learning Model that detects erosion where naked eye could not.',
    date: 'CBA',
  },
  {
    id: 9,
    title: 'CBA',
    content: './blogimg/Blue.jpg',
    text: 'Rotor AI is an Vision Learning Model that detects erosion where naked eye could not.',
    date: 'CBA',
  },
  {
    id: 10,
    title: 'CBA',
    content: './blogimg/Blue.jpg',
    text: 'Rotor AI is an Vision Learning Model that detects erosion where naked eye could not.',
    date: 'CBA',
  },
];

const BlogPosts = () => {
  const router = useRouter();
  const searchParams = useSearchParams();
  const id = searchParams.get('id');
  const post = data.find((item) => item.id === Number(id));

  if (!post) return <div>Post not found</div>;
  return (
    <>
      <div className="w-full h-auto border-0 border-amber-300 rounded-lg p-2 text-gray-300">
        {/*change padding */}
        <div className="font-ocr mb-5 mt-5">[ {post.date} ]</div>
        <div className="font-AT text-4xl text-white mb-10">{post.title}</div>
        {/* <div className="text-base">{post.content}</div> */}
        <div className="font-Inter text-lg max-w-6xl leading-relaxed">
          {post.text}
        </div>
      </div>

      <div className="flex justify-center items-center mt-10">
        <div
          className="border-2 rounded-full px-6 py-2 cursor-pointer transition-all hover:bg-white hover:text-black font-AT"
          onClick={() => router.push('/blog')}
        >
          BACK
        </div>
      </div>
    </>
  );
};

export default BlogPosts;
