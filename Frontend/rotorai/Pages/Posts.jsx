'use client';
import { useRouter, useSearchParams } from 'next/navigation';
import React from 'react';

const data = [
  {
    id: 1,
    title: 'Announcing RotorAI',
    content: './blogimg/Blue.jpg',
    text: 'Rotor AI is an Vision Learning Model that detects erosion where naked eye could not.',
    date: '2 OCOTBER 2025',
  },
  {
    id: 2,
    title: 'Announcing RotorAI',
    content: './blogimg/Blue.jpg',
    text: 'Rotor AI is an Vision Learning Model that detects erosion where naked eye could not.',
    date: '2 OCOTBER 2025',
  },
  {
    id: 3,
    title: 'Announcing RotorAI',
    content: './blogimg/Blue.jpg',
    text: 'Rotor AI is an Vision Learning Model that detects erosion where naked eye could not.',
    date: '2 OCOTBER 2025',
  },
  {
    id: 4,
    title: 'Announcing RotorAI',
    content: './blogimg/Blue.jpg',
    text: 'Rotor AI is an Vision Learning Model that detects erosion where naked eye could not.',
    date: '2 OCOTBER 2025',
  },
  {
    id: 5,
    title: 'Announcing RotorAI',
    content: './blogimg/Blue.jpg',
    text: 'Rotor AI is an Vision Learning Model that detects erosion where naked eye could not.',
    date: '2 OCOTBER 2025',
  },
  {
    id: 6,
    title: 'Announcing RotorAI',
    content: './blogimg/Blue.jpg',
    text: 'Rotor AI is an Vision Learning Model that detects erosion where naked eye could not.',
    date: '2 OCOTBER 2025',
  },

  {
    id: 7,
    title: 'Announcing RotorAI',
    content: './blogimg/Blue.jpg',
    text: 'Rotor AI is an Vision Learning Model that detects erosion where naked eye could not.',
    date: '2 OCOTBER 2025',
  },
  {
    id: 8,
    title: 'Announcing RotorAI',
    content: './blogimg/Blue.jpg',
    text: 'Rotor AI is an Vision Learning Model that detects erosion where naked eye could not.',
    date: '2 OCOTBER 2025',
  },
  {
    id: 9,
    title: 'Announcing RotorAI',
    content: './blogimg/Blue.jpg',
    text: 'Rotor AI is an Vision Learning Model that detects erosion where naked eye could not.',
    date: '2 OCOTBER 2025',
  },
  {
    id: 10,
    title: 'Announcing RotorAI',
    content: './blogimg/Blue.jpg',
    text: 'Rotor AI is an Vision Learning Model that detects erosion where naked eye could not.',
    date: '2 OCOTBER 2025',
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
      <div
        className="w-full h-[450px] border-2 border-amber-300 rounded-lg bg-[#0a0a0a] backdrop-blur-3xl shadow-xl shadow-white/10 ring-2 ring-white/10 transition-all p-2 hover:cursor-pointer"
        onClick={() => router.push('/blog')}
      >
        {/*change padding */}
        <div>{post.content}</div>
        <div>{post.id}</div>
        <div>{post.text}</div>
        <div>{id}</div>
      </div>
    </>
  );
};

export default BlogPosts;
