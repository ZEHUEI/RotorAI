'use client';
import { Suspense } from 'react';
import BlogIndividual from '@/Pages/Posts';

export default function BlogPage() {
  return (
    <main className="relative px-1.5 md:px-20 lg:px-40 text-[#D3D1CE]">
      <Suspense fallback={<div>Loading...</div>}>
        <BlogIndividual />
      </Suspense>
    </main>
  );
}
