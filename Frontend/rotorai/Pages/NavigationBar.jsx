'use client';
import React, { useState, useEffect, useRef, useMemo } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';

const NavigationBar = () => {
  const pathname = usePathname();

  const items = [
    { label: 'ROTORAI', href: '/' },
    { label: 'ABOUT', href: '/about' },
    { label: 'BLOG', href: '/blog' },
  ];

  useEffect(() => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }, [pathname]);

  return (
    <>
      <div className="fixed top-0 left-0 right-0 z-50 w-full backdrop-blur-lg overflow-hidden text-2xl">
        <div className="flex flex-row gap-14 px-1.5 md:px-20 lg:px-40 xl:px-40 2xl:px-40 py-4 font-[AT]">
          {items.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              scroll={false}
              className={`relative
                ${pathname === item.href ? 'text-[#FFFFFF]' : 'hover:text-[#FFFFFF] hover:underline underline-offset-3'}`}
            >
              {item.label}
            </Link>
          ))}
        </div>
      </div>
    </>
  );
};

export default NavigationBar;
