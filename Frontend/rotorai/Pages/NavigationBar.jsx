'use client';
import React, { useState, useEffect, useRef, useMemo } from 'react';
import Main from '@/Pages/Main';
import Blog from '@/Pages/Blog';
import About from '@/Pages/About';

const NavigationBar = () => {
  const [current, setCurrent] = useState('ROTORAI');
  const items = ['ROTORAI', 'ABOUT', 'BLOG'];
  const handleClick = (index) => {
    setCurrent(items[index]);
  };

  useEffect(() => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }, [current]);

  return (
    <>
      <div className="fixed top-0 left-0 right-0 z-50 w-full backdrop-blur-lg overflow-hidden text-2xl">
        <div className="flex flex-row gap-14 px-1.5 md:px-20 lg:px-40 xl:px-40 2xl:px-40 py-4 font-[AT]">
          {items.map((item, index) => (
            <button
              key={item}
              onClick={() => handleClick(index)}
              className="cursor-pointer"
            >
              <span
                className={`relative ${current === item ? 'text-[#FFFFFF]' : 'hover:text-[#FFFFFF] hover:underline underline-offset-3'}`}
              >
                {item}
              </span>
            </button>
          ))}
        </div>
      </div>

      <div className="pt-20">
        {current === 'ROTORAI' && <Main />}
        {current === 'ABOUT' && <About />}
        {current === 'BLOG' && <Blog />}
      </div>
    </>
  );
};

export default NavigationBar;
