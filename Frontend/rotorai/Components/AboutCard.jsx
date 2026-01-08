'use client';
import React from 'react';

const AboutCard = ({ text1, text2 }) => {
  return (
    <div className="w-full max-w-xl h-full max-h-auto rounded-xl bg-[#0a0a0a] backdrop-blur-3xl shadow-xl shadow-white/10 flex flex-col items-center justify-center ring-2 ring-white/10 transition-all p-6">
      {text1 && (
        <>
          <div className="font-AT text-xl mb-4">{text1}</div>
        </>
      )}
      {text2 && (
        <>
          <div className="font-Inter text-base text-gray-300">{text2}</div>
        </>
      )}
    </div>
  );
};

export default AboutCard;
